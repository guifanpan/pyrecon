import numpy as np
from scipy import ndimage

from pyrecon import IterativeFFTReconstruction, IterativeFFTReconstruction2LPT, MultiGridReconstruction2LPT
from pyrecon.multigrid import OriginalMultiGridReconstruction
from pyrecon import _multigrid


class FourierBox(object):

    def __init__(self, nmesh, boxsize):
        self.nmesh, self.boxsize = nmesh, boxsize
        k1d = 2 * np.pi * np.fft.fftfreq(nmesh, d=boxsize / nmesh)
        kz1d = 2 * np.pi * np.fft.rfftfreq(nmesh, d=boxsize / nmesh)
        self.k = np.meshgrid(k1d, k1d, kz1d, indexing='ij')
        self.k2 = sum(kk**2 for kk in self.k)
        self.k2[0, 0, 0] = 1.
        self.nyq = [np.abs(kk) == np.pi * nmesh / boxsize for kk in self.k]

    def r2c(self, x):
        return np.fft.rfftn(x)

    def c2r(self, x):
        return np.fft.irfftn(x, s=(self.nmesh,) * 3, axes=(0, 1, 2))

    def gaussian_field(self, kcut, sigma, seed=42):
        rng = np.random.default_rng(seed)
        delta_k = self.r2c(rng.standard_normal((self.nmesh,) * 3)) * np.where(self.k2**0.5 < kcut, self.k2**(-3. / 8.), 0.)
        delta_k[0, 0, 0] = 0.
        delta = self.c2r(delta_k)
        return delta * sigma / delta.std()

    def grad_inv_lap(self, x):
        x_k = self.r2c(x)
        toret = []
        for iaxis in range(3):
            tmp = -1j * self.k[iaxis] * x_k / self.k2
            tmp[self.nyq[iaxis]] = 0.
            toret.append(self.c2r(tmp))
        return np.array(toret)

    def tidal_source(self, delta):
        delta_k = self.r2c(delta)
        h = {(i, j): self.c2r(self.k[i] * self.k[j] / self.k2 * delta_k) for i in range(3) for j in range(i, 3)}
        return h[0, 0] * h[1, 1] - h[0, 1]**2 + h[0, 0] * h[2, 2] - h[0, 2]**2 + h[1, 1] * h[2, 2] - h[1, 2]**2


def get_shifts(cls, delta, boxsize, f=0., los='x', field='disp', **kwargs):
    nmesh = delta.shape[0]
    recon = cls(f=f, bias=1., los=los, nmesh=nmesh, boxsize=boxsize, boxcenter=boxsize / 2., **kwargs)
    recon.mesh_delta = recon.pm.create(type='real')
    recon.mesh_delta.value[...] = delta
    recon.run()
    positions = np.moveaxis(np.indices((nmesh,) * 3), 0, -1).reshape(-1, 3) * (boxsize / nmesh)  # mesh nodes
    return np.moveaxis(recon.read_shifts(positions, field=field).reshape((nmesh,) * 3 + (3,)), -1, 0), recon


def test_second_order_term():
    # Psi(2LPT) - Psi(Zeldovich) must be grad lap^{-1} S, with coefficient exactly 1
    nmesh, boxsize = 64, 500.
    box = FourierBox(nmesh, boxsize)
    delta = box.gaussian_field(kcut=0.06, sigma=0.1)
    ref = box.grad_inv_lap(box.tidal_source(delta))
    for cls1, cls2, rtol in [(IterativeFFTReconstruction, IterativeFFTReconstruction2LPT, 1e-3),
                             (OriginalMultiGridReconstruction, MultiGridReconstruction2LPT, 2e-2)]:
        shifts1 = get_shifts(cls1, delta, boxsize)[0]
        shifts2 = get_shifts(cls2, delta, boxsize)[0]
        diff = shifts2 - shifts1
        coeff = np.sum(diff * ref) / np.sum(ref * ref)
        assert abs(coeff - 1.) < rtol, (cls2.__name__, coeff)
        assert np.corrcoef(diff.ravel(), ref.ravel())[0, 1] > 0.999


def test_rsd():
    # RSD displacement is f * (psi1 + (1 + D2) * psi2) along the line-of-sight
    nmesh, boxsize, f, D2 = 64, 500., 0.8, -0.4
    box = FourierBox(nmesh, boxsize)
    delta = box.gaussian_field(kcut=0.06, sigma=0.1)
    for cls in [IterativeFFTReconstruction2LPT, MultiGridReconstruction2LPT]:
        disp, recon = get_shifts(cls, delta, boxsize, f=f, field='disp', D2=D2)
        positions = np.moveaxis(np.indices((nmesh,) * 3), 0, -1).reshape(-1, 3) * (boxsize / nmesh)
        if hasattr(recon, 'mesh_psi'):
            psi1 = np.array([recon._readout(psi, positions) for psi in recon.mesh_psi])
        else:
            psi1 = _multigrid.read_finite_difference_cic(recon.mesh_phi, positions, recon.boxcenter).T
        psi1 = psi1.reshape(disp.shape)
        rsd = recon.read_shifts(positions, field='rsd').T.reshape(disp.shape)
        assert np.allclose(rsd[1:], 0.)
        assert np.allclose(rsd[0], f * (psi1[0] + (1. + D2) * (disp[0] - psi1[0])))
        assert np.allclose(recon.read_shifts(positions, field='disp+rsd').T.reshape(disp.shape), disp + rsd)


def test_exact_inversion():
    # Exact Eulerian density from a known displacement field: solve q = x - Psi(q) at each mesh node x, then 1 + delta = 1 / J(q).
    # The 2LPT reconstructed displacement must match Psi(q(x)) much better than the Zeldovich one.
    nmesh, boxsize, D2 = 64, 1000., -3. / 7.
    box = FourierBox(nmesh, boxsize)
    delta_lin = box.gaussian_field(kcut=0.04, sigma=0.05)
    psi = -box.grad_inv_lap(delta_lin) + D2 * box.grad_inv_lap(box.tidal_source(delta_lin))
    psi_k = [box.r2c(p) for p in psi]
    dpsi = [[box.c2r(1j * box.k[j] * psi_k[i]) for j in range(3)] for i in range(3)]
    cellsize = boxsize / nmesh
    coeffs = [ndimage.spline_filter(p / cellsize, order=3, mode='grid-wrap') for p in psi]
    grid = np.indices((nmesh,) * 3).astype('f8')
    q = grid.copy()
    for it in range(50):
        psi_q = np.array([ndimage.map_coordinates(c, q, order=3, mode='grid-wrap', prefilter=False) for c in coeffs])
        q_new = grid - psi_q
        converged = np.abs(q_new - q).max() < 1e-10
        q = q_new
        if converged: break
    assert converged
    psi_eulerian = psi_q * cellsize
    jac = np.array([[ndimage.map_coordinates(d, q, order=3, mode='grid-wrap') for d in row] for row in dpsi])
    jac = np.linalg.det(np.moveaxis(np.eye(3)[:, :, None, None, None] + jac, (0, 1), (-2, -1)))
    delta = 1. / jac - 1.
    delta -= delta.mean()

    def residual(cls):
        shifts = get_shifts(cls, delta, boxsize)[0]
        diff = (shifts - shifts.mean(axis=(1, 2, 3), keepdims=True)) - (psi_eulerian - psi_eulerian.mean(axis=(1, 2, 3), keepdims=True))
        return np.sqrt(np.mean(diff**2))

    residual1, residual2 = residual(IterativeFFTReconstruction), residual(IterativeFFTReconstruction2LPT)
    # residuals are 2nd and 3rd order, respectively; expected ratio ~ 0.05 for sigma = 0.05
    assert residual2 < 0.15 * residual1, (residual1, residual2)


if __name__ == '__main__':

    test_second_order_term()
    test_rsd()
    test_exact_inversion()
