"""Implementation of Burden et al. 2015 (https://arxiv.org/abs/1504.02591) algorithm."""

from .recon import BaseReconstruction
from . import utils


class IterativeFFTReconstruction(BaseReconstruction):
    """
    Implementation of Burden et al. 2015 (https://arxiv.org/abs/1504.02591)
    field-level (as opposed to :class:`IterativeFFTParticleReconstruction`) algorithm.
    """
    _compressed = True
    _f_z = True
    _bias_z = True

    def run(self, niterations=3):
        """
        Run reconstruction, i.e. compute Zeldovich displacement fields :attr:`mesh_psi`.

        Parameters
        ----------
        niterations : int, default=3
            Number of iterations.
        """
        self._iter = 0
        self.mesh_delta_real = self.mesh_delta.copy()
        for iter in range(niterations):
            self._iterate()
        del self.mesh_delta
        self.mesh_psi = self._compute_psi()
        del self.mesh_delta_real

    def _iterate(self):
        if self.mpicomm.rank == 0:
            self.log_info('Running iteration {:d}.'.format(self._iter))
        # This is an implementation of eq. 22 and 24 in https://arxiv.org/pdf/1504.02591.pdf
        # \delta_{g,\mathrm{real},n} is self.mesh_delta_real
        # \delta_{g,\mathrm{red}} is self.mesh_delta
        # First compute \delta(k)/k^{2} based on current \delta_{g,\mathrm{real},n} to estimate \phi_{\mathrm{est},n} (eq. 24)
        delta_k = self.mesh_delta_real.r2c()
        for kslab, slab in zip(delta_k.slabs.x, delta_k.slabs):
            utils.safe_divide(slab, sum(kk**2 for kk in kslab), inplace=True)

        self.mesh_delta_real = self.mesh_delta.copy()
        # Now compute \beta \nabla \cdot (\nabla \phi_{\mathrm{est},n} \cdot \hat{r}) \hat{r}
        # In the plane-parallel case (self.los is a given vector), this is simply \beta IFFT((\hat{k} \cdot \hat{\eta})^{2} \delta(k))
        if self.los is not None:
            # global los
            disp_deriv_k = delta_k.copy()
            for kslab, slab in zip(disp_deriv_k.slabs.x, disp_deriv_k.slabs):
                slab[...] *= sum(kk * ll for kk, ll in zip(kslab, self.los))**2  # delta_k already divided by k^{2}
            factor = self.beta
            # remove RSD part
            if self._iter == 0:
                # Burden et al. 2015: 1504.02591, eq. 12 (flat sky approximation)
                factor /= (1. + self.beta)
            self.mesh_delta_real -= factor * disp_deriv_k.c2r()
            del disp_deriv_k
        else:
            # In the local los case, \beta \nabla \cdot (\nabla \phi_{\mathrm{est},n} \cdot \hat{r}) \hat{r} is:
            # \beta \partial_{i} \partial_{j} \phi_{\mathrm{est},n} \hat{r}_{j} \hat{r}_{i}
            # i.e. \beta IFFT(k_{i} k_{j} \delta(k) / k^{2}) \hat{r}_{i} \hat{r}_{j} => 6 FFTs
            for iaxis in range(delta_k.ndim):
                for jaxis in range(iaxis, delta_k.ndim):
                    disp_deriv = delta_k.copy()
                    for kslab, islab, slab in zip(disp_deriv.slabs.x, disp_deriv.slabs.i, disp_deriv.slabs):
                        mask = (islab[iaxis] != self.nmesh[iaxis] // 2) & (islab[jaxis] != self.nmesh[jaxis] // 2)
                        mask |= (islab[iaxis] == self.nmesh[iaxis] // 2) & (islab[jaxis] == self.nmesh[jaxis] // 2)
                        slab[...] *= kslab[iaxis] * kslab[jaxis] * mask  # delta_k already divided by k^{2}
                    disp_deriv = disp_deriv.c2r()
                    for rslab, slab in zip(disp_deriv.slabs.x, disp_deriv.slabs):
                        rslab = self._transform_rslab(rslab)
                        slab[...] *= utils.safe_divide(rslab[iaxis] * rslab[jaxis], sum(rr**2 for rr in rslab))
                    factor = (1. + (iaxis != jaxis)) * self.beta  # we have j >= i and double-count j > i to account for j < i
                    if self._iter == 0:
                        # Burden et al. 2015: 1504.02591, eq. 12 (flat sky approximation)
                        factor /= (1. + self.beta)
                    # remove RSD part
                    self.mesh_delta_real -= factor * disp_deriv
        self._iter += 1

    def _compute_psi(self):
        # Compute Zeldovich displacements given reconstructed real space density
        delta_k = self.mesh_delta_real.r2c()
        psis = []
        for iaxis in range(delta_k.ndim):
            psi = delta_k.copy()
            for kslab, islab, slab in zip(psi.slabs.x, psi.slabs.i, psi.slabs):
                mask = islab[iaxis] != self.nmesh[iaxis] // 2
                slab[...] *= 1j * utils.safe_divide(kslab[iaxis], sum(kk**2 for kk in kslab)) * mask
            psis.append(psi.c2r())
            del psi
        return psis


class IterativeFFTReconstruction2LPT(IterativeFFTReconstruction):
    """
    Extension of IterativeFFTReconstruction that includes second-order LPT
    (tidal field) corrections in the displacement field.

    The iterative RSD correction is identical to the original algorithm; only
    the final displacement computation is augmented with the second-order
    potential.

    Parameters
    ----------
    D2 : float, optional
        Second-order growth factor (typically negative, e.g. -0.3).
        If not provided, computed from omega_m using the EdS-like approximation.
    omega_m : float, default=0.31
        Matter density parameter, used to estimate D2 when not given.
    Other parameters are passed to IterativeFFTReconstruction.
    """
    def __init__(self, *args, D2=None, omega_m=0.31, **kwargs):
        super().__init__(*args, **kwargs)
        if D2 is None:
            # Buchert & Ehlers (1993) approximate formula for LCDM
            self.D2 = -3.0 / 7.0 * omega_m ** (-1.0 / 143.0)
        else:
            self.D2 = float(D2)
        self.omega_m = omega_m

    def _compute_psi(self):
        """
        Compute Zeldovich (1st-order) + tidal (2nd-order) displacement fields.

        Overrides the parent method to include the second-order correction.
        The workflow:
            1. Compute first-order displacement as before.
            2. Compute first-order potential phi1 from the real-space density.
            3. Compute the tidal source delta2 = sum_{i>j}(phi_ii*phi_jj - phi_ij^2).
            4. Solve for second-order potential phi2 (Poisson eq. with source delta2).
            5. Add D2 * (-grad phi2) to the displacement.

        Returns
        -------
        psis : list of RealField
            List of three real-space displacement fields (x, y, z components).
        """
        # ---- Step 1: Get the real-space density in Fourier space ----
        delta_k = self.mesh_delta_real.r2c()   # already divided by bias

        # ---- Step 2: Compute first-order potential phi1 (in Fourier) ----
        # phi1_k = - delta_k / k^2
        phi1_k = delta_k.copy()
        for kslab, slab in zip(phi1_k.slabs.x, phi1_k.slabs):
            k2 = sum(kk * kk for kk in kslab)
            # Use safe_divide to handle k=0 and avoid IndexError
            slab[...] = -utils.safe_divide(slab, k2)

        # ---- Step 3: Inverse FFT to get phi1 in real space ----
        phi1 = phi1_k.c2r()

        # ---- Step 4: Compute Hessian of phi1 via FFT ----
        phi1_k2 = phi1.r2c()   # forward FFT of phi1

        # Dictionary to store the six Hessian components in real space
        hessian = {}
        for i in range(3):
            for j in range(i, 3):
                comp_k = phi1_k2.copy()
                for kslab, slab in zip(comp_k.slabs.x, comp_k.slabs):
                    slab[...] *= -kslab[i] * kslab[j]   # frequency factor for ∂i∂j
                hessian[(i, j)] = comp_k.c2r()

        # ---- Step 5: Compute tidal source delta2 ----
        phi_xx, phi_yy, phi_zz = hessian[(0,0)], hessian[(1,1)], hessian[(2,2)]
        phi_xy, phi_xz, phi_yz = hessian[(0,1)], hessian[(0,2)], hessian[(1,2)]

        delta2 = phi1.pm.create(type='real', value=0.)
        for slab, sxx, syy, szz, sxy, sxz, syz in zip(
                delta2.slabs,
                phi_xx.slabs, phi_yy.slabs, phi_zz.slabs,
                phi_xy.slabs, phi_xz.slabs, phi_yz.slabs):
            slab[...] = (sxx * syy - sxy * sxy) + \
                        (sxx * szz - sxz * sxz) + \
                        (syy * szz - syz * syz)

        # Clean up large intermediate arrays
        del phi1_k, phi1, phi1_k2, hessian
        del phi_xx, phi_yy, phi_zz, phi_xy, phi_xz, phi_yz

        # ---- Step 6: Compute first-order displacements (original) ----
        psis = []
        for iaxis in range(delta_k.ndim):
            psi_k = delta_k.copy()
            for kslab, islab, slab in zip(psi_k.slabs.x, psi_k.slabs.i, psi_k.slabs):
                mask = islab[iaxis] != self.nmesh[iaxis] // 2
                k2 = sum(kk * kk for kk in kslab)
                slab[...] = 1j * utils.safe_divide(kslab[iaxis], k2) * slab * mask
            psis.append(psi_k.c2r())
            del psi_k

        # ---- Step 7: Compute second-order displacements and add D2*psi2 ----
        delta2_k = delta2.r2c()
        for iaxis in range(delta_k.ndim):
            psi2_k = delta2_k.copy()
            for kslab, islab, slab in zip(psi2_k.slabs.x, psi2_k.slabs.i, psi2_k.slabs):
                mask = islab[iaxis] != self.nmesh[iaxis] // 2
                k2 = sum(kk * kk for kk in kslab)
                slab[...] = 1j * utils.safe_divide(kslab[iaxis], k2) * slab * mask
            psi2 = psi2_k.c2r()
            psis[iaxis] += self.D2 * psi2
            del psi2_k, psi2

        # Final clean-up
        del delta_k, delta2_k, delta2

        return psis