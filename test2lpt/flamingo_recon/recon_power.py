"""
Reconstruction power of Zeldovich vs 2LPT reconstruction on a FLAMINGO L1_m10 sub-box (DM particles, real space, b = 1).
Truth: displacement x - q (q from particle IDs on the 900^3 lattice) and peculiar velocity.
Outputs: recon_power_results.npz (per-particle and field-level statistics for several smoothing radii).
"""
import sys
import time
import numpy as np

from pyrecon import IterativeFFTReconstruction2LPT, setup_logging
from pyrecon.second_order import get_D2

setup_logging('warning')
fn = sys.argv[1] if len(sys.argv) > 1 else 'flamingo_subbox.npz'
out = sys.argv[2] if len(sys.argv) > 2 else 'recon_power_results.npz'
smoothings_h = [float(r) for r in sys.argv[3].split(',')] if len(sys.argv) > 3 else [5., 7.5, 10., 15., 20.]  # Mpc/h
margin, cellsize, field_cell = 30., 2., 5.  # Mpc

d = np.load(fn)
x, q, v = d['x'], d['q'], d['v'].astype('f8')
L, lo, hi, z, h, omega_m0, H = float(d['boxsize']), float(d['lo']), float(d['hi']), float(d['redshift']), float(d['h']), float(d['Omega_m']), float(d['H'])
a = 1. / (1. + z)
omega_mz = omega_m0 * (1 + z)**3 * (100. * h / H)**2
f = omega_mz**0.55
D2 = get_D2(omega_mz)
psi_true = (x - q + L / 2.) % L - L / 2.
print(f'{len(x)} particles in [{lo:.1f}, {hi:.1f}] Mpc; z = {z}, H = {H:.2f} km/s/Mpc, Omega_m(z) = {omega_mz:.3f}, f = {f:.3f}, D2 = {D2:.4f}', flush=True)
print(f'mean psi_true = {psi_true.mean(axis=0)} Mpc, rms |psi_true| = {np.sqrt(np.mean(np.sum(psi_true**2, -1))):.2f} Mpc', flush=True)

interior = np.all((x > lo + margin) & (x < hi - margin), axis=1)
rng = np.random.default_rng(42)
randoms = rng.uniform(lo, hi, size=(2 * len(x), 3))

# coarse field-level mesh over the interior cube
nf = int(round((hi - lo - 2 * margin) / field_cell))
edge_lo = lo + margin


def paint(values, positions, weights):
    # NGP mass-weighted average of values in cells of the interior cube
    idx = np.floor((positions - edge_lo) / field_cell).astype('i8')
    ok = np.all((idx >= 0) & (idx < nf), axis=1)
    flat = np.ravel_multi_index(idx[ok].T, (nf,) * 3)
    w = np.bincount(flat, weights=weights[ok], minlength=nf**3)
    toret = [np.bincount(flat, weights=(weights * values[:, i])[ok], minlength=nf**3) for i in range(values.shape[1])]
    with np.errstate(invalid='ignore'):
        toret = np.array([np.where(w > 0, t / w, 0.) for t in toret]).reshape((-1,) + (nf,) * 3)
    return toret, (w > 0).reshape((nf,) * 3)


taper1d = np.ones(nf)
nt = max(nf // 8, 1)
taper1d[:nt] = 0.5 * (1 - np.cos(np.pi * (np.arange(nt) + 0.5) / nt))
taper1d[-nt:] = taper1d[:nt][::-1]
taper = taper1d[:, None, None] * taper1d[None, :, None] * taper1d[None, None, :]
kf = 2 * np.pi * np.fft.fftfreq(nf, d=field_cell)
kmod = np.sqrt(sum(kk**2 for kk in np.meshgrid(kf, kf, kf, indexing='ij')))
kedges = np.arange(1.5, nf // 2, 1.) * 2 * np.pi / (nf * field_cell)
kbin = np.digitize(kmod.ravel(), kedges)


def cross_spectra(A, B, mask):
    # vector-field auto/cross power in k bins (sum over components), after subtracting means and tapering
    Ak = [np.fft.fftn((a - a[mask].mean()) * mask * taper) for a in A]
    Bk = [np.fft.fftn((b - b[mask].mean()) * mask * taper) for b in B]
    paa = sum(np.abs(ak)**2 for ak in Ak).ravel()
    pbb = sum(np.abs(bk)**2 for bk in Bk).ravel()
    pab = sum((ak * bk.conj()).real for ak, bk in zip(Ak, Bk)).ravel()
    nb = len(kedges) - 1
    s = lambda p: np.bincount(kbin, weights=p, minlength=nb + 2)[1:nb + 1]
    return s(paa), s(pbb), s(pab)


def stats(true, rec):
    true = true - true.mean(axis=0); rec = rec - rec.mean(axis=0)
    return dict(r=np.sum(true * rec) / np.sqrt(np.sum(true**2) * np.sum(rec**2)),
                slope=np.sum(true * rec) / np.sum(rec**2),  # true = slope * rec
                resid=np.sqrt(np.mean(np.sum((true - rec)**2, -1)) / np.mean(np.sum(true**2, -1))))


results = dict(smoothings_h=np.array(smoothings_h), kmid=0.5 * (kedges[1:] + kedges[:-1]), f=f, D2=D2, a=a, H=H, h=h, z=z)
mass = np.ones(len(x))
pt_true, mask = paint(psi_true[interior], x[interior], mass[interior])
vt_true, _ = paint(v[interior], x[interior], mass[interior])
for R in smoothings_h:
    t0 = time.time()
    recon = IterativeFFTReconstruction2LPT(f=0., bias=1., los=None, positions=randoms, cellsize=cellsize, D2=D2, dtype='f8')
    recon.assign_data(x)
    recon.assign_randoms(randoms)
    recon.set_density_contrast(smoothing_radius=R / h)
    recon.run()
    psi1 = np.empty_like(x); psi2 = np.empty_like(x)
    for i in range(3):
        psi1[:, i] = recon._readout(recon.mesh_psi[i], x)
        psi2[:, i] = recon._readout(recon.mesh_psi2[i], x)
    del recon
    rec = {'ZA': (psi1, psi1), '2LPT': (psi1 + psi2, psi1 + (1. + D2) * psi2)}
    for name, (pdisp, pvel) in rec.items():
        vrec = f * a * H * pvel
        sd, sv = stats(psi_true[interior], pdisp[interior]), stats(v[interior], vrec[interior])
        pr, _ = paint(pdisp[interior], x[interior], mass[interior])
        vr, _ = paint(vrec[interior], x[interior], mass[interior])
        ptt, prr, ptr = cross_spectra(pt_true, pr, mask)
        vtt, vrr, vtr = cross_spectra(vt_true, vr, mask)
        key = f'{name}_R{R:g}'
        results[key + '_disp'] = np.array([sd['r'], sd['slope'], sd['resid']])
        results[key + '_vel'] = np.array([sv['r'], sv['slope'], sv['resid']])
        results[key + '_rk_disp'] = ptr / np.sqrt(ptt * prr)
        results[key + '_rk_vel'] = vtr / np.sqrt(vtt * vrr)
        results[key + '_tk_disp'] = ptr / prr
        results[key + '_tk_vel'] = vtr / vrr
        results[key + '_field_resid_disp'] = np.sqrt(np.sum(((pt_true - pt_true[:, mask].mean(axis=1)[:, None, None, None]) - (pr - pr[:, mask].mean(axis=1)[:, None, None, None]))[:, mask]**2) / np.sum((pt_true - pt_true[:, mask].mean(axis=1)[:, None, None, None])[:, mask]**2))
        print(f'R = {R:g} Mpc/h {name:4s}: particle disp r = {sd["r"]:.4f}, slope = {sd["slope"]:.3f}, resid = {sd["resid"]:.3f} | '
              f'vel r = {sv["r"]:.4f}, slope = {sv["slope"]:.3f}, resid = {sv["resid"]:.3f} | field disp resid = {results[key + "_field_resid_disp"]:.3f}', flush=True)
    if R == 10.:
        sl = slice(None, None, 7)
        results['sample_x'], results['sample_true'], results['sample_v'] = x[interior][sl], psi_true[interior][sl], v[interior][sl]
        results['sample_za'], results['sample_2lpt'] = psi1[interior][sl], (psi1 + psi2)[interior][sl]
        results['field_true'], results['field_za'] = pt_true[0, :, :, nf // 2], paint(psi1[interior], x[interior], mass[interior])[0][0, :, :, nf // 2]
        results['field_2lpt'] = paint((psi1 + psi2)[interior], x[interior], mass[interior])[0][0, :, :, nf // 2]
        results['field_extent'] = np.array([edge_lo, edge_lo + nf * field_cell])
    print(f'  ({time.time() - t0:.0f} s)', flush=True)
np.savez(out, **results)
