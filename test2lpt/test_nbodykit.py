import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from nbodykit.lab import cosmology, LogNormalCatalog
from nbodykit import setup_logging


def growth_rate_approx(z, omega_m0=0.315):
    ez2 = omega_m0 * (1.0 + z) ** 3 + (1.0 - omega_m0)
    omega_m_z = omega_m0 * (1.0 + z) ** 3 / ez2
    return omega_m_z ** 0.55


def gaussian_window(kx, ky, kz, Rsm):
    kk2 = kx**2 + ky**2 + kz**2
    return np.exp(-0.5 * kk2 * Rsm**2)


def shell_average_product(fk1, fk2, kmag, bins):
    prod = np.real(fk1 * np.conj(fk2)).ravel()
    km = kmag.ravel()

    kvals, pvals = [], []
    idx = np.digitize(km, bins) - 1
    for i in range(len(bins) - 1):
        m = idx == i
        if np.any(m):
            kvals.append(km[m].mean())
            pvals.append(prod[m].mean())
    return np.asarray(kvals), np.asarray(pvals)


setup_logging()

# ----------------------------
# 1) Generate one mock catalog
# ----------------------------
z = 0.55
cosmo = cosmology.Planck15
Plin = cosmology.LinearPower(cosmo, z, transfer="EisensteinHu")

boxsize = 1380.0   # Mpc/h, large enough for linear-scale test
nmesh = 256
bias = 2.0
nbar = 3e-4        # (h/Mpc)^3

cat = LogNormalCatalog(
    Plin=Plin,
    nbar=nbar,
    BoxSize=boxsize,
    Nmesh=nmesh,
    bias=bias,
    seed=42,
)

pos = np.asarray(cat["Position"])          # Mpc/h
vel = np.asarray(cat["Velocity"])          # km/s
voff = np.asarray(cat["VelocityOffset"])   # Mpc/h

print("Nobj =", cat.csize)
print("columns =", cat.columns)

# store velocity-offset components as catalog columns
cat["ux"] = voff[:, 0]
cat["uy"] = voff[:, 1]
cat["uz"] = voff[:, 2]

# ----------------------------
# 2) Paint density and velocity-offset fields to mesh
# ----------------------------
density_mesh = cat.to_mesh(
    Nmesh=nmesh,
    BoxSize=boxsize,
    compensated=True,
    window="cic",
    position="Position",
)

ux_mesh = cat.to_mesh(
    Nmesh=nmesh,
    BoxSize=boxsize,
    compensated=True,
    window="cic",
    position="Position",
    value="ux",
)
uy_mesh = cat.to_mesh(
    Nmesh=nmesh,
    BoxSize=boxsize,
    compensated=True,
    window="cic",
    position="Position",
    value="uy",
)
uz_mesh = cat.to_mesh(
    Nmesh=nmesh,
    BoxSize=boxsize,
    compensated=True,
    window="cic",
    position="Position",
    value="uz",
)

delta = np.asarray(density_mesh.paint(mode="real")) - 1.0
ux = np.asarray(ux_mesh.paint(mode="real"))
uy = np.asarray(uy_mesh.paint(mode="real"))
uz = np.asarray(uz_mesh.paint(mode="real"))

# ----------------------------
# 3) Smooth both fields, then compute theta = div(u)
#    Here u is already in Mpc/h because VelocityOffset is in Mpc/h.
# ----------------------------
Rsm = 12.0  # Mpc/h, tune this if needed

dx = boxsize / nmesh
k1d = 2.0 * np.pi * np.fft.fftfreq(nmesh, d=dx)
kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing="ij")
kmag = np.sqrt(kx**2 + ky**2 + kz**2)

W = gaussian_window(kx, ky, kz, Rsm)

delta_k = np.fft.fftn(delta) * W
ux_k = np.fft.fftn(ux) * W
uy_k = np.fft.fftn(uy) * W
uz_k = np.fft.fftn(uz) * W

theta_k = 1j * (kx * ux_k + ky * uy_k + kz * uz_k)
theta = np.fft.ifftn(theta_k).real
delta_sm = np.fft.ifftn(delta_k).real

f = growth_rate_approx(z)

# ----------------------------
# 4) Scatter: theta/f vs -delta
# ----------------------------
rng = np.random.default_rng(1234)
flat_x = (-delta_sm).ravel()
flat_y = (theta / f).ravel()

nsamp = min(50000, flat_x.size)
idx = rng.choice(flat_x.size, size=nsamp, replace=False)

plt.figure(figsize=(6, 6))
plt.scatter(flat_x[idx], flat_y[idx], s=2, alpha=0.15)

lim = np.nanmax(np.abs(np.concatenate([flat_x[idx], flat_y[idx]])))
plt.plot([-lim, lim], [-lim, lim], "k--", lw=1)

plt.xlabel(r"$-\delta$")
plt.ylabel(r"$\theta/f$")
plt.title("nbodykit mock: continuity test")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("theta_over_f_vs_minus_delta.png", dpi=300)
plt.show()

# ----------------------------
# 5) Power spectrum ratios
#    In linear theory on large scales:
#      P_theta,delta / (-f P_dd) -> 1
#      P_theta,theta / (f^2 P_dd) -> 1
#      r(k) = P_td / sqrt(P_tt P_dd) -> -1
# ----------------------------
bins = np.linspace(kmag[kmag > 0].min(), 0.25, 25)

k_dd, P_dd = shell_average_product(delta_k, delta_k, kmag, bins)
k_tt, P_tt = shell_average_product(theta_k, theta_k, kmag, bins)
k_td, P_td = shell_average_product(theta_k, delta_k, kmag, bins)

ratio_td = P_td / (-f * P_dd)
ratio_tt = P_tt / (f**2 * P_dd)
r_corr = P_td / np.sqrt(P_tt * P_dd)

plt.figure(figsize=(7, 5))
plt.plot(k_td, ratio_td, "o-", label=r"$P_{\theta\delta}/(-f P_{\delta\delta})$")
plt.plot(k_tt, ratio_tt, "s-", label=r"$P_{\theta\theta}/(f^2 P_{\delta\delta})$")
plt.axhline(1.0, color="k", ls="--", lw=1)
plt.xlabel(r"$k\;[h\,\mathrm{Mpc}^{-1}]$")
plt.ylabel("ratio")
plt.ylim(0.0, 2.0)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("power_spectrum_ratios.png", dpi=300)
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(k_td, r_corr, "o-")
plt.axhline(-1.0, color="k", ls="--", lw=1)
plt.xlabel(r"$k\;[h\,\mathrm{Mpc}^{-1}]$")
plt.ylabel(r"$r(k)=P_{\theta\delta}/\sqrt{P_{\theta\theta}P_{\delta\delta}}$")
plt.ylim(-1.1, 1.1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("cross_correlation_coefficient.png", dpi=300)
plt.show()

print("delta_sm mean/std =", delta_sm.mean(), delta_sm.std())
print("theta mean/std    =", theta.mean(), theta.std())
print("first few ratio_td =", ratio_td[:5])
print("first few ratio_tt =", ratio_tt[:5])
print("first few r(k)     =", r_corr[:5])