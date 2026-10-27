import numpy as np
from astropy.cosmology import Planck18 as cosmology
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
from pyrecon.iterative_fft import IterativeFFTReconstruction
# import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
# -----------------------------
# 1) 读 Abacus halo catalog
# -----------------------------
path = "/home/griffin/program/cosmic_velocity_reconstruction/AbacusSummit/z0.450/"

# 如果是普通 box catalog，用 x_com / v_com
# 如果是 lightcone catalog，用 halo_lc=True, pos_interp / vel_interp / redshift_interp
halo_lc = False  # True for lightcone, False for box catalog

if halo_lc:
    cat = CompaSOHaloCatalog(
        path,
        halo_lc=True,
        cleaned=True,
        fields=["pos_interp", "vel_interp", "redshift_interp", "origin"]
    )
    halos = cat.halos
    pos = np.asarray(halos["pos_interp"], dtype=np.float64)   # (N, 3)
    vel = np.asarray(halos["vel_interp"], dtype=np.float64)   # (N, 3)
    z = np.asarray(halos["redshift_interp"], dtype=np.float64)
else:
    cat = CompaSOHaloCatalog(
        path,
        cleaned=True,
        fields=["x_com", "v_com"],
    )
    halos = cat.halos
    pos = np.asarray(halos["x_com"], dtype=np.float64)
    vel = np.asarray(halos["v_com"], dtype=np.float64)
    # z = np.asarray(halos["redshift_interp"], dtype=np.float64)

# plot slice of halos in x-y plane in a constrained z range
pos2_cen = np.mean(pos[:, 2])
mask = (pos[:, 2] > pos2_cen - 20) & (pos[:, 2] < pos2_cen + 20)

# downgrade the number of points for plotting, keeping only 20 thousand points

rng_mask = np.random.default_rng(1234)
mask = mask & (rng_mask.random(len(pos)) < 20000 / np.sum(mask))

plt.figure(figsize=(8, 8))
scatter = plt.scatter(pos[mask][:, 0], pos[mask][:, 1], s=1, alpha=0.5, c=vel[mask][:, 2])

plt.xlabel("x [Mpc/h]")
plt.ylabel("y [Mpc/h]")
plt.title("Abacus Halo positions in x-y plane")
# plt.xlim(0, 2000)
# plt.ylim(0, 2000)
plt.colorbar(scatter, label="Velocity z [km/s]")
plt.savefig("./figs/halo_positions_xy.pdf", dpi=300)


# -----------------------------
# 2) 构造 randoms
# -----------------------------
# pyrecon 的典型流程需要 randoms（除非你明确假设均匀选择函数）
# 这里给一个 box-like 的最简 randoms 示例：
# 若是 light cone，建议你先在薄壳/局部 patch 里定义一个近似 box 再做

Nrand = 10 * len(pos)

# 这里假设你的坐标已经在一个盒子里，边界大致在 [0, Lbox)
# 如果 Abacus 原始坐标是以中心对称分布，请自己先平移到正盒内
Lbox = 2000.0
# boxcenter will be set to the (weighted) center of the input positions below
boxcenter = None

data_weights = np.ones(len(pos), dtype=np.float64)

# compute weighted center from the positions (use data_weights)
boxcenter = np.average(pos, axis=0, weights=data_weights)

# Generate randoms centered around boxcenter
rng = np.random.default_rng(1234)
randoms = rng.uniform(-Lbox/2.0, Lbox/2.0, size=(Nrand, 3)) + boxcenter

randoms_weights = np.ones(len(randoms), dtype=np.float64)

# -----------------------------
# 3) 跑 pyrecon
# -----------------------------
z = 0.45  # 你可以换成对应的 redshift
f=0.8  #
H = cosmology.H(z).value  # km/s/Mpc
a = 1.0 / (1.0 + z)  # scale factor

# LOS 可以是 None（局域 LOS）、'x'/'y'/'z'，或一个 3-vector
# 对 box 测试，通常先固定 z 作为 LOS
recon = IterativeFFTReconstruction(
    f=0.8,                 # 你可以换成对应 redshift 的增长率
    bias=2.0,              # halo bias 需要你按样本校准
    # los='z',
    nmesh=256,
    boxsize=Lbox,
    boxcenter=boxcenter
)

recon.assign_data(pos, data_weights)
recon.assign_randoms(randoms, randoms_weights)
recon.set_density_contrast(smoothing_radius=10.0)
recon.run()

# -----------------------------
# 4) 取重建后的“位移修正后位置”
# -----------------------------
# 对 pyrecon，常用的是读出 shift 后的位置，再据此构造你的速度估计
shifts = recon.read_shifts(pos) # unit: Mpc/h

# shifts from pyrecon are in units of Mpc/h. Convert to Mpc by dividing by h.
h = cosmology.H0.value / 100.0
vel_rec = shifts * (f * H * a) / h  # km/s
print(f"Reconstructed velocity shape: {vel_rec.shape}, true velocity shape: {vel.shape}")
print(f"Mean reconstructed velocity: {np.mean(vel_rec, axis=0)}, mean true velocity: {np.mean(vel, axis=0)}")
# vel_rec has shape (N, 3), compute correlation along LOS (z-direction)
r = np.mean(vel[:, 2] * vel_rec[:, 2]) / (np.std(vel_rec[:, 2]) * np.std(vel[:, 2]))
print(f"Correlation coefficient between true LOS velocity and reconstructed LOS velocity: {r:.4f}")

# Plot the reconstructed velocity vs true velocity along LOS
plt.figure(figsize=(8, 8))
# plt.scatter(vel[:, 2], vel_rec[:, 2], s=1, alpha=0.5)
scatter_rec = plt.scatter(pos[mask][:, 0], pos[mask][:, 1], s=1, alpha=0.5, c=vel_rec[mask][:, 2])

plt.xlabel("True LOS Velocity [km/s]")
plt.ylabel("Reconstructed LOS Velocity [km/s]")
plt.title("Reconstructed vs True LOS Velocity")
plt.colorbar(scatter_rec, label="Reconstructed Velocity z [km/s]")
# Add a 1:1 line for reference
# plt.plot([-1000, 1000], [-1000, 1000], color='red', linestyle='--')
# plt.xlim(-1000, 1000)
# plt.ylim(-1000, 1000)
plt.savefig("./figs/velocity_reconstruction.pdf", dpi=300)  
plt.show()

plt.figure(figsize=(8, 8))
plt.scatter(vel[mask][:, 2], vel_rec[mask][:, 2], s=1, alpha=0.5)
plt.xlabel("True LOS Velocity [km/s]")
plt.ylabel("Reconstructed LOS Velocity [km/s]")
plt.title("Reconstructed vs True LOS Velocity")
plt.savefig("./figs/velocity_reconstruction_scatter.pdf", dpi=300)
plt.show()