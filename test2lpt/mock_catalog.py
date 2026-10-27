import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.cosmology import Planck15
from nbodykit.lab import cosmology, LogNormalCatalog
from nbodykit import setup_logging

from pyrecon.iterative_fft import IterativeFFTReconstruction

# 如果你的环境里这个类名可用，再打开
# from pyrecon.iterative_fft_2lpt import IterativeFFTReconstruction2LPT


def growth_rate_approx(z, omega_m0=0.315):
    ez2 = omega_m0 * (1.0 + z) ** 3 + (1.0 - omega_m0)
    omega_m_z = omega_m0 * (1.0 + z) ** 3 / ez2
    return omega_m_z ** 0.55


def build_randoms(boxsize, nrand, seed=1234):
    rng = np.random.default_rng(seed)
    randoms_positions = rng.uniform(0.0, boxsize, size=(nrand, 3))
    randoms_weights = np.ones(nrand, dtype="f8")
    return randoms_positions, randoms_weights


setup_logging()

# mock redshift and cosmology
z = 0.55
cosmo = cosmology.Planck15
Plin = cosmology.LinearPower(cosmo, z, transfer="EisensteinHu")

# generate a mock catalog
boxsize = 256.0
cat = LogNormalCatalog(
    Plin=Plin,
    nbar=5e-3,      # (h/Mpc)^3
    BoxSize=boxsize,
    Nmesh=128,
    bias=2.0,
    seed=42,
)

pos = np.asarray(cat["Position"])
vel = np.asarray(cat["Velocity"])
voff = np.asarray(cat["VelocityOffset"])
print("mock catalog generated.")
print("number of objects =", cat.csize)
print("columns =", cat.columns)
print("Velocity mean =", np.mean(vel, axis=0))
print("Velocity std =", np.std(vel, axis=0))
print("VelocityOffset mean =", np.mean(voff, axis=0))
print("VelocityOffset std =", np.std(voff, axis=0))
# fixed box center for a periodic mock
boxcenter = np.array([0.5 * boxsize, 0.5 * boxsize, 0.5 * boxsize])

# randoms
randoms_positions, randoms_weights = build_randoms(
    boxsize=boxsize,
    nrand=max(len(pos) * 10, 1000),
    seed=1234
)

# reconstruction params
f_growth_default = growth_rate_approx(z)
bias_default = 2.0
nmesh = 256
smoothing_radius = 0.5  # Mpc/h

recon = IterativeFFTReconstruction(
    f=f_growth_default,
    bias=bias_default,
    los=None,
    nmesh=nmesh,
    boxsize=boxsize,
    boxcenter=boxcenter,
)
LOS = np.array([0., 0., 1.])

# # 把 real-space mock 变成 “观测到的” redshift-space catalog
# rsd_pos = pos + voff * LOS   # voff 是 VelocityOffset，单位 Mpc/h

# recon = IterativeFFTReconstruction(
#     f=f_growth_default,
#     bias=bias_default,
#     los=LOS,
#     nmesh=nmesh,
#     boxsize=boxsize,
#     boxcenter=boxcenter,
# )

# recon.assign_data(rsd_pos, np.ones(len(rsd_pos)))
# recon.assign_randoms(randoms_positions, randoms_weights)
# recon.set_density_contrast(smoothing_radius=smoothing_radius)
# recon.run()

# pos_rec = recon.read_shifted_positions(rsd_pos)
# psi = pos_rec - rsd_pos

recon.assign_data(pos, np.ones(len(pos), dtype="f8"))
recon.assign_randoms(randoms_positions, randoms_weights)
recon.set_density_contrast(smoothing_radius=smoothing_radius)
recon.run()

# reconstructed shifted positions
pos_rec = recon.read_shifted_positions(pos)
psi = pos_rec - pos  # same unit as input positions, i.e. Mpc/h
print("Statistics of displacement field (psi):")
print("  Mean:", np.mean(psi, axis=0))
print("  Std:", np.std(psi, axis=0))

# if you want velocity from displacement, make sure unit conversion is consistent
a = 1.0 / (1.0 + z)
# H = cosmo.H(z).to_value(u.km / u.s / u.Mpc)
# h = cosmo.h if hasattr(cosmo, "h") else 0.677  # fallback if needed
h = 0.677  # fallback if needed
H = Planck15.H(z).to_value(u.km / u.s / u.Mpc)  # Hubble parameter at redshift z in km/s/Mpc
# NOTE: if psi is in Mpc/h, convert to Mpc first by dividing by h

v_recon_vec = (a * H * f_growth_default / h) * psi   # km/s

# comparison along z (example LOS)
v_true_los = vel[:, 2]
v_rec_los = v_recon_vec[:, 2]

# diagnostics
print("true v_los mean/std:", np.mean(v_true_los), np.std(v_true_los))
print("rec  v_los mean/std:", np.mean(v_rec_los), np.std(v_rec_los))

# scatter plot
plt.figure(figsize=(6, 6))
plt.scatter(v_true_los, v_rec_los, s=2, alpha=0.2)
lim = np.max(np.abs([v_true_los.min(), v_true_los.max(), v_rec_los.min(), v_rec_los.max()]))
plt.plot([-lim, lim], [-lim, lim], "k--", lw=1)
plt.xlabel("True $v_{\\rm los}$ [km/s]")
plt.ylabel("Reconstructed $v_{\\rm los}$ [km/s]")
plt.grid(True, alpha=0.3)
plt.tight_layout()
# plt.savefig("./figs/vlos_scatter.png", dpi=300)
plt.show()

# displacement magnitude histogram
psi_mag = np.linalg.norm(psi, axis=1)
plt.figure(figsize=(8, 5))
plt.hist(psi_mag, bins=30, alpha=0.8, edgecolor="none")
plt.xlabel("|displacement| [Mpc/h]")
plt.ylabel("Frequency")
plt.grid(True, alpha=0.3)
plt.tight_layout()
# plt.savefig("./figs/displacement_histogram.pdf", dpi=300)
plt.show()


#verify the power spectrum of the mock catalog and the reconstructed displacement field 
import numpy as np
import matplotlib.pyplot as plt
from nbodykit.algorithms import FFTPower
from nbodykit.lab import cosmology

# ========================
# 1. 密度功率谱 P_density(k)
# ========================
# 直接将模拟星表绘制到网格上（权重为1，因为每个粒子代表一个星系）
density_mesh = cat.to_mesh(Nmesh=128, window='tsc', compensated=True)

# shotnoise='auto' 会自动用 1/nbar 扣除散粒噪声
P_density = FFTPower(density_mesh, mode='1d', dk=0.02)

# 提取结果
k_density = P_density.power['k']
P_density_vals = P_density.power['power'].real  # 已扣除 shotnoise

# ========================
# 2. 速度散度功率谱 P_theta(k)
# ========================
from nbodykit.transform import MeshSource

# 假设你的星系目录为 'cat'，包含 'Position' 和 'Velocity' 列
# 1. 将目录转换为网格，获取速度场分量
vel_mesh = cat.to_mesh(weight='Velocity', Nmesh=128, window='tsc', compensated=True)
vel_field = vel_mesh.compute()  # 获取 (Nmesh, Nmesh, Nmesh, 3) 的数组

# 2. 在傅里叶空间计算散度：θ(k) = i k · v(k)
# 首先，对每个速度分量进行FFT
vel_k = np.fft.fftn(vel_field, axes=(0,1,2))

# 创建波数网格
Nmesh = vel_field.shape[0]
k_vec = np.fft.fftfreq(Nmesh, d=boxsize/Nmesh)  # 注意这里的 d 是网格间距
kx, ky, kz = np.meshgrid(k_vec, k_vec, k_vec, indexing='ij')

# 计算散度的傅里叶变换：θ(k) = i (kx*vx + ky*vy + kz*vz)
# 注意：这里忽略了 (a f H) 的归一化因子，后续可以补上
theta_k = 1j * (kx * vel_k[...,0] + ky * vel_k[...,1] + kz * vel_k[...,2])

# 3. 将散度场变换回实空间，并创建网格对象
theta_x = np.fft.ifftn(theta_k, axes=(0,1,2)).real

# 创建一个新的网格源，用于功率谱计算
theta_mesh = MeshSource(BoxSize=boxsize, Nmesh=Nmesh, dtype='f8')
theta_mesh[:] = theta_x  # 将数据存入网格

# 4. 计算功率谱
from nbodykit.algorithms import FFTPower
P_theta = FFTPower(theta_mesh, mode='1d', dk=0.02, shotnoise=0.0)

# 现在你可以像之前一样使用 P_theta 了
k_theta = P_theta.power['k']
P_theta_vals = P_theta.power['power'].real
# ========================
# 3. （可选）互功率谱 P_cross(k) = <δ_g * θ>
# ========================
# 用多网格对象计算互相关（需要将两个场同时传入）
# 我们可以使用 FFTPower 的 multipoles 模式，或者直接构建两个网格对象用 compute_power
from nbodykit.algorithms import FFTPower  # 已导入

# 方法：将密度网格和速度散度网格放入一个列表，使用 MultipleFFTPower
# 但更简单：利用 FFTPower 的 '1d' 模式，分别计算自谱后，再手动算互谱？
# nbodykit 标准做法是使用 MultipleFFTPower

from nbodykit.algorithms import MultipleFFTPower

# 注意：两个网格必须有相同的 BoxSize 和 Nmesh
mesh_list = [density_mesh, theta_mesh]
P_cross = MultipleFFTPower(mesh_list, mode='1d', dk=0.02, shotnoise=[1/cat.attrs['nbar'], 0.0])

# 提取互谱 (0,1) 分量
k_cross = P_cross.pairs[(0,1)]['k']
P_cross_vals = P_cross.pairs[(0,1)]['power'].real  # 注意互谱的实部

# ========================
# 4. 理论线性曲线
# ========================
z = 0.55
cosmo = cosmology.Planck15
Plin = cosmology.LinearPower(cosmo, z, transfer='EisensteinHu')

# 生成 k 网格（与模拟结果对齐）
k_theory = np.logspace(np.log10(k_density[1]), np.log10(k_density[-1]), 200)
Plin_theory = np.array([Plin(k) for k in k_theory])

bias_mock = 2.0
f_growth = growth_rate_approx(z)  # 你之前定义的函数

# ========================
# 5. 绘图验证
# ========================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# ---- 子图1: 密度功率谱 ----
ax = axes[0, 0]
ax.plot(k_density, k_density**3 * P_density_vals / (2 * np.pi**2), 
        'o-', label='Mock $P_{gg}$', ms=4)
ax.plot(k_theory, k_theory**3 * (bias_mock**2 * Plin_theory) / (2 * np.pi**2), 
        'k--', label=r'$b^2 P_{lin}$', lw=2)
ax.set_xlabel(r'$k$ [$h$/Mpc]')
ax.set_ylabel(r'$\Delta^2(k)$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_title('Density Power Spectrum')

# ---- 子图2: 速度散度功率谱 ----
ax = axes[0, 1]
ax.plot(k_theta, k_theta**3 * P_theta_vals / (2 * np.pi**2), 
        's-', label='Mock $P_{\\theta\\theta}$', ms=4)
ax.plot(k_theory, k_theory**3 * Plin_theory / (2 * np.pi**2), 
        'r--', label=r'$P_{lin}$', lw=2)
ax.set_xlabel(r'$k$ [$h$/Mpc]')
ax.set_ylabel(r'$\Delta^2(k)$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_title('Velocity Divergence Power Spectrum')

# ---- 子图3: 互功率谱 ----
ax = axes[1, 0]
ax.plot(k_cross, k_cross**3 * P_cross_vals / (2 * np.pi**2), 
        'd-', label='Mock $P_{g\\theta}$', ms=4)
ax.plot(k_theory, k_theory**3 * (bias_mock * Plin_theory) / (2 * np.pi**2), 
        'g--', label=r'$b P_{lin}$', lw=2)
ax.set_xlabel(r'$k$ [$h$/Mpc]')
ax.set_ylabel(r'$\Delta^2(k)$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_title('Density-Velocity Cross Spectrum')

# ---- 子图4: 相关性系数 r(k) = P_cross / sqrt(P_gg * P_theta) ----
ax = axes[1, 1]
# 注意：需要将 P 插值到相同的 k 值上，这里简化：直接使用 k_cross 对应的值
from scipy.interpolate import interp1d
P_density_interp = interp1d(k_density, P_density_vals, bounds_error=False, fill_value=0)
P_theta_interp = interp1d(k_theta, P_theta_vals, bounds_error=False, fill_value=0)
P_d_at_cross = P_density_interp(k_cross)
P_t_at_cross = P_theta_interp(k_cross)
r_corr = P_cross_vals / np.sqrt(P_d_at_cross * P_t_at_cross)

ax.axhline(1.0, color='k', linestyle='--', label='Linear theory')
ax.plot(k_cross, r_corr, 'o-', ms=4, label='Mock $r_{g\\theta}$')
ax.set_xlabel(r'$k$ [$h$/Mpc]')
ax.set_ylabel(r'$r_{g\theta}(k)$')
ax.set_xscale('log')
ax.set_ylim(0.5, 1.5)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_title('Cross-correlation Coefficient')

plt.tight_layout()
# plt.savefig("power_spectrum_validation.pdf", dpi=200)
plt.show()

# 打印大尺度（低 k）的比值，快速验证
mask_low_k = (k_density > 0.02) & (k_density < 0.1)
ratio = P_density_vals[mask_low_k] / (bias_mock**2 * np.array([Plin(k) for k in k_density[mask_low_k]]))
print(f"P_density / (b^2 P_lin) 在大尺度上的平均值 = {np.mean(ratio):.3f} ± {np.std(ratio):.3f}")