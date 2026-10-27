import os
import numpy as np
import unyt
import hdfstream
import swiftsimio as sw
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
# from pyrecon import IterativeFFTReconstruction
from pyrecon.iterative_fft import IterativeFFTReconstruction
from pyrecon.iterative_fft_2lpt import IterativeFFTReconstruction2LPT

# ============================================================
# Utilities
# ============================================================
def cic_deposit_mass(local_pos, mass, boxsize, ngrid):
    dx = boxsize / ngrid
    grid_mass = np.zeros((ngrid, ngrid, ngrid), dtype=np.float64)

    u = local_pos / dx
    i0 = np.floor(u).astype(np.int64)
    frac = u - i0

    wx0 = 1.0 - frac[:, 0]
    wx1 = frac[:, 0]
    wy0 = 1.0 - frac[:, 1]
    wy1 = frac[:, 1]
    wz0 = 1.0 - frac[:, 2]
    wz1 = frac[:, 2]

    for p in range(local_pos.shape[0]):
        ix0 = i0[p, 0] % ngrid
        iy0 = i0[p, 1] % ngrid
        iz0 = i0[p, 2] % ngrid
        m = mass[p]
        for dxi, wx in ((0, wx0[p]), (1, wx1[p])):
            ix = (ix0 + dxi) % ngrid
            for dyi, wy in ((0, wy0[p]), (1, wy1[p])):
                iy = (iy0 + dyi) % ngrid
                for dzi, wz in ((0, wz0[p]), (1, wz1[p])):
                    iz = (iz0 + dzi) % ngrid
                    grid_mass[ix, iy, iz] += m * wx * wy * wz

    rho = grid_mass / (dx**3)
    return rho


def tidal_deformation_from_particles(pos, mass, center, cut_radius=8.0, ngrid=64, smooth_R=1.5):
    """
    pos, center in Mpc
    mass in arbitrary consistent mass unit
    Returns:
        T_center, trace, D_tid, A_tid
    """
    rel = pos - center
    r = np.linalg.norm(rel, axis=1)
    sel = r < cut_radius
    rel = rel[sel]
    mass = mass[sel]

    if rel.shape[0] < 10:
        return None

    # shift into [0, 2*cut_radius)
    boxsize = 2.0 * cut_radius
    local_pos = rel + cut_radius

    rho = cic_deposit_mass(local_pos, mass, boxsize=boxsize, ngrid=ngrid)
    delta = rho / np.mean(rho) - 1.0

    dx = boxsize / ngrid
    kfreq = 2.0 * np.pi * np.fft.fftfreq(ngrid, d=dx)
    kx, ky, kz = np.meshgrid(kfreq, kfreq, kfreq, indexing="ij")
    k2 = kx**2 + ky**2 + kz**2

    delta_k = np.fft.fftn(delta)
    if smooth_R > 0:
        delta_k *= np.exp(-0.5 * k2 * smooth_R**2)

    inv_k2 = np.zeros_like(k2)
    mask = k2 > 0
    inv_k2[mask] = 1.0 / k2[mask]

    Txx = np.fft.ifftn((kx * kx) * inv_k2 * delta_k).real
    Tyy = np.fft.ifftn((ky * ky) * inv_k2 * delta_k).real
    Tzz = np.fft.ifftn((kz * kz) * inv_k2 * delta_k).real
    Txy = np.fft.ifftn((kx * ky) * inv_k2 * delta_k).real
    Txz = np.fft.ifftn((kx * kz) * inv_k2 * delta_k).real
    Tyz = np.fft.ifftn((ky * kz) * inv_k2 * delta_k).real

    c = ngrid // 2
    T = np.array([
        [Txx[c, c, c], Txy[c, c, c], Txz[c, c, c]],
        [Txy[c, c, c], Tyy[c, c, c], Tyz[c, c, c]],
        [Txz[c, c, c], Tyz[c, c, c], Tzz[c, c, c]],
    ], dtype=np.float64)

    trace = np.trace(T)
    Tt = T - np.eye(3) * trace / 3.0
    D_tid = np.sqrt(np.sum(Tt**2))
    A_tid = D_tid / (np.abs(trace) + 1e-8)

    return T, trace, D_tid, A_tid
# ============================================================
def weighted_mean_vector(values, weights=None):
    """
    values: (N, 3) array with units
    weights: (N,) array or None
    Returns weighted mean vector (3,) with same units.
    """
    if values.shape[0] == 0:
        return np.full(3, np.nan) * values.units
    if weights is None:
        return np.mean(values, axis=0)
    wsum = np.sum(weights)
    if wsum == 0:
        return np.full(3, np.nan) * values.units
    return np.sum(weights[:, np.newaxis] * values, axis=0) / wsum

def growth_rate_approx(z, omega_m0=0.315):
    ez2 = omega_m0 * (1.0 + z) ** 3 + (1.0 - omega_m0)
    omega_m_z = omega_m0 * (1.0 + z) ** 3 / ez2
    return omega_m_z ** 0.55

def build_randoms(data_positions, nrand_factor=10, seed=1234):
    rng = np.random.default_rng(seed)
    mins = np.min(data_positions, axis=0)
    maxs = np.max(data_positions, axis=0)
    nrand = max(len(data_positions) * nrand_factor, 1000)
    randoms_positions = rng.uniform(mins, maxs, size=(nrand, 3))
    randoms_weights = np.ones(nrand, dtype="f8")
    return randoms_positions, randoms_weights

# ============================================================
# Main settings
# ============================================================
os.makedirs("./figs", exist_ok=True)

root_dir = hdfstream.open("cosma", "/")
soap_file = root_dir["FLAMINGO/L1_m10/L1_m10/SOAP-HBT/halo_properties_0070.hdf5"]
snap_file = root_dir["FLAMINGO/L1_m10/L1_m10/snapshots/flamingo_0070/flamingo_0070.hdf5"]

# ----- 用户参数 -----
# 质量筛选 (M200c 单位: Msun, SOAP中为1e10 Msun)
mass_min = 5e11/1e10
mass_max = 1e15/1e10

# 统计样本大小（从所有合格晕中抽取多少个用于计算气体速度）
sample_size = 50          # 例如 50 个晕
random_seed = 42          # 固定随机种子保证可重复

# 计算气体平均速度时是否质量加权
weighted_by_mass = True

# 重建参数
f_growth_default = 0.8
bias_default = 2.0
nmesh = 256*2
smoothing_radius = 15.0   # Mpc

# 读取气体时的搜索半径 (Mpc) —— 只需大于晕尺寸
search_radius = 16.0 * unyt.Mpc

# --------------------------

# ============================================================
# 1. 读取所有符合条件的晕
# ============================================================
soap = sw.load(soap_file)
halo_pos = soap.input_halos.halo_centre
halo_m200c = soap.spherical_overdensity_200_crit.total_mass   # 单位: 1e10 Msun
halo_index = soap.input_halos.halo_catalogue_index

mass_mask = (halo_m200c > mass_min ) & (halo_m200c < mass_max )
halo_pos_all = halo_pos[mass_mask]
halo_m200c_all = halo_m200c[mass_mask]
halo_index_all = halo_index[mass_mask]

# 按质量排序（可选，用于后续抽取）
sorted_idx = np.argsort(halo_m200c_all)[::-1]   # 从大到小
halo_pos_sorted = halo_pos_all[sorted_idx]
halo_mass_sorted = halo_m200c_all[sorted_idx]   # 转换为 Msun
halo_index_sorted = halo_index_all[sorted_idx]

print(f"总合格晕数量: {len(halo_mass_sorted)}")

# ============================================================
# 2. 使用所有晕重建速度场
# ============================================================
data_positions = np.asarray(halo_pos_sorted).astype("f8")
data_weights = np.asarray(halo_mass_sorted, dtype="f8")   # 质量作为权重

# 设置 box
mins = data_positions.min(axis=0)
maxs = data_positions.max(axis=0)
boxsize = (maxs - mins).max()
boxcenter = 0.5 * (maxs + mins)

randoms_positions, randoms_weights = build_randoms(data_positions, nrand_factor=10, seed=1234)

recon = IterativeFFTReconstruction(
    f=f_growth_default,
    bias=bias_default,
    los=None,              # 各向同性重建
    nmesh=nmesh,
    boxsize=boxsize,
    boxcenter=boxcenter,
)
recon.assign_data(data_positions, data_weights)
recon.assign_randoms(randoms_positions, randoms_weights)
recon.set_density_contrast(smoothing_radius=smoothing_radius)
recon.run()

# 获取重建后的位移场
data_positions_rec = recon.read_shifted_positions(data_positions)
psi = data_positions_rec - data_positions   # 位移 (Mpc)

# Adding 2LPT reconstruction
recon2lpt = IterativeFFTReconstruction2LPT(
    f=f_growth_default,
    bias=bias_default,
    los=None,              # 各向同性重建
    nmesh=nmesh,
    boxsize=boxsize,
    boxcenter=boxcenter,
)
recon2lpt.assign_data(data_positions, data_weights)
recon2lpt.assign_randoms(randoms_positions, randoms_weights)
recon2lpt.set_density_contrast(smoothing_radius=smoothing_radius)
recon2lpt.run()

# Obtain the 2LPT reconstructed positions and displacements
data_positions_rec2 = recon2lpt.read_shifted_positions(data_positions)
psi2 = data_positions_rec2 - data_positions   # 位移 (Mpc)




# 转换为速度 (三维)
z_recon = 0.4
a = 1.0 / (1.0 + z_recon)
H = cosmo.H(z_recon).to_value(u.km / u.s / u.Mpc)
f = growth_rate_approx(z_recon)
v_recon_vec = (a * H * f) * psi   # 单位: km/s
v_recon_vec = unyt.unyt_array(v_recon_vec, "km/s")

# load the two LPT reconstructed velocity
v_recon_vec_2lpt = (a * H * f) * psi2
v_recon_vec_2lpt = unyt.unyt_array(v_recon_vec_2lpt, "km/s")


# Plot the histogram of the velocity magnitudes for both reconstructions
v_magnitudes_1lpt = np.linalg.norm(v_recon_vec, axis=1)
v_magnitudes_2lpt = np.linalg.norm(v_recon_vec_2lpt, axis=1)
plt.figure(figsize=(8, 5))
plt.hist(v_magnitudes_1lpt, bins=30, alpha=0.5, label='1LPT', color='blue',   edgecolor='none')
plt.hist(v_magnitudes_2lpt, bins=30, alpha=0.5, label='2LPT', color='orange', edgecolor='none')
plt.xlabel('Velocity Magnitude [km/s]')
plt.ylabel('Frequency')
plt.legend()
plt.title('Velocity Magnitude Distribution')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("./figs/velocity_magnitude_histogram.pdf", dpi=300)
plt.show()

# 现在 v_recon_vec 是一个 (N_halo, 3) 的数组，顺序与 data_positions 相同（即 sorted）

# ============================================================
# 3. 从所有晕中随机抽取样本（或按质量分层）
# ============================================================
# 这里使用随机抽样，不重复
rng = np.random.default_rng(random_seed)
total_halos = len(halo_mass_sorted)
if sample_size > total_halos:
    sample_size = total_halos
    print(f"样本大小超过总晕数，使用全部 {total_halos} 个晕")
sample_indices = rng.choice(total_halos, size=sample_size, replace=False)

# 也可按质量分层抽样（如均匀抽取不同质量区间），这里暂用随机

# 存储结果
mass_sample = []
delta_v_norm_sample = []   # 范数差值
delta_2lpt_sample = []     # 2LPT 范数差值
delta_v_vec_sample = []    # 三维向量差值
tidal_sample = []
print(f"\n开始处理 {len(sample_indices)} 个样本晕 ...")

for i in sample_indices:
    # 获取该晕的信息
    pos = halo_pos_sorted[i]
    mass = halo_mass_sorted[i]
    hidx = halo_index_sorted[i]
    v_recon = v_recon_vec[i]   # 三维重建速度
    v_recon_2lpt = v_recon_vec_2lpt[i]   # 三维2LPT重建速度

    # 读取该晕周围气体
    region = [[x - search_radius, x + search_radius] for x in pos]
    mask = sw.mask(snap_file)
    mask.constrain_spatial(region)
    snap = sw.load(snap_file, mask=mask)

    # gas_pos = snap.gas.coordinates
    # gas_vel = snap.gas.velocities
    # gas_halo_index = snap.gas.halo_catalogue_index
    # gas_mass = snap.gas.masses

    # # 属于该晕的气体
    # in_halo = gas_halo_index == hidx
    # if np.sum(in_halo) == 0:
    #     print(f"晕 {hidx} 没有气体粒子，跳过")
    #     continue

    # gas_vel_halo = gas_vel[in_halo, :]          # (N, 3)
    # gas_mass_halo = gas_mass[in_halo] if weighted_by_mass else None

    # # 计算平均三维速度
    # v_gas_mean = weighted_mean_vector(gas_vel_halo, weights=gas_mass_halo)
    # if np.any(np.isnan(v_gas_mean)):
    #     print(f"晕 {hidx} 平均速度无效，跳过")
    #     continue
    
    # using Dark Matter Particles to compute the mean velocity of the halo
    dm_pos = snap.dark_matter.coordinates
    dm_vel = snap.dark_matter.velocities
    dm_halo_index = snap.dark_matter.halo_catalogue_index
    dm_mass = snap.dark_matter.masses

    in_halo_dm = dm_halo_index == hidx
    if np.sum(in_halo_dm) == 0:
        print(f"晕 {hidx} 没有暗物质粒子，跳过")
        continue
    dm_vel_halo = dm_vel[in_halo_dm, :]
    dm_mass_halo = dm_mass[in_halo_dm] if weighted_by_mass else None
    v_gas_mean = weighted_mean_vector(dm_vel_halo, weights=dm_mass_halo)


    # tidal field around the halo
    tidal_out = tidal_deformation_from_particles(
        pos=np.asarray(dm_pos).astype("f8"),
        mass=np.asarray(dm_mass).astype("f8"),
        center=np.asarray(pos).astype("f8"),
        cut_radius=search_radius.to_value("Mpc"),
        ngrid=64,
        smooth_R=1.5,
    )

    if tidal_out is None:
        print(f"晕 {hidx} tidal 计算失败，跳过")
        continue

    T, tidal_trace, tidal_def, tidal_rel = tidal_out
    # 速度差范数
    delta_v = np.linalg.norm(v_gas_mean - v_recon)
    delta_v_2lpt = np.linalg.norm(v_gas_mean - v_recon_2lpt)
    mass_sample.append(mass)
    delta_v_norm_sample.append(delta_v.to_value("km/s"))
    delta_v_vec_sample.append(v_gas_mean - v_recon)
    delta_2lpt_sample.append(delta_v_2lpt.to_value("km/s"))
    tidal_sample.append(tidal_rel)   # 或 tidal_def

print(f"成功处理 {len(mass_sample)} 个晕")

# ============================================================
# 4. 绘制结果
# ============================================================
if len(mass_sample) == 0:
    raise RuntimeError("没有成功处理任何晕，请检查数据。")

mass_sample = np.array(mass_sample)
delta_sample = np.array(delta_v_norm_sample)
# delta_vec_sample = np.array(delta_v_vec_sample)
delta_2lpt_sample = np.array(delta_2lpt_sample)

# 4a. 散点图
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(mass_sample, delta_sample, s=40, alpha=0.7, edgecolor='k')
ax.scatter(mass_sample, delta_2lpt_sample, s=40, alpha=0.7, edgecolor='k', marker='x', color='r')
ax.set_xscale('log')
ax.set_xlabel(r"Halo mass $M_{\rm 200c}$ [ $\times 10^{10} M_\odot$]")
ax.set_ylabel(r"$\Delta v = |\mathbf{v}_{\rm gas} - \mathbf{v}_{\rm recon}|$ [km/s]")
ax.grid(True, alpha=0.3)
ax.set_title("Velocity difference norm vs halo mass")
fig.tight_layout()
fig.savefig("./figs/delta_norm_vs_mass_scatter.pdf", dpi=300)
plt.show()
