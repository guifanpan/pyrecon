import numpy as np
# from pyrecon.multigrid_2lpt import MultiGridReconstruction2LPT
from pyrecon.multigrid import OriginalMultiGridReconstruction
# from pyrecon import utils
from pyrecon.iterative_fft import IterativeFFTReconstruction

# 生成模拟数据
n = 20000
np.random.seed(123)
data_x = np.random.uniform(100, 900, n)
data_y = np.random.uniform(100, 900, n)
data_z = np.random.uniform(100, 900, n)
data_w = np.ones(n)                     # 权重可设为全 1，也可不传（默认全 1）

rand_x = np.random.uniform(50, 950, 10*n)
rand_y = np.random.uniform(50, 950, 10*n)
rand_z = np.random.uniform(50, 950, 10*n)
rand_w = np.ones(10*n)

# 将坐标堆叠成 (N, 3) 数组
data_pos = np.column_stack([data_x, data_y, data_z])
rand_pos = np.column_stack([rand_x, rand_y, rand_z])

# 初始化重建对象
# recon1 = OriginalMultiGridReconstruction(
#     f=0.8, bias=2.0, nmesh=128,
#     boxsize=[1000., 1000., 1000.],
#     boxcenter=[500., 500., 500.]
# )
recon1 = IterativeFFTReconstruction(
    f=0.8, bias=2.0, nmesh=128,
    boxsize=[1000., 1000., 1000.],
    boxcenter=[500., 500., 500.]
)
# 正确传入位置和权重
recon1.assign_data(data_pos, weights=data_w)     # 或只传 data_pos，默认权重为 1
recon1.assign_randoms(rand_pos, weights=rand_w)

# 计算密度对比（需要先调用）
recon1.set_density_contrast(smoothing_radius=15.0)

# 执行重建
recon1.run()

# 读取位移场
pos = np.column_stack([data_x, data_y, data_z])   # 与 data_pos 相同，也可直接用 data_pos
shifts1 = recon1.read_shifts(pos, field='disp')
print("1LPT shifts RMS:", np.sqrt(np.mean(shifts1**2)))

# # 初始化
# recon = MultiGridReconstruction2LPT(
#     f=0.8, bias=2.0, nmesh=128,
#     boxsize=[1000.0, 1000.0, 1000.0],
#     boxcenter=[500.0, 500.0, 500.0],
#     D2=-0.35, omega_m=0.31,
#     los=None  # 局部视线方向，或 'z'
# )

# # 分配
# recon.assign_data([data_x, data_y, data_z], data_w, position_type='xyz')
# recon.assign_randoms([rand_x, rand_y, rand_z], rand_w, position_type='xyz')

# # 密度对比度
# recon.set_density_contrast(smoothing_radius=15.0, threshold_randoms=0.5)  # 提高阈值

# # 运行 2LPT
# recon.run()

# # 读取位移
# pos = np.column_stack([data_x, data_y, data_z])
# shifts = recon.read_shifts(pos, field='disp')
# print("Shifts: mean =", np.mean(shifts, axis=0), "std =", np.std(shifts, axis=0))
# print("Any NaN?", np.any(np.isnan(shifts)))