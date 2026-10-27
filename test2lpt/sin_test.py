import numpy as np
import matplotlib.pyplot as plt

# from pyrecon import IterativeFFTReconstruction
from pyrecon.iterative_fft import IterativeFFTReconstruction
from pyrecon.iterative_fft_2lpt import IterativeFFTReconstruction2LPT

def make_lattice_test(nside=32, boxsize=500.0, amplitude=2.0, mode=3):
    """
    规则三维晶格 + 单模纵向位移场
    psi_x = A sin(k x), psi_y = psi_z = 0
    """

    grid = (np.arange(nside) + 0.5) * boxsize / nside
    qx, qy, qz = np.meshgrid(grid, grid, grid, indexing="ij")
    q = np.column_stack([qx.ravel(), qy.ravel(), qz.ravel()])

    k = 2.0 * np.pi * mode / boxsize

    psi_true = np.zeros_like(q)
    psi_true[:, 0] = amplitude * np.sin(k * q[:, 0])

    x = (q + psi_true) % boxsize
    w = np.ones(len(q), dtype=np.float64)
    return q, x, psi_true, w, k

def validate_psi_recovery():
    boxsize = 500.0
    nmesh = 256
    nside = 32
    amplitude = 2.0
    mode = 50

    # 1) 生成测试样本
    q, x, psi_true, w, k = make_lattice_test(
        nside=nside, boxsize=boxsize, amplitude=amplitude, mode=mode
    )

    # 2) 运行 IterativeFFTReconstruction
    #    f=0.0：先做“纯实空间位移反演”测试，避免把 RSD 项混进来
    recon = IterativeFFTReconstruction(
        f=0.0,
        bias=1.0,
        los=None,
        nmesh=nmesh,
        boxsize=boxsize,
        boxcenter=boxsize / 2.0,
        wrap=True,
    )

    recon.assign_data(x, w)
    # 均匀采样测试可以不加 randoms；如果你想更接近观测流程，再加 randoms
    recon.set_density_contrast(smoothing_radius=5.0)
    recon.run(niterations=3)

    # 3) 读取重建位移场
    #    field='disp' 表示只取 Zeldovich displacement，不加 RSD 项
    psi_rec = recon.read_shifts(x, field="disp")

    # 4) 评估恢复
    true_x = psi_true[:, 0]
    rec_x = psi_rec[:, 0]

    alpha = np.sum(true_x * rec_x) / np.sum(true_x**2)
    corr = np.corrcoef(true_x, rec_x)[0, 1]
    rms_true = np.sqrt(np.mean(true_x**2))
    rms_err = np.sqrt(np.mean((rec_x - true_x)**2))

    print("===== Psi recovery test =====")
    print(f"Npart          : {len(x)}")
    print(f"boxsize        : {boxsize}")
    print(f"mode           : {mode}")
    print(f"amplitude      : {amplitude}")
    print(f"RMS true psi_x : {rms_true:.6f}")
    print(f"RMS error      : {rms_err:.6f}")
    print(f"Relative error : {rms_err / rms_true:.6f}")
    print(f"recovery slope α = {alpha:.6f}")
    print(f"corr(psi_true, psi_rec) = {corr:.6f}")

    recon2LPT = IterativeFFTReconstruction2LPT(
        f=0.0,
        bias=1.0,
        los=None,
        nmesh=nmesh,
        boxsize=boxsize,
        boxcenter=boxsize / 2.0,
        wrap=True,
    )

    recon2LPT.assign_data(x, w)
    recon2LPT.set_density_contrast(smoothing_radius=5.0)
    recon2LPT.run(niterations=3)
    psi_rec_2lpt = recon2LPT.read_shifts(x, field="disp")

    # report 2LPT recovery
    rec_x_2lpt = psi_rec_2lpt[:, 0]
    alpha_2lpt = np.sum(true_x * rec_x_2lpt) / np.sum(true_x**2)
    corr_2lpt = np.corrcoef(true_x, rec_x_2lpt)[0, 1]
    rms_err_2lpt = np.sqrt(np.mean((rec_x_2lpt - true_x)**2))

    print("===== Psi recovery test (2LPT) =====")
    print(f"RMS error      : {rms_err_2lpt:.6f}")
    print(f"Relative error : {rms_err_2lpt / rms_true:.6f}")
    print(f"recovery slope α = {alpha_2lpt:.6f}")
    print(f"corr(psi_true, psi_rec_2lpt) = {corr_2lpt:.6f}")    

    

    plt.figure(figsize=(8, 6))
    plt.scatter(true_x, rec_x, s=3, alpha=0.5,color="blue", label="1LPT")
    plt.scatter(true_x, rec_x_2lpt, s=5, alpha=0.5, color="orange", label="2LPT")
    plt.plot([-amplitude, amplitude], [-amplitude, amplitude], "r--",color="gray",alpha=0.5 )
    plt.xlabel("True psi_x")
    plt.ylabel("Recovered psi_x")
    plt.title("Psi Recovery Test")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    validate_psi_recovery()