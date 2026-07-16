from .multigrid import OriginalMultiGridReconstruction

# class MultiGridReconstruction2LPT(OriginalMultiGridReconstruction):
#     """
#     MultiGrid reconstruction including second-order LPT (tidal) corrections.
#     Inherits from OriginalMultiGridReconstruction and overrides run().

#     Parameters
#     ----------
#     D2 : float, optional
#         Second-order growth factor (typically negative, e.g. -0.3).
#         If not provided, computed from omega_m using EdS-like approximation.
#     omega_m : float, optional, default=0.31
#         Matter density parameter, used to estimate D2 if not provided.
#     Other parameters are passed to OriginalMultiGridReconstruction.
#     """
#     def __init__(self, *args, D2=None, omega_m=0.31, **kwargs):
#         super().__init__(*args, **kwargs)
#         if D2 is None:
#             # Approximate formula for D2 in LCDM (Buchert & Ehlers, 1993)
#             self.D2 = -3.0 / 7.0 * omega_m ** (-1.0 / 143.0)
#         else:
#             self.D2 = float(D2)
#         self._omega_m = omega_m

#     def run(self, jacobi_damping_factor=0.4, jacobi_niterations=5, vcycle_niterations=6):
#         """
#         Run 2LPT reconstruction:
#             1. Solve first-order potential phi1 from self.mesh_delta.
#             2. Compute tidal source term delta2 from phi1 (Kitaura formula).
#             3. Solve second-order potential phi2 from delta2.
#             4. Combine: phi_total = phi1 + D2 * phi2.
#         The combined potential is stored in self.mesh_phi.
#         """
#         self.jacobi_damping_factor = float(jacobi_damping_factor)
#         self.jacobi_niterations = int(jacobi_niterations)
#         self.vcycle_niterations = int(vcycle_niterations)

#         if self.mpicomm.rank == 0:
#             self.log_info('2LPT: Starting first-order (1LPT) solve.')
        
#         # Step 1: Solve for phi1
#         # Note: _fmg consumes no source, it reads self.mesh_delta and returns new field
#         phi1 = self._fmg(self.mesh_delta)

#         if self.mpicomm.rank == 0:
#             self.log_info('2LPT: Computing tidal source delta2 from phi1.')

#         # Step 2: Compute delta2 = sum_{i>j} (phi_ii * phi_jj - phi_ij^2)
#         # Use FFT to compute Hessian components efficiently
#         # 2a: Forward FFT of phi1
#         phi1_k = phi1.r2c()

#         # Helper to compute a Hessian component in real space
#         # component = IFFT( -ki * kj * phi1_k )
#         def hessian_component(ki, kj):
#             # ki, kj are slab arrays (kx, ky, or kz) for the local slab
#             # We'll iterate over slabs and multiply
#             comp_k = phi1_k.copy()  # ComplexField
#             for slab, kslab_i, kslab_j in zip(comp_k.slabs, ki.slabs, kj.slabs):
#                 # kslab_i and kslab_j are the wave-number arrays for this slab
#                 # They should have the same shape as slab
#                 slab[...] *= -kslab_i * kslab_j
#             return comp_k.c2r()  # inverse FFT to real space

#         # Get wave-number slabs from the complex field (they are stored in the x attribute)
#         # phi1_k.slabs.x returns a tuple of (kx_slabs, ky_slabs, kz_slabs) for each slab
#         # We'll extract them once
#         kx_slabs = phi1_k.slabs.x[0]  # list of slabs for kx
#         ky_slabs = phi1_k.slabs.x[1]
#         kz_slabs = phi1_k.slabs.x[2]

#         # Compute diagonal components
#         phi_xx = hessian_component(kx_slabs, kx_slabs)
#         phi_yy = hessian_component(ky_slabs, ky_slabs)
#         phi_zz = hessian_component(kz_slabs, kz_slabs)
#         # Compute off-diagonal components
#         phi_xy = hessian_component(kx_slabs, ky_slabs)
#         phi_xz = hessian_component(kx_slabs, kz_slabs)
#         phi_yz = hessian_component(ky_slabs, kz_slabs)

#         # Combine to form delta2 (tidal source)
#         # delta2 = (phi_xx*phi_yy - phi_xy^2) + (phi_xx*phi_zz - phi_xz^2) + (phi_yy*phi_zz - phi_yz^2)
#         delta2_field = phi1.pm.create(type='real', value=0.)  # allocate
#         for slab, slab_xx, slab_yy, slab_zz, slab_xy, slab_xz, slab_yz in zip(
#                 delta2_field.slabs, phi_xx.slabs, phi_yy.slabs, phi_zz.slabs,
#                 phi_xy.slabs, phi_xz.slabs, phi_yz.slabs):
#             slab[...] = (slab_xx * slab_yy - slab_xy * slab_xy) + \
#                         (slab_xx * slab_zz - slab_xz * slab_xz) + \
#                         (slab_yy * slab_zz - slab_yz * slab_yz)

#         # Clean up intermediate fields to save memory
#         del phi_xx, phi_yy, phi_zz, phi_xy, phi_xz, phi_yz, phi1_k

#         if self.mpicomm.rank == 0:
#             self.log_info('2LPT: Solving second-order potential phi2.')

#         # Step 3: Solve for phi2 using the same multigrid solver, with source = delta2_field
#         phi2 = self._fmg(delta2_field)

#         if self.mpicomm.rank == 0:
#             self.log_info('2LPT: Combining potentials with D2 = {:.4f}'.format(self.D2))

#         # Step 4: Combine: phi_total = phi1 + D2 * phi2
#         # Add D2 * phi2 to phi1 in-place
#         for slab1, slab2 in zip(phi1.slabs, phi2.slabs):
#             slab1[...] += self.D2 * slab2

#         # Store combined potential as final mesh_phi
#         self.mesh_phi = phi1

#         # Clean up
#         del delta2_field, phi2

#         if self.mpicomm.rank == 0:
#             self.log_info('2LPT reconstruction completed.')


# multigrid_2lpt.py
import numpy as np
from .multigrid import OriginalMultiGridReconstruction

class MultiGridReconstruction2LPT(OriginalMultiGridReconstruction):
    def __init__(self, *args, D2=None, omega_m=0.31, **kwargs):
        super().__init__(*args, **kwargs)
        if D2 is None:
            self.D2 = -3.0 / 7.0 * omega_m ** (-1.0 / 143.0)
        else:
            self.D2 = float(D2)
        self._omega_m = omega_m
            
    def run(self, jacobi_damping_factor=0.4, jacobi_niterations=5, vcycle_niterations=6):
        self.jacobi_damping_factor = float(jacobi_damping_factor)
        self.jacobi_niterations = int(jacobi_niterations)
        self.vcycle_niterations = int(vcycle_niterations)

        if self.mpicomm.rank == 0:
            self.log_info('2LPT: Starting first-order solve.')

        # --- 添加常数偏移以避免中心点奇异 ---
        source = self.mesh_delta
        # 计算一个很小的偏移量（基于源项振幅的 1e-8 倍，但至少 1e-8）
        minv = self.mpicomm.allreduce(np.min(source.value))
        maxv = self.mpicomm.allreduce(np.max(source.value))
        offset = 1e-8 * (maxv - minv) + 1e-8
        if self.mpicomm.rank == 0:
            self.log_info(f'Adding offset {offset:.2e} to source.')
        source.value += offset
        # ------------------------------------

        phi1 = self._fmg(source)

        # 检查 phi1
        if np.any(~np.isfinite(phi1.value)):
            if self.mpicomm.rank == 0:
                self.log_warning('phi1 contains NaN/Inf; replacing with zero.')
            phi1.value = np.nan_to_num(phi1.value)

        if self.mpicomm.rank == 0:
            self.log_info('2LPT: Computing tidal source delta2.')

        # Step 2: Compute delta2 from phi1 (unchanged)
        phi1_k = phi1.r2c()
        kx_slabs, ky_slabs, kz_slabs = [], [], []
        for kvec in phi1_k.slabs.x:
            kx_slabs.append(kvec[0])
            ky_slabs.append(kvec[1])
            kz_slabs.append(kvec[2])

        def hessian_component(ki_slabs, kj_slabs):
            comp_k = phi1_k.copy()
            for slab, ki, kj in zip(comp_k.slabs, ki_slabs, kj_slabs):
                slab[...] *= -ki * kj
            return comp_k.c2r()

        phi_xx = hessian_component(kx_slabs, kx_slabs)
        phi_yy = hessian_component(ky_slabs, ky_slabs)
        phi_zz = hessian_component(kz_slabs, kz_slabs)
        phi_xy = hessian_component(kx_slabs, ky_slabs)
        phi_xz = hessian_component(kx_slabs, kz_slabs)
        phi_yz = hessian_component(ky_slabs, kz_slabs)

        delta2_field = phi1.pm.create(type='real', value=0.)
        for (slab, sxx, syy, szz, sxy, sxz, syz) in zip(
                delta2_field.slabs, phi_xx.slabs, phi_yy.slabs, phi_zz.slabs,
                phi_xy.slabs, phi_xz.slabs, phi_yz.slabs):
            slab[...] = (sxx*syy - sxy*sxy) + (sxx*szz - sxz*sxz) + (syy*szz - syz*syz)

        del phi_xx, phi_yy, phi_zz, phi_xy, phi_xz, phi_yz, phi1_k

        # 裁剪 delta2
        delta2_vals = delta2_field.value
        if np.any(~np.isfinite(delta2_vals)):
            if self.mpicomm.rank == 0:
                self.log_warning('delta2 contains NaN/Inf; clipping to [-100, 100].')
            delta2_vals = np.clip(np.nan_to_num(delta2_vals), -100.0, 100.0)
            delta2_field.value = delta2_vals

        if self.mpicomm.rank == 0:
            self.log_info('2LPT: Solving second-order potential.')

        # --- 对 delta2 也加偏移 ---
        minv2 = self.mpicomm.allreduce(np.min(delta2_vals))
        maxv2 = self.mpicomm.allreduce(np.max(delta2_vals))
        offset2 = 1e-8 * (maxv2 - minv2) + 1e-8
        if self.mpicomm.rank == 0:
            self.log_info(f'Adding offset {offset2:.2e} to delta2.')
        delta2_field.value += offset2
        # -------------------------

        phi2 = self._fmg(delta2_field)

        # 检查 phi2
        if np.any(~np.isfinite(phi2.value)):
            if self.mpicomm.rank == 0:
                self.log_warning('phi2 contains NaN/Inf; replacing with zero.')
            phi2.value = np.nan_to_num(phi2.value)

        if self.mpicomm.rank == 0:
            self.log_info(f'2LPT: Combining with D2 = {self.D2:.4f}')

        # Step 4: Combine
        for slab1, slab2 in zip(phi1.slabs, phi2.slabs):
            slab1[...] += self.D2 * slab2

        self.mesh_phi = phi1
        del delta2_field, phi2

        if self.mpicomm.rank == 0:
            self.log_info('2LPT reconstruction completed.')