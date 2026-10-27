"""Second-order (2LPT) extension of :class:`OriginalMultiGridReconstruction`."""

from .recon import ReconstructionError, format_positions_wrapper
from .multigrid import OriginalMultiGridReconstruction
from .second_order import BaseSecondOrderReconstruction, tidal_source, inverse_laplacian
from . import _multigrid


class MultiGridReconstruction2LPT(BaseSecondOrderReconstruction, OriginalMultiGridReconstruction):
    r"""
    :class:`OriginalMultiGridReconstruction` with second-order displacement :math:`\nabla \nabla^{-2} S` added to the Zeldovich displacement,
    see :mod:`second_order`.
    :math:`S` is computed from the Hessian of the (real-space) multigrid potential :attr:`mesh_phi`;
    since :math:`S` is a real-space source, the second-order potential :attr:`mesh_phi2` solves the Poisson equation (without RSD term),
    which is done with FFTs (the multigrid solver is periodic as well).
    """
    def run(self, *args, **kwargs):
        """Run first-order multigrid reconstruction (see :meth:`OriginalMultiGridReconstruction.run`), then compute second-order potential :attr:`mesh_phi2`."""
        super(MultiGridReconstruction2LPT, self).run(*args, **kwargs)
        if self.mpicomm.rank == 0: self.log_info('Computing second-order displacement potential.')
        # read_finite_difference_cic returns +grad, hence mesh_phi2 = lap^{-1} S to get grad lap^{-1} S
        self.mesh_phi2 = inverse_laplacian(tidal_source(self.mesh_phi, is_potential=True))

    @format_positions_wrapper(return_input_type=False)
    def read_shifts(self, positions, field='disp+rsd'):
        """
        Read displacement at input positions by deriving the displacement potentials :attr:`mesh_phi` and :attr:`mesh_phi2` (finite difference scheme),
        with RSD displacement ``f * (psi1 + (1 + D2) * psi2)`` along the line-of-sight.
        See :meth:`BaseReconstruction.read_shifts` for input parameters.
        """
        field = field.lower()
        allowed_fields = ['disp', 'rsd', 'disp+rsd']
        if field not in allowed_fields:
            raise ReconstructionError('Unknown field {}. Choices are {}'.format(field, allowed_fields))
        shifts1 = _multigrid.read_finite_difference_cic(self.mesh_phi, positions, self.boxcenter)
        shifts2 = _multigrid.read_finite_difference_cic(self.mesh_phi2, positions, self.boxcenter)
        return self._combine_shifts(positions, shifts1, shifts2, field=field)
