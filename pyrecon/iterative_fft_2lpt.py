"""Second-order (2LPT) extension of :class:`IterativeFFTReconstruction`."""

import numpy as np

from .recon import ReconstructionError, format_positions_wrapper
from .iterative_fft import IterativeFFTReconstruction
from .second_order import BaseSecondOrderReconstruction, tidal_source, gradient_inverse_laplacian


class IterativeFFTReconstruction2LPT(BaseSecondOrderReconstruction, IterativeFFTReconstruction):
    r"""
    :class:`IterativeFFTReconstruction` with second-order displacement :math:`\nabla \nabla^{-2} S` added to the Zeldovich displacement,
    see :mod:`second_order`. The iterative (linear) RSD removal is unchanged.
    Zeldovich and second-order displacements are stored in :attr:`mesh_psi` and :attr:`mesh_psi2`, respectively.
    """
    def _compute_psi(self):
        # Computed from the real-space density, before it is deleted in run()
        self.mesh_psi2 = gradient_inverse_laplacian(tidal_source(self.mesh_delta_real))
        return super(IterativeFFTReconstruction2LPT, self)._compute_psi()

    @format_positions_wrapper(return_input_type=False)
    def read_shifts(self, positions, field='disp+rsd'):
        """
        Read displacement at input positions: Zeldovich + second-order displacements,
        with RSD displacement ``f * (psi1 + (1 + D2) * psi2)`` along the line-of-sight.
        See :meth:`BaseReconstruction.read_shifts` for input parameters.
        """
        field = field.lower()
        allowed_fields = ['disp', 'rsd', 'disp+rsd']
        if field not in allowed_fields:
            raise ReconstructionError('Unknown field {}. Choices are {}'.format(field, allowed_fields))
        shifts1, shifts2 = np.empty_like(positions), np.empty_like(positions)
        for iaxis, (psi1, psi2) in enumerate(zip(self.mesh_psi, self.mesh_psi2)):
            shifts1[:, iaxis] = self._readout(psi1, positions)
            shifts2[:, iaxis] = self._readout(psi2, positions)
        return self._combine_shifts(positions, shifts1, shifts2, field=field)
