r"""
Second-order (2LPT) corrections shared by :class:`IterativeFFTReconstruction2LPT` and :class:`MultiGridReconstruction2LPT`.

Reconstruction needs the displacement at the observed (Eulerian) position :math:`\mathbf{x}`, i.e. :math:`\Psi(\mathbf{q}(\mathbf{x}))`.
Expanding mass conservation :math:`1 + \delta = 1 / J` to second order, at :math:`\mathbf{q} = \mathbf{x} - \Psi^{(1)}`, gives:

.. math::

    \Psi(\mathbf{x}) = -\nabla \nabla^{-2} \delta + \nabla \nabla^{-2} S, \quad S = \sum_{i < j} \left(\phi_{,ii} \phi_{,jj} - \phi_{,ij}^{2}\right), \quad \nabla^{2} \phi = \delta

with coefficient exactly 1 in front of :math:`\nabla \nabla^{-2} S`, independently of the second-order growth factor :math:`D_{2}`.
Note this is *not* the forward 2LPT formula :math:`\Psi = -\nabla \phi^{(1)} + D_{2} \nabla \phi^{(2)}`, which applies in Lagrangian space.

The velocity (hence RSD) field is the time derivative at fixed :math:`\mathbf{q}`; the second-order displacement growing as :math:`D_{1}^{2}`:

.. math::

    \mathbf{u} / (f a H) = -\nabla \nabla^{-2} \delta + (1 + D_{2} / D_{1}^{2}) \nabla \nabla^{-2} S

consistent with the difference of SPT kernels :math:`F_{2} - G_{2} = (2/7)(1 - \mu^{2})` for :math:`D_{2} / D_{1}^{2} = -3/7`.
"""

import numpy as np

from . import utils


def get_D2(omega_m=1.):
    r"""
    Return the second-order growth factor :math:`D_{2} / D_{1}^{2} \simeq -3/7 \Omega_{m}^{-1/143}`
    (Bouchet et al. 1995, https://arxiv.org/abs/astro-ph/9406013), with :math:`\Omega_{m}` taken at the redshift of the sample.
    """
    return -3. / 7. * omega_m ** (-1. / 143.)


def tidal_source(field, is_potential=False):
    r"""
    Return second-order source :math:`S = \sum_{i < j} (\phi_{,ii} \phi_{,jj} - \phi_{,ij}^{2})` (as a real field).

    Parameters
    ----------
    field : RealField
        Density contrast :math:`\delta`, or displacement potential if ``is_potential``
        (only the Hessian matters, up to a sign, which :math:`S` is insensitive to).
        Must be a field whose buffer is consistent with FFTs (i.e. not modified with ``field.value = array``).

    is_potential : bool, default=False
        Whether ``field`` is a potential (Hessian is :math:`-k_{i} k_{j} \phi(\mathbf{k})`)
        rather than a density (Hessian is :math:`k_{i} k_{j} \delta(\mathbf{k}) / k^{2}`).
    """
    field_k = field.r2c()
    nmesh = field.pm.Nmesh
    ndim = len(nmesh)
    hessian = {}
    for iaxis in range(ndim):
        for jaxis in range(iaxis, ndim):
            comp = field_k.copy()
            for kslab, islab, slab in zip(comp.slabs.x, comp.slabs.i, comp.slabs):
                # as in IterativeFFTReconstruction._iterate: zero odd-derivative Nyquist modes
                mask = (islab[iaxis] != nmesh[iaxis] // 2) & (islab[jaxis] != nmesh[jaxis] // 2)
                mask |= (islab[iaxis] == nmesh[iaxis] // 2) & (islab[jaxis] == nmesh[jaxis] // 2)
                if is_potential:
                    slab[...] *= -kslab[iaxis] * kslab[jaxis] * mask
                else:
                    slab[...] *= utils.safe_divide(kslab[iaxis] * kslab[jaxis] * mask, sum(kk**2 for kk in kslab))
            hessian[iaxis, jaxis] = comp.c2r()
            del comp
    del field_k
    toret = field.pm.create(type='real', value=0.)
    for iaxis in range(ndim):
        for jaxis in range(iaxis + 1, ndim):
            toret[...] += hessian[iaxis, iaxis] * hessian[jaxis, jaxis] - hessian[iaxis, jaxis]**2
    return toret


def inverse_laplacian(field):
    r"""Return :math:`\nabla^{-2}` ``field`` (zero mean), i.e. :math:`-\mathrm{field}(\mathbf{k}) / k^{2}`."""
    field_k = field.r2c()
    for kslab, slab in zip(field_k.slabs.x, field_k.slabs):
        utils.safe_divide(slab, -sum(kk**2 for kk in kslab), inplace=True)
    return field_k.c2r()


def gradient_inverse_laplacian(field):
    r"""Return the three components of :math:`\nabla \nabla^{-2}` ``field``, i.e. :math:`-i \mathbf{k} \, \mathrm{field}(\mathbf{k}) / k^{2}`."""
    field_k = field.r2c()
    nmesh = field.pm.Nmesh
    toret = []
    for iaxis in range(len(nmesh)):
        comp = field_k.copy()
        for kslab, islab, slab in zip(comp.slabs.x, comp.slabs.i, comp.slabs):
            mask = islab[iaxis] != nmesh[iaxis] // 2
            slab[...] *= -1j * utils.safe_divide(kslab[iaxis], sum(kk**2 for kk in kslab)) * mask
        toret.append(comp.c2r())
        del comp
    return toret


class BaseSecondOrderReconstruction(object):
    """
    Mixin adding second-order growth factor :attr:`D2` and combination of first- and second-order shifts.
    Must come before the reconstruction class in the bases.
    """
    def __init__(self, *args, D2=None, omega_m=1., **kwargs):
        r"""
        Parameters
        ----------
        D2 : float, default=None
            Second-order growth factor :math:`D_{2} / D_{1}^{2}`, only used for the RSD (velocity) part of the shifts.
            If ``None``, computed from ``omega_m`` with :func:`get_D2`.

        omega_m : float, default=1.
            Matter density parameter :math:`\Omega_{m}(z)` at the redshift of the sample, used if ``D2`` is ``None``.
            Its impact is tiny: :math:`\Omega_{m}^{-1/143}`.

        args, kwargs : tuple, dict
            Arguments for the first-order reconstruction class.
        """
        super(BaseSecondOrderReconstruction, self).__init__(*args, **kwargs)
        self.D2 = get_D2(omega_m) if D2 is None else float(D2)

    def _combine_shifts(self, positions, shifts1, shifts2, field='disp+rsd'):
        """
        Combine first-order (Zeldovich) ``shifts1`` and second-order ``shifts2`` shifts read at ``positions``.
        The displacement is ``shifts1 + shifts2``; the RSD displacement is ``f * (shifts1 + (1 + D2) * shifts2)`` projected along the line-of-sight.
        """
        shifts = shifts1 + shifts2
        if field == 'disp':
            return shifts
        if self.los is None:
            los = utils.safe_divide(positions, utils.distance(positions)[:, None])
        else:
            los = self.los.astype(shifts.dtype)
        f = self.f
        if getattr(self, 'f_callable', None) is not None:
            f = self.f_callable(utils.distance(positions))[..., None]
        velocity = shifts1 + (1. + self.D2) * shifts2
        rsd = f * (np.sum(velocity * los, axis=-1)[:, None] * los)
        if field == 'rsd':
            return rsd
        # field == 'disp+rsd'
        shifts += rsd
        return shifts
