"""Contains functions for calculating induced velocities.

Three backend modes available via PTERASOFTWARE_BACKEND environment variable: serial_cpu
(default), parallel_cpu (multi-core), or gpu_cuda (GPU).

See pterasoftware._gpu_config for configuration details.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

# Import configuration for backend dispatch
from pterasoftware._gpu_config import get_config, select_biot_savart_function

# Squire's parameter relates to the size of the vortex cores and the rate at which they
# grow. The value of this parameter is slightly controversial. It dramatically affects
# the stability of the result. I'm using this value, as cited for use in flapping-wing
# vehicles in "Role of Filament Strain in the Free-Vortex Modeling of Rotor Wakes"
# (Ananthan and Leishman, 2004). It is unitless.
_squire = 1.0e-4

# Lamb's constant relates to the size of the vortex cores and the rate at which they
# grow. The value of this parameter is well agreed upon, and published in "Extended
# Unsteady Vortex-Lattice Method for Insect Flapping Wings" (Nguyen et al., 2016). It is
# unitless.
_lamb = 1.25643

# The local machine error is used to detect degenerate (zero length) LineVortices.
_eps = np.finfo(float).eps

# The relative tolerance for scale invariant singularity checks. It provides two orders
# of magnitude of safety margin before catastrophic cancellation in 1.0 - cos(theta)
# begins at theta ~ 2.1e-8.
_tol = 1.0e-10

# Pre compute 4 * pi and 4.0 * _lamb as they used repeatedly.
_four_pi = 4.0 * math.pi
_four_lamb = 4.0 * _lamb


@njit(cache=True, fastmath=True)
def collapsed_velocities_from_ring_vortices(
    stackP_GP1_CgP1: np.ndarray,
    stackBrrvp_GP1_CgP1: np.ndarray,
    stackFrrvp_GP1_CgP1: np.ndarray,
    stackFlrvp_GP1_CgP1: np.ndarray,
    stackBlrvp_GP1_CgP1: np.ndarray,
    strengths: np.ndarray,
    r_c0s: np.ndarray,
    singularity_counts: np.ndarray,
    ages: np.ndarray | None = None,
    nu: float = 0.0,
) -> np.ndarray:
    """Takes in a group of points and the attributes of a group of RingVortices and
    finds the cumulative induced velocity at every point.

    This function's performance has been highly optimized for unsteady simulations via
    Numba. While using Numba dramatically increases unsteady simulation performance, it
    does cause a performance drop for the less intense steady simulations.

    :param stackP_GP1_CgP1: A (N,3) ndarray of floats representing the positions of N
        points (in the first Airplane's geometry axes, relative to the first Airplane's
        CG). The units are in meters.
    :param stackBrrvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        M RingVortices' back right vertices (in the first Airplane's geometry axes,
        relative to the first Airplane's CG). The units are in meters.
    :param stackFrrvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        the M RingVortices' front right vertices (in the first Airplane's geometry axes,
        relative to the first Airplane's CG). The units are in meters.
    :param stackFlrvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        the M RingVortices' front left vertices (in the first Airplane's geometry axes,
        relative to the first Airplane's CG). The units are in meters.
    :param stackBlrvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        the M RingVortices' back left vertices (in the first Airplane's geometry axes,
        relative to the first Airplane's CG). The units are in meters.
    :param strengths: A (M,) ndarray of floats representing the strengths of the M
        RingVortices. The units are in meters squared per second.
    :param r_c0s: A (M,) ndarray of floats representing the initial core radii of the M
        RingVortices. Based on results from Ramasamy and Leishman (2007), a reasonable
        value that works across scales is 3.0% the chord length of each LineVortices'
        parent Wing. The units are in meters.
    :param singularity_counts: A (4,) ndarray of int64 representing the cumulative
        counts of singularity events. Index mapping: [0] degenerate filament, [1] vertex
        start proximity, [2] vertex end proximity, [3] collinearity. Counts are
        incremented in place and accumulate across all four legs.
    :param ages: For bound RingVortices, this must be None. For RingVortices that have
        been shed into the wake, it must be a (M,) ndarray of floats representing the
        ages of the M RingVortices in seconds. The default is None.
    :param nu: A non negative float representing the kinematic viscosity of the fluid.
        The units are in meters squared per second. The default is 0.0.
    :return: A (N,3) ndarray of floats for the cumulative induced velocity at each of
        the N points (in the first Airplane's geometry axes, observed from the Earth
        frame). The units are in meters per second.
    """
    stackVInd_GP1__E = np.zeros((stackP_GP1_CgP1.shape[0], 3))

    # Get the velocity induced by each leg of each RingVortex (in the first Airplane's
    # geometry axes, observed from the Earth frame). Unrolled loop eliminates list
    # allocation overhead and array indexing (estimated 2-3% improvement).
    stackVInd_GP1__E += _collapsed_velocities_from_line_vortices(
        stackP_GP1_CgP1=stackP_GP1_CgP1,
        stackSlvp_GP1_CgP1=stackBrrvp_GP1_CgP1,
        stackElvp_GP1_CgP1=stackFrrvp_GP1_CgP1,
        strengths=strengths,
        r_c0s=r_c0s,
        singularity_counts=singularity_counts,
        ages=ages,
        nu=nu,
    )
    stackVInd_GP1__E += _collapsed_velocities_from_line_vortices(
        stackP_GP1_CgP1=stackP_GP1_CgP1,
        stackSlvp_GP1_CgP1=stackFrrvp_GP1_CgP1,
        stackElvp_GP1_CgP1=stackFlrvp_GP1_CgP1,
        strengths=strengths,
        r_c0s=r_c0s,
        singularity_counts=singularity_counts,
        ages=ages,
        nu=nu,
    )
    stackVInd_GP1__E += _collapsed_velocities_from_line_vortices(
        stackP_GP1_CgP1=stackP_GP1_CgP1,
        stackSlvp_GP1_CgP1=stackFlrvp_GP1_CgP1,
        stackElvp_GP1_CgP1=stackBlrvp_GP1_CgP1,
        strengths=strengths,
        r_c0s=r_c0s,
        singularity_counts=singularity_counts,
        ages=ages,
        nu=nu,
    )
    stackVInd_GP1__E += _collapsed_velocities_from_line_vortices(
        stackP_GP1_CgP1=stackP_GP1_CgP1,
        stackSlvp_GP1_CgP1=stackBlrvp_GP1_CgP1,
        stackElvp_GP1_CgP1=stackBrrvp_GP1_CgP1,
        strengths=strengths,
        r_c0s=r_c0s,
        singularity_counts=singularity_counts,
        ages=ages,
        nu=nu,
    )
    return stackVInd_GP1__E


@njit(
    cache=True, fastmath=True
)  # Phase 3.4+: fastmath safe with explicit singularity checks
def collapsed_velocities_from_ring_vortices_chordwise_segments(
    stackP_GP1_CgP1: np.ndarray,
    stackBrrvp_GP1_CgP1: np.ndarray,
    stackFrrvp_GP1_CgP1: np.ndarray,
    stackFlrvp_GP1_CgP1: np.ndarray,
    stackBlrvp_GP1_CgP1: np.ndarray,
    strengths: np.ndarray,
    r_c0s: np.ndarray,
    singularity_counts: np.ndarray,
    ages: np.ndarray | None = None,
    nu: float = 0.0,
) -> np.ndarray:
    """Takes in a group of points and the attributes of a group of RingVortices and
    finds the cumulative induced velocity at every point due to the RingVortices' left
    and right LineVortex legs.

    This function's performance has been highly optimized for unsteady simulations via
    Numba. While using Numba dramatically increases unsteady simulation performance, it
    does cause a performance drop for the less intense steady simulations.

    :param stackP_GP1_CgP1: A (N,3) ndarray of floats representing the positions of N
        points (in the first Airplane's geometry axes, relative to the first Airplane's
        CG). The units are in meters.
    :param stackBrrvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        M RingVortices' back right vertices (in the first Airplane's geometry axes,
        relative to the first Airplane's CG). The units are in meters.
    :param stackFrrvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        the M RingVortices' front right vertices (in the first Airplane's geometry axes,
        relative to the first Airplane's CG). The units are in meters.
    :param stackFlrvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        the M RingVortices' front left vertices (in the first Airplane's geometry axes,
        relative to the first Airplane's CG). The units are in meters.
    :param stackBlrvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        the M RingVortices' back left vertices (in the first Airplane's geometry axes,
        relative to the first Airplane's CG). The units are in meters.
    :param strengths: A (M,) ndarray of floats representing the strengths of the M
        RingVortices. The units are in meters squared per second.
    :param r_c0s: A (M,) ndarray of floats representing the initial core radii of the M
        RingVortices. Based on results from Ramasamy and Leishman (2007), a reasonable
        value that works across scales is 3.0% the chord length of each LineVortices'
        parent Wing. The units are in meters.
    :param singularity_counts: A (4,) ndarray of int64 representing the cumulative
        counts of singularity events. Index mapping: [0] degenerate filament, [1] vertex
        start proximity, [2] vertex end proximity, [3] collinearity. Counts are
        incremented in place and accumulate across both legs.
    :param ages: For bound RingVortices, this must be None. For RingVortices that have
        been shed into the wake, it must be a (M,) ndarray of floats representing the
        ages of the M RingVortices in seconds. The default is None.
    :param nu: A non negative float representing the kinematic viscosity of the fluid.
        The units are in meters squared per second. The default is 0.0.
    :return: A (N,3) ndarray of floats for the cumulative induced velocity at each of
        the N points (in the first Airplane's geometry axes, observed from the Earth
        frame) due to the M RingVortices' left and right LineVortex legs. The units are
        in meters per second.
    """
    stackVInd_GP1__E = np.zeros((stackP_GP1_CgP1.shape[0], 3))

    # Get the velocity induced by the left and right legs of each RingVortex (in the
    # first Airplane's geometry axes, observed from the Earth frame). Unrolled loop
    # eliminates list allocation overhead and array indexing (estimated 1-2% improvement).
    stackVInd_GP1__E += _collapsed_velocities_from_line_vortices(
        stackP_GP1_CgP1=stackP_GP1_CgP1,
        stackSlvp_GP1_CgP1=stackBrrvp_GP1_CgP1,
        stackElvp_GP1_CgP1=stackFrrvp_GP1_CgP1,
        strengths=strengths,
        r_c0s=r_c0s,
        singularity_counts=singularity_counts,
        ages=ages,
        nu=nu,
    )
    stackVInd_GP1__E += _collapsed_velocities_from_line_vortices(
        stackP_GP1_CgP1=stackP_GP1_CgP1,
        stackSlvp_GP1_CgP1=stackFlrvp_GP1_CgP1,
        stackElvp_GP1_CgP1=stackBlrvp_GP1_CgP1,
        strengths=strengths,
        r_c0s=r_c0s,
        singularity_counts=singularity_counts,
        ages=ages,
        nu=nu,
    )
    return stackVInd_GP1__E


@njit(cache=True, fastmath=False)
def expanded_velocities_from_ring_vortices(
    stackP_GP1_CgP1: np.ndarray,
    stackBrrvp_GP1_CgP1: np.ndarray,
    stackFrrvp_GP1_CgP1: np.ndarray,
    stackFlrvp_GP1_CgP1: np.ndarray,
    stackBlrvp_GP1_CgP1: np.ndarray,
    strengths: np.ndarray,
    r_c0s: np.ndarray,
    singularity_counts: np.ndarray,
    ages: np.ndarray | None = None,
    nu: float = 0.0,
) -> np.ndarray:
    """Takes in a group of points and the attributes of a group of RingVortices and
    finds the induced velocity at every point due to each RingVortex.

    This function's performance has been highly optimized for unsteady simulations via
    Numba. While using Numba dramatically increases unsteady simulation performance, it
    does cause a performance drop for the less intense steady simulations.

    :param stackP_GP1_CgP1: A (N,3) ndarray of floats representing the positions of N
        points (in the first Airplane's geometry axes, relative to the first Airplane's
        CG). The units are in meters.
    :param stackBrrvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        M RingVortices' back right vertices (in the first Airplane's geometry axes,
        relative to the first Airplane's CG). The units are in meters.
    :param stackFrrvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        the M RingVortices' front right vertices (in the first Airplane's geometry axes,
        relative to the first Airplane's CG). The units are in meters.
    :param stackFlrvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        the M RingVortices' front left vertices (in the first Airplane's geometry axes,
        relative to the first Airplane's CG). The units are in meters.
    :param stackBlrvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        the M RingVortices' back left vertices (in the first Airplane's geometry axes,
        relative to the first Airplane's CG). The units are in meters.
    :param strengths: A (M,) ndarray of floats representing the strengths of the M
        RingVortices. The units are in meters squared per second.
    :param r_c0s: A (M,) ndarray of floats representing the initial core radii of the M
        RingVortices. Based on results from Ramasamy and Leishman (2007), a reasonable
        value that works across scales is 3.0% the chord length of each LineVortices'
        parent Wing. The units are in meters.
    :param singularity_counts: A (4,) ndarray of int64 representing the cumulative
        counts of singularity events. Index mapping: [0] degenerate filament, [1] vertex
        start proximity, [2] vertex end proximity, [3] collinearity. Counts are
        incremented in place and accumulate across all four legs.
    :param ages: For bound RingVortices, this must be None. For RingVortices that have
        been shed into the wake, it must be a (M,) ndarray of floats representing the
        ages of the M RingVortices in seconds. The default is None.
    :param nu: A non negative float representing the kinematic viscosity of the fluid.
        The units are in meters squared per second. The default is 0.0.
    :return: A (N,M,3) ndarray of floats for the induced velocity at each of the N
        points (in the first Airplane's geometry axes, observed from the Earth frame)
        due to each of the M RingVortices. The units are in meters per second.
    """
    gridVInd_GP1__E = np.zeros((stackP_GP1_CgP1.shape[0], strengths.shape[0], 3))

    # Get the velocity induced by each leg of each RingVortex (in the first Airplane's
    # geometry axes, observed from the Earth frame). Unrolled loop eliminates list
    # allocation overhead and array indexing (estimated 2-3% improvement).
    gridVInd_GP1__E += _expanded_velocities_from_line_vortices(
        stackP_GP1_CgP1=stackP_GP1_CgP1,
        stackSlvp_GP1_CgP1=stackBrrvp_GP1_CgP1,
        stackElvp_GP1_CgP1=stackFrrvp_GP1_CgP1,
        strengths=strengths,
        r_c0s=r_c0s,
        singularity_counts=singularity_counts,
        ages=ages,
        nu=nu,
    )
    gridVInd_GP1__E += _expanded_velocities_from_line_vortices(
        stackP_GP1_CgP1=stackP_GP1_CgP1,
        stackSlvp_GP1_CgP1=stackFrrvp_GP1_CgP1,
        stackElvp_GP1_CgP1=stackFlrvp_GP1_CgP1,
        strengths=strengths,
        r_c0s=r_c0s,
        singularity_counts=singularity_counts,
        ages=ages,
        nu=nu,
    )
    gridVInd_GP1__E += _expanded_velocities_from_line_vortices(
        stackP_GP1_CgP1=stackP_GP1_CgP1,
        stackSlvp_GP1_CgP1=stackFlrvp_GP1_CgP1,
        stackElvp_GP1_CgP1=stackBlrvp_GP1_CgP1,
        strengths=strengths,
        r_c0s=r_c0s,
        singularity_counts=singularity_counts,
        ages=ages,
        nu=nu,
    )
    gridVInd_GP1__E += _expanded_velocities_from_line_vortices(
        stackP_GP1_CgP1=stackP_GP1_CgP1,
        stackSlvp_GP1_CgP1=stackBlrvp_GP1_CgP1,
        stackElvp_GP1_CgP1=stackBrrvp_GP1_CgP1,
        strengths=strengths,
        r_c0s=r_c0s,
        singularity_counts=singularity_counts,
        ages=ages,
        nu=nu,
    )
    return gridVInd_GP1__E


# TODO: Remove the ability to specify HorseshoeVortex ages they are never used in
#  unsteady simulations.
@njit(cache=True, fastmath=False)
def collapsed_velocities_from_horseshoe_vortices(
    stackP_GP1_CgP1: np.ndarray,
    stackBrhvp_GP1_CgP1: np.ndarray,
    stackFrhvp_GP1_CgP1: np.ndarray,
    stackFlhvp_GP1_CgP1: np.ndarray,
    stackBlhvp_GP1_CgP1: np.ndarray,
    strengths: np.ndarray,
    r_c0s: np.ndarray,
    singularity_counts: np.ndarray,
    nu: float = 0.0,
) -> np.ndarray:
    """Takes in a group of points and the attributes of a group of HorseshoeVortices and
    finds the cumulative induced velocity at every point.

    This function's performance has been highly optimized for unsteady simulations via
    Numba. While using Numba dramatically increases unsteady simulation performance, it
    does cause a performance drop for the less intense steady simulations.

    :param stackP_GP1_CgP1: A (N,3) ndarray of floats representing the positions of N
        points (in the first Airplane's geometry axes, relative to the first Airplane's
        CG). The units are in meters.
    :param stackBrhvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        M HorseshoeVortices' back right vertices (in the first Airplane's geometry axes,
        relative to the first Airplane's CG). The units are in meters.
    :param stackFrhvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        the M HorseshoeVortices' front right vertices (in the first Airplane's geometry
        axes, relative to the first Airplane's CG). The units are in meters.
    :param stackFlhvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        the M HorseshoeVortices' front left vertices (in the first Airplane's geometry
        axes, relative to the first Airplane's CG). The units are in meters.
    :param stackBlhvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        the M HorseshoeVortices' back left vertices (in the first Airplane's geometry
        axes, relative to the first Airplane's CG). The units are in meters.
    :param strengths: A (M,) ndarray of floats representing the strengths of the M
        HorseshoeVortices. The units are in meters squared per second.
    :param r_c0s: A (M,) ndarray of floats representing the initial core radii of the M
        HorseshoeVortices. Based on results from Ramasamy and Leishman (2007), a
        reasonable value that works across scales is 3.0% the chord length of each
        LineVortices' parent Wing. The units are in meters.
    :param singularity_counts: A (4,) ndarray of int64 representing the cumulative
        counts of singularity events. Index mapping: [0] degenerate filament, [1] vertex
        start proximity, [2] vertex end proximity, [3] collinearity. Counts are
        incremented in place and accumulate across all three legs.
    :param nu: A non negative float representing the kinematic viscosity of the fluid.
        The units are in meters squared per second. The default is 0.0.
    :return: A (N,3) ndarray of floats for the cumulative induced velocity at each of
        the N points (in the first Airplane's geometry axes, observed from the Earth
        frame). The units are in meters per second.
    """
    listStackSlvp_GP1_CgP1 = [
        stackBrhvp_GP1_CgP1,
        stackFrhvp_GP1_CgP1,
        stackFlhvp_GP1_CgP1,
    ]
    listStackElvp_GP1_CgP1 = [
        stackFrhvp_GP1_CgP1,
        stackFlhvp_GP1_CgP1,
        stackBlhvp_GP1_CgP1,
    ]

    stackVInd_GP1__E = np.zeros((stackP_GP1_CgP1.shape[0], 3))

    # Get the velocity induced by each leg of each HorseshoeVortex (in the first
    # Airplane's geometry axes, observed from the Earth frame). Unrolled loop eliminates
    # list allocation overhead and array indexing (estimated 2-3% improvement).
    stackVInd_GP1__E += _collapsed_velocities_from_line_vortices(
        stackP_GP1_CgP1=stackP_GP1_CgP1,
        stackSlvp_GP1_CgP1=stackBrhvp_GP1_CgP1,
        stackElvp_GP1_CgP1=stackFrhvp_GP1_CgP1,
        strengths=strengths,
        r_c0s=r_c0s,
        singularity_counts=singularity_counts,
        ages=None,
        nu=nu,
    )
    stackVInd_GP1__E += _collapsed_velocities_from_line_vortices(
        stackP_GP1_CgP1=stackP_GP1_CgP1,
        stackSlvp_GP1_CgP1=stackFrhvp_GP1_CgP1,
        stackElvp_GP1_CgP1=stackFlhvp_GP1_CgP1,
        strengths=strengths,
        r_c0s=r_c0s,
        singularity_counts=singularity_counts,
        ages=None,
        nu=nu,
    )
    stackVInd_GP1__E += _collapsed_velocities_from_line_vortices(
        stackP_GP1_CgP1=stackP_GP1_CgP1,
        stackSlvp_GP1_CgP1=stackFlhvp_GP1_CgP1,
        stackElvp_GP1_CgP1=stackBlhvp_GP1_CgP1,
        strengths=strengths,
        r_c0s=r_c0s,
        singularity_counts=singularity_counts,
        ages=None,
        nu=nu,
    )
    return stackVInd_GP1__E


# TODO: Remove the ability to specify HorseshoeVortex ages they are never used in
#  unsteady simulations.
@njit(cache=True, fastmath=False)
def expanded_velocities_from_horseshoe_vortices(
    stackP_GP1_CgP1: np.ndarray,
    stackBrhvp_GP1_CgP1: np.ndarray,
    stackFrhvp_GP1_CgP1: np.ndarray,
    stackFlhvp_GP1_CgP1: np.ndarray,
    stackBlhvp_GP1_CgP1: np.ndarray,
    strengths: np.ndarray,
    r_c0s: np.ndarray,
    singularity_counts: np.ndarray,
    nu: float = 0.0,
) -> np.ndarray:
    """Takes in a group of points and the attributes of a group of HorseshoeVortices and
    finds the induced velocity at every point due to each HorseshoeVortex.

    This function's performance has been highly optimized for unsteady simulations via
    Numba. While using Numba dramatically increases unsteady simulation performance, it
    does cause a performance drop for the less intense steady simulations.

    :param stackP_GP1_CgP1: A (N,3) ndarray of floats representing the positions of N
        points (in the first Airplane's geometry axes, relative to the first Airplane's
        CG). The units are in meters.
    :param stackBrhvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        M HorseshoeVortices' back right vertices (in the first Airplane's geometry axes,
        relative to the first Airplane's CG). The units are in meters.
    :param stackFrhvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        the M HorseshoeVortices' front right vertices (in the first Airplane's geometry
        axes, relative to the first Airplane's CG). The units are in meters.
    :param stackFlhvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        the M HorseshoeVortices' front left vertices (in the first Airplane's geometry
        axes, relative to the first Airplane's CG). The units are in meters.
    :param stackBlhvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        the M HorseshoeVortices' back left vertices (in the first Airplane's geometry
        axes, relative to the first Airplane's CG). The units are in meters.
    :param strengths: A (M,) ndarray of floats representing the strengths of M
        HorseshoeVortices. The units are in meters squared per second.
    :param r_c0s: A (M,) ndarray of floats representing the initial core radii of the M
        HorseshoeVortices. Based on results from Ramasamy and Leishman (2007), a
        reasonable value that works across scales is 3.0% the chord length of each
        LineVortices' parent Wing. The units are in meters.
    :param singularity_counts: A (4,) ndarray of int64 representing the cumulative
        counts of singularity events. Index mapping: [0] degenerate filament, [1] vertex
        start proximity, [2] vertex end proximity, [3] collinearity. Counts are
        incremented in place and accumulate across all three legs.
    :param nu: A non negative float representing the kinematic viscosity of the fluid.
        The units are in meters squared per second. The default is 0.0.
    :return: A (N,M,3) ndarray of floats for the induced velocity at each of the N
        points (in the first Airplane's geometry axes, observed from the Earth frame)
        due to each of the M HorseshoeVortices. The units are in meters per second.
    """
    gridVInd_GP1__E = np.zeros((stackP_GP1_CgP1.shape[0], strengths.shape[0], 3))

    # Get the velocity induced by each leg of each HorseshoeVortex (in the first
    # Airplane's geometry axes, observed from the Earth frame). Unrolled loop eliminates
    # list allocation overhead and array indexing (estimated 2-3% improvement).
    gridVInd_GP1__E += _expanded_velocities_from_line_vortices(
        stackP_GP1_CgP1=stackP_GP1_CgP1,
        stackSlvp_GP1_CgP1=stackBrhvp_GP1_CgP1,
        stackElvp_GP1_CgP1=stackFrhvp_GP1_CgP1,
        strengths=strengths,
        r_c0s=r_c0s,
        singularity_counts=singularity_counts,
        ages=None,
        nu=nu,
    )
    gridVInd_GP1__E += _expanded_velocities_from_line_vortices(
        stackP_GP1_CgP1=stackP_GP1_CgP1,
        stackSlvp_GP1_CgP1=stackFrhvp_GP1_CgP1,
        stackElvp_GP1_CgP1=stackFlhvp_GP1_CgP1,
        strengths=strengths,
        r_c0s=r_c0s,
        singularity_counts=singularity_counts,
        ages=None,
        nu=nu,
    )
    gridVInd_GP1__E += _expanded_velocities_from_line_vortices(
        stackP_GP1_CgP1=stackP_GP1_CgP1,
        stackSlvp_GP1_CgP1=stackFlhvp_GP1_CgP1,
        stackElvp_GP1_CgP1=stackBlhvp_GP1_CgP1,
        strengths=strengths,
        r_c0s=r_c0s,
        singularity_counts=singularity_counts,
        ages=None,
        nu=nu,
    )
    return gridVInd_GP1__E


@njit(
    cache=True, fastmath=True
)  # Phase 3.4+: fastmath safe (explicit singularity checks)
def _collapsed_velocities_from_line_vortices(
    stackP_GP1_CgP1: np.ndarray,
    stackSlvp_GP1_CgP1: np.ndarray,
    stackElvp_GP1_CgP1: np.ndarray,
    strengths: np.ndarray,
    r_c0s: np.ndarray,
    singularity_counts: np.ndarray,
    ages: np.ndarray | None = None,
    nu: float = 0.0,
) -> np.ndarray:
    """Takes in a group of points and the attributes of a group of LineVortices and
    finds the cumulative induced velocity at every point.

    This function uses a modified version of the Biot-Savart law to create a smooth
    induced velocity decay based on a LineVortex's core radius. The core radius grows
    from an initial value based on the LineVortex's age.

    This function's performance has been highly optimized for unsteady simulations via
    Numba. While using Numba dramatically increases unsteady simulation performance, it
    does cause a performance drop for the less intense steady simulations.

    **Citation:**

    Core radius equation adapted from Eq. 3 of: "A Reynolds Number-Based Blade Tip
    Vortex Model"

    Authors: Manikandan Ramasamy and J. Gordon Leishman

    Biot-Savart equation adapted from: "Extended Unsteady Vortex-Lattice Method for
    Insect Flapping Wings"

    Authors: Anh Tuan Nguyen, Joong-Kwan Kim, Jong-Seob Han, and Jae-Hung Han

    :param stackP_GP1_CgP1: A (N,3) ndarray of floats representing the positions of N
        points (in the first Airplane's geometry axes, relative to the first Airplane's
        CG). The units are in meters.
    :param stackSlvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        the M LineVortices' starting vertices (in the first Airplane's geometry axes,
        relative to the first Airplane's CG). The units are in meters.
    :param stackElvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        the M LineVortices' ending vertices (in the first Airplane's geometry axes,
        relative to the first Airplane's CG). The units are in meters.
    :param strengths: A (M,) ndarray of floats representing the strengths of the M
        LineVortices. The units are in meters squared per second.
    :param r_c0s: A (M,) ndarray of floats representing the initial core radii of the M
        LineVortices. Based on results from Ramasamy and Leishman (2007), a reasonable
        value that works across scales is 3.0% the chord length of each LineVortices'
        parent Wing. The units are in meters.
    :param singularity_counts: A (4,) ndarray of int64 representing the cumulative
        counts of singularity events. Index mapping: [0] degenerate filament, [1] vertex
        start proximity, [2] vertex end proximity, [3] collinearity. Counts are
        incremented in place and accumulate across calls.
    :param ages: For bound LineVortices, this must be None. For LineVortices that have
        been shed into the wake, it must be a (M,) ndarray of floats representing the
        ages of the M LineVortices in seconds. The default is None.
    :param nu: A non negative float representing the kinematic viscosity of the fluid.
        The units are in meters squared per second. The default is 0.0.
    :return: A (N,3) ndarray of floats for the cumulative induced velocity at each of
        the N points (in the first Airplane's geometry axes, observed from the Earth
        frame). The units are in meters per second.
    """
    num_vortices = stackSlvp_GP1_CgP1.shape[0]
    num_points = stackP_GP1_CgP1.shape[0]

    # Initialize an empty array, which we will fill with the induced velocities (in
    # the first Airplane's geometry axes, observed from the Earth frame).
    stackVInd_GP1__E = np.zeros((num_points, 3))

    # If the user didn't specify any ages, set the age of each LineVortex to 0.0
    # seconds.
    if ages is None:
        ages = np.zeros(num_vortices)

    for vortex_id in range(num_vortices):
        Slvp_GP1_CgP1 = stackSlvp_GP1_CgP1[vortex_id]
        Elvp_GP1_CgP1 = stackElvp_GP1_CgP1[vortex_id]

        # The r0_GP1 vector goes from the LineVortex's start point to its end point (in
        # the first Airplane's geometry axes).
        r0X_GP1 = Elvp_GP1_CgP1[0] - Slvp_GP1_CgP1[0]
        r0Y_GP1 = Elvp_GP1_CgP1[1] - Slvp_GP1_CgP1[1]
        r0Z_GP1 = Elvp_GP1_CgP1[2] - Slvp_GP1_CgP1[2]

        # Find r0_GP1's length.
        r0 = math.sqrt(r0X_GP1**2.0 + r0Y_GP1**2.0 + r0Z_GP1**2.0)

        # Skip degenerate filaments where the start and end points coincide.
        if r0 < _eps:
            singularity_counts[0] += 1
            continue

        strength = strengths[vortex_id]
        age = ages[vortex_id]
        r_c0 = r_c0s[vortex_id]

        # Pre compute r0 * _tol outside the inner loop.
        r0_times_tol = r0 * _tol

        # Calculate the radius of the LineVortex's core squared. The initial core radius
        # ensures nonzero regularization even for bound vortices with zero age.
        r_c_sq = r_c0**2.0 + _four_lamb * (nu + _squire * abs(strength)) * age

        c_1 = strength / _four_pi
        c_2 = r0**2.0 * r_c_sq

        for point_id in range(num_points):
            P_GP1_CgP1 = stackP_GP1_CgP1[point_id]

            # The r1_GP1 vector goes from P_GP1_CgP1 to the LineVortex's start point (in
            # the first Airplane's geometry axes).
            r1X_GP1 = Slvp_GP1_CgP1[0] - P_GP1_CgP1[0]
            r1Y_GP1 = Slvp_GP1_CgP1[1] - P_GP1_CgP1[1]
            r1Z_GP1 = Slvp_GP1_CgP1[2] - P_GP1_CgP1[2]

            # The r2_GP1 vector goes from P_GP1_CgP1 to the LineVortex's end point (in
            # the first Airplane's geometry axes).
            r2X_GP1 = Elvp_GP1_CgP1[0] - P_GP1_CgP1[0]
            r2Y_GP1 = Elvp_GP1_CgP1[1] - P_GP1_CgP1[1]
            r2Z_GP1 = Elvp_GP1_CgP1[2] - P_GP1_CgP1[2]

            # The r3_GP1 vector is the cross product of r1_GP1 and r2_GP1 (in the first
            # Airplane's geometry axes).
            r3X_GP1 = r1Y_GP1 * r2Z_GP1 - r1Z_GP1 * r2Y_GP1
            r3Y_GP1 = r1Z_GP1 * r2X_GP1 - r1X_GP1 * r2Z_GP1
            r3Z_GP1 = r1X_GP1 * r2Y_GP1 - r1Y_GP1 * r2X_GP1

            # Find the lengths of r1_GP1 and r2_GP1.
            r1 = math.sqrt(r1X_GP1**2.0 + r1Y_GP1**2.0 + r1Z_GP1**2.0)
            r2 = math.sqrt(r2X_GP1**2.0 + r2Y_GP1**2.0 + r2Z_GP1**2.0)

            # Check for singularities using scale invariant criteria. The vertex
            # proximity checks (r1/r0 and r2/r0 but refactored below to use
            # multiplication instead of slower division) guard 1/r singularities.
            if r1 < r0_times_tol:
                singularity_counts[1] += 1
                continue
            if r2 < r0_times_tol:
                singularity_counts[2] += 1
                continue

            # Cache squared length of r3_GP1 as it is used in the c_4 calculation.
            r3_sq = r3X_GP1**2.0 + r3Y_GP1**2.0 + r3Z_GP1**2.0

            # Find the length of r3_GP1.
            r3 = math.sqrt(r3_sq)

            # Cache r1 * r2 as it is used for the collinearity check and twice in the
            # c_4 calculation.
            r1_times_r2 = r1 * r2

            c_3 = r1X_GP1 * r2X_GP1 + r1Y_GP1 * r2Y_GP1 + r1Z_GP1 * r2Z_GP1

            # The collinearity check (r3/(r1*r2) = |sin(theta)| but with the same
            # multiplication instead of division refactor) guards catastrophic
            # cancellation in 1-cos(theta).
            if r3 < (_tol * r1_times_r2):
                # Collinearity can indicate one of two things. If the point is collinear
                # and between the filament's vertices, it is a true singularity (the
                # Biot-Savart equation diverges), so we exclude the contribution as it
                # is the influence of the filament on itself. If the point is collinear
                # and off to one side of the filament, it isn't a true singularity, as
                # the Biot-Savart equation (if calculated with infinite precision)
                # correctly returns zero induced velocity. However, we still run into
                # the catastrophic cancellation issue, so we again manually return zero
                # induced velocity contribution. These two situations are distinguished
                # by the sign of the c_3 (the dot product of r1 and r2).
                if c_3 < 0.0:
                    singularity_counts[3] += 1
                continue

            c_4 = c_1 * (r1 + r2) * (r1_times_r2 - c_3) / (r1_times_r2 * (r3_sq + c_2))
            stackVInd_GP1__E[point_id, 0] += c_4 * r3X_GP1
            stackVInd_GP1__E[point_id, 1] += c_4 * r3Y_GP1
            stackVInd_GP1__E[point_id, 2] += c_4 * r3Z_GP1
    return stackVInd_GP1__E


@njit(cache=True, fastmath=False)
def _expanded_velocities_from_line_vortices(
    stackP_GP1_CgP1: np.ndarray,
    stackSlvp_GP1_CgP1: np.ndarray,
    stackElvp_GP1_CgP1: np.ndarray,
    strengths: np.ndarray,
    r_c0s: np.ndarray,
    singularity_counts: np.ndarray,
    ages: np.ndarray | None = None,
    nu: float = 0.0,
) -> np.ndarray:
    """Takes in a group of points and the attributes of a group of LineVortices and
    finds the induced velocity at every point due to each LineVortex.

    This function uses a modified version of the Biot-Savart law to create a smooth
    induced velocity decay based on a LineVortex's core radius. The core radius grows
    from an initial value based on the LineVortex's age.

    This function's performance has been highly optimized for unsteady simulations via
    Numba. While using Numba dramatically increases unsteady simulation performance, it
    does cause a performance drop for the less intense steady simulations.

    **Citation:**

    Core radius equation adapted from Eq. 3 of: "A Reynolds Number-Based Blade Tip
    Vortex Model"

    Authors: Manikandan Ramasamy and J. Gordon Leishman

    Biot-Savart equation adapted from: "Extended Unsteady Vortex-Lattice Method for
    Insect Flapping Wings"

    Authors: Anh Tuan Nguyen, Joong-Kwan Kim, Jong-Seob Han, and Jae-Hung Han

    :param stackP_GP1_CgP1: A (N,3) ndarray of floats representing the positions of N
        points (in the first Airplane's geometry axes, relative to the first Airplane's
        CG). The units are in meters.
    :param stackSlvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of M
        LineVortices' starting vertices (in the first Airplane's geometry axes, relative
        to the first Airplane's CG). The units are in meters.
    :param stackElvp_GP1_CgP1: A (M,3) ndarray of floats representing the positions of
        the M LineVortices' ending vertices (in the first Airplane's geometry axes,
        relative to the first Airplane's CG). The units are in meters.
    :param strengths: A (M,) ndarray of floats representing the strengths of the M
        LineVortices. The units are in meters squared per second.
    :param r_c0s: A (M,) ndarray of floats representing the initial core radii of the M
        LineVortices. Based on results from Ramasamy and Leishman (2007), a reasonable
        value that works across scales is 3.0% the chord length of each LineVortices'
        parent Wing. The units are in meters.
    :param singularity_counts: A (4,) ndarray of int64 representing the cumulative
        counts of singularity events. Index mapping: [0] degenerate filament, [1] vertex
        start proximity, [2] vertex end proximity, [3] collinearity. Counts are
        incremented in place and accumulate across calls.
    :param ages: For bound LineVortices, this must be None. For LineVortices that have
        been shed into the wake, it must be a (M,) ndarray of floats representing the
        ages of the M LineVortices in seconds. The default is None.
    :param nu: A non negative float representing the kinematic viscosity of the fluid.
        The units are in meters squared per second. The default is 0.0.
    :return: A (N,M,3) ndarray of floats for the induced velocity at each of the N
        points (in the first Airplane's geometry axes, observed from the Earth frame)
        due to each of the M LineVortices. The units are in meters per second.
    """
    num_vortices = stackSlvp_GP1_CgP1.shape[0]
    num_points = stackP_GP1_CgP1.shape[0]

    # Initialize an empty array, which we will fill with the induced velocities (in the
    # first Airplane's geometry axes, observed from the Earth frame).
    gridVInd_GP1__E = np.zeros((num_points, num_vortices, 3))

    # If the user didn't specify any ages, set the age of each LineVortex to 0.0
    # seconds.
    if ages is None:
        ages = np.zeros(num_vortices)

    for vortex_id in range(num_vortices):
        Slvp_GP1_CgP1 = stackSlvp_GP1_CgP1[vortex_id]
        Elvp_GP1_CgP1 = stackElvp_GP1_CgP1[vortex_id]

        # The r0_GP1 vector goes from the LineVortex's start point to its end point (in
        # the first Airplane's geometry axes).
        r0X_GP1 = Elvp_GP1_CgP1[0] - Slvp_GP1_CgP1[0]
        r0Y_GP1 = Elvp_GP1_CgP1[1] - Slvp_GP1_CgP1[1]
        r0Z_GP1 = Elvp_GP1_CgP1[2] - Slvp_GP1_CgP1[2]

        # Find r0_GP1's length.
        r0 = math.sqrt(r0X_GP1**2.0 + r0Y_GP1**2.0 + r0Z_GP1**2.0)

        # Skip degenerate filaments where the start and end points coincide.
        if r0 < _eps:
            singularity_counts[0] += 1
            continue

        strength = strengths[vortex_id]
        age = ages[vortex_id]
        r_c0 = r_c0s[vortex_id]

        # Pre compute r0 * _tol outside the inner loop.
        r0_times_tol = r0 * _tol

        # Calculate the radius of the LineVortex's core squared. The initial core radius
        # ensures nonzero regularization even for bound vortices with zero age.
        r_c_sq = r_c0**2.0 + _four_lamb * (nu + _squire * abs(strength)) * age

        c_1 = strength / _four_pi
        c_2 = r0**2.0 * r_c_sq

        for point_id in range(num_points):
            P_GP1_CgP1 = stackP_GP1_CgP1[point_id]

            # The r1_GP1 vector goes from P_GP1_CgP1 to the LineVortex's start point (in
            # the first Airplane's geometry axes).
            r1X_GP1 = Slvp_GP1_CgP1[0] - P_GP1_CgP1[0]
            r1Y_GP1 = Slvp_GP1_CgP1[1] - P_GP1_CgP1[1]
            r1Z_GP1 = Slvp_GP1_CgP1[2] - P_GP1_CgP1[2]

            # The r2_GP1 vector goes from P_GP1_CgP1 to the LineVortex's end point (in
            # the first Airplane's geometry axes).
            r2X_GP1 = Elvp_GP1_CgP1[0] - P_GP1_CgP1[0]
            r2Y_GP1 = Elvp_GP1_CgP1[1] - P_GP1_CgP1[1]
            r2Z_GP1 = Elvp_GP1_CgP1[2] - P_GP1_CgP1[2]

            # The r3_GP1 vector is the cross product of r1_GP1 and r2_GP1 (in the first
            # Airplane's geometry axes).
            r3X_GP1 = r1Y_GP1 * r2Z_GP1 - r1Z_GP1 * r2Y_GP1
            r3Y_GP1 = r1Z_GP1 * r2X_GP1 - r1X_GP1 * r2Z_GP1
            r3Z_GP1 = r1X_GP1 * r2Y_GP1 - r1Y_GP1 * r2X_GP1

            # Find the lengths of r1_GP1 and r2_GP1.
            r1 = math.sqrt(r1X_GP1**2.0 + r1Y_GP1**2.0 + r1Z_GP1**2.0)
            r2 = math.sqrt(r2X_GP1**2.0 + r2Y_GP1**2.0 + r2Z_GP1**2.0)

            # Check for singularities using scale invariant criteria. The vertex
            # proximity checks (r1/r0 and r2/r0 but refactored below to use
            # multiplication instead of slower division) guard 1/r singularities.
            if r1 < r0_times_tol:
                singularity_counts[1] += 1
                continue
            if r2 < r0_times_tol:
                singularity_counts[2] += 1
                continue

            # Cache squared length of r3_GP1 as it is used in the c_4 calculation.
            r3_sq = r3X_GP1**2.0 + r3Y_GP1**2.0 + r3Z_GP1**2.0

            # Find the length of r3_GP1.
            r3 = math.sqrt(r3_sq)

            # Cache r1 * r2 as it is used for the collinearity check and twice in the
            # c_4 calculation.
            r1_times_r2 = r1 * r2

            c_3 = r1X_GP1 * r2X_GP1 + r1Y_GP1 * r2Y_GP1 + r1Z_GP1 * r2Z_GP1

            # The collinearity check (r3/(r1*r2) = |sin(theta)| but with the same
            # multiplication instead of division refactor) guards catastrophic
            # cancellation in 1-cos(theta).
            if r3 < (_tol * r1_times_r2):
                # Collinearity can indicate one of two things. If the point is collinear
                # and between the filament's vertices, it is a true singularity (the
                # Biot-Savart equation diverges), so we exclude the contribution as it
                # is the influence of the filament on itself. If the point is collinear
                # and off to one side of the filament, it isn't a true singularity, as
                # the Biot-Savart equation (if calculated with infinite precision)
                # correctly returns zero induced velocity. However, we still run into
                # the catastrophic cancellation issue, so we again manually return zero
                # induced velocity contribution. These two situations are distinguished
                # by the sign of the c_3 (the dot product of r1 and r2).
                if c_3 < 0.0:
                    singularity_counts[3] += 1
                continue

            c_4 = c_1 * (r1 + r2) * (r1_times_r2 - c_3) / (r1_times_r2 * (r3_sq + c_2))
            gridVInd_GP1__E[point_id, vortex_id, 0] = c_4 * r3X_GP1
            gridVInd_GP1__E[point_id, vortex_id, 1] = c_4 * r3Y_GP1
            gridVInd_GP1__E[point_id, vortex_id, 2] = c_4 * r3Z_GP1
    return gridVInd_GP1__E


# ==== PARALLELIZED IMPLEMENTATION FOR ISSUE #140 ====
# Parallel outer loop with thread-local accumulators - NO RACE CONDITIONS
# Each thread processes one complete vortex, then merges results sequentially
# See: https://github.com/camUrban/PteraSoftware/issues/140


@njit(fastmath=True, parallel=True)  # Phase 3.4+: fastmath safe with singularity checks
def _collapsed_velocities_from_line_vortices_parallel(
    stackP_GP1_CgP1: np.ndarray,
    stackSlvp_GP1_CgP1: np.ndarray,
    stackElvp_GP1_CgP1: np.ndarray,
    strengths: np.ndarray,
    r_c0s: np.ndarray,
    singularity_counts: np.ndarray,
    ages: np.ndarray | None = None,
    nu: float = 0.0,
) -> np.ndarray:
    """Parallel vortex loop with thread-local accumulators (Issue #140).

    Parallelizes outer loop (vortices) with each thread processing one complete vortex
    independently. Results stored in non-overlapping array slices to avoid race
    conditions. After prange barrier, results are merged serially.

    Thread safety: - Each thread: unique vortex_id from prange (no conflicts) - Each
    thread: local_velocities (thread-private) - Each thread: local_counts (thread-
    private) - Storage: vortex_results[vortex_id] (non-overlapping per thread) - After
    prange: merge happens at barrier (all threads synchronized)
    """
    num_vortices = stackSlvp_GP1_CgP1.shape[0]
    num_points = stackP_GP1_CgP1.shape[0]

    # Pre-allocate non-overlapping storage for each thread's results
    vortex_results = np.zeros((num_vortices, num_points, 3))
    vortex_counts = np.zeros((num_vortices, 4), dtype=np.int64)

    if ages is None:
        ages = np.zeros(num_vortices)

    # Parallel outer loop over vortices - each thread gets unique vortex_id
    for vortex_id in prange(num_vortices):
        # Thread-local accumulators (created fresh for each iteration)
        local_velocities = np.zeros((num_points, 3))
        local_counts = np.zeros(4, dtype=np.int64)

        Slvp_GP1_CgP1 = stackSlvp_GP1_CgP1[vortex_id]
        Elvp_GP1_CgP1 = stackElvp_GP1_CgP1[vortex_id]

        r0X_GP1 = Elvp_GP1_CgP1[0] - Slvp_GP1_CgP1[0]
        r0Y_GP1 = Elvp_GP1_CgP1[1] - Slvp_GP1_CgP1[1]
        r0Z_GP1 = Elvp_GP1_CgP1[2] - Slvp_GP1_CgP1[2]

        r0 = math.sqrt(r0X_GP1**2.0 + r0Y_GP1**2.0 + r0Z_GP1**2.0)

        if r0 < _eps:
            local_counts[0] = np.int64(1)
        else:
            strength = strengths[vortex_id]
            age = ages[vortex_id]
            r_c0 = r_c0s[vortex_id]

            r0_times_tol = r0 * _tol
            r_c_sq = r_c0**2.0 + _four_lamb * (nu + _squire * abs(strength)) * age

            c_1 = strength / _four_pi
            c_2 = r0**2.0 * r_c_sq

            # Serial inner loop - each thread processes its vortex completely
            for point_id in range(num_points):
                P_GP1_CgP1 = stackP_GP1_CgP1[point_id]

                r1X_GP1 = Slvp_GP1_CgP1[0] - P_GP1_CgP1[0]
                r1Y_GP1 = Slvp_GP1_CgP1[1] - P_GP1_CgP1[1]
                r1Z_GP1 = Slvp_GP1_CgP1[2] - P_GP1_CgP1[2]

                r2X_GP1 = Elvp_GP1_CgP1[0] - P_GP1_CgP1[0]
                r2Y_GP1 = Elvp_GP1_CgP1[1] - P_GP1_CgP1[1]
                r2Z_GP1 = Elvp_GP1_CgP1[2] - P_GP1_CgP1[2]

                r3X_GP1 = r1Y_GP1 * r2Z_GP1 - r1Z_GP1 * r2Y_GP1
                r3Y_GP1 = r1Z_GP1 * r2X_GP1 - r1X_GP1 * r2Z_GP1
                r3Z_GP1 = r1X_GP1 * r2Y_GP1 - r1Y_GP1 * r2X_GP1

                r1 = math.sqrt(r1X_GP1**2.0 + r1Y_GP1**2.0 + r1Z_GP1**2.0)
                r2 = math.sqrt(r2X_GP1**2.0 + r2Y_GP1**2.0 + r2Z_GP1**2.0)

                if r1 < r0_times_tol:
                    local_counts[1] += 1
                    continue
                if r2 < r0_times_tol:
                    local_counts[2] += 1
                    continue

                r3_sq = r3X_GP1**2.0 + r3Y_GP1**2.0 + r3Z_GP1**2.0

                r3 = math.sqrt(r3_sq)

                r1_times_r2 = r1 * r2

                c_3 = r1X_GP1 * r2X_GP1 + r1Y_GP1 * r2Y_GP1 + r1Z_GP1 * r2Z_GP1

                if r3 < (_tol * r1_times_r2):
                    if c_3 < 0.0:
                        local_counts[3] += 1
                    continue

                c_4 = (
                    c_1
                    * (r1 + r2)
                    * (r1_times_r2 - c_3)
                    / (r1_times_r2 * (r3_sq + c_2))
                )
                local_velocities[point_id, 0] += c_4 * r3X_GP1
                local_velocities[point_id, 1] += c_4 * r3Y_GP1
                local_velocities[point_id, 2] += c_4 * r3Z_GP1

        # Store in non-overlapping slice for this thread (no race condition).
        # Unroll inner loop to reduce loop overhead.
        vortex_results[vortex_id, :, :] = local_velocities

        # Unroll count storage (4 iterations -> 4 direct assignments)
        vortex_counts[vortex_id, 0] = local_counts[0]
        vortex_counts[vortex_id, 1] = local_counts[1]
        vortex_counts[vortex_id, 2] = local_counts[2]
        vortex_counts[vortex_id, 3] = local_counts[3]

    # AFTER prange barrier: parallel merge of vortex results.
    # Parallelize over points (no race conditions - each point processes uniquely).
    # Changed loop order from (vortex, point) to (point, vortex) to enable prange on
    # point_id. Estimated improvement: significant speedup on multi-core (eliminates
    # O(N²) serial bottleneck).
    result = np.zeros((num_points, 3))
    for point_id in prange(num_points):
        for vortex_id in range(num_vortices):
            result[point_id, 0] += vortex_results[vortex_id, point_id, 0]
            result[point_id, 1] += vortex_results[vortex_id, point_id, 1]
            result[point_id, 2] += vortex_results[vortex_id, point_id, 2]

    # Serial merge of singularity counts (4 types, fast)
    for i in range(4):
        for vortex_id in range(num_vortices):
            singularity_counts[i] += vortex_counts[vortex_id, i]

    return result


# ==============================================================================
# BACKEND DISPATCH LAYER - Issue #140, Phase 3.4+
# ==============================================================================
# This section enables runtime selection between serial CPU, parallel CPU, and GPU
# backends via environment variables or runtime API.
#
# Environment Variables (set before import):
# - PTERASOFTWARE_BACKEND: 'serial_cpu' (default), 'parallel_cpu', or 'gpu_cuda'
# - PTERASOFTWARE_FORCE_GPU: '1' to error if GPU unavailable
#
# API Functions:
# - get_aerodynamics_backend(): Check currently configured backend
# - set_aerodynamics_backend(backend): Switch backend at runtime
# - get_backend_info(): Get configuration details for debugging


def get_aerodynamics_backend() -> str:
    """Get currently configured aerodynamics backend.

    Returns:     str: Backend name ('serial_cpu', 'parallel_cpu', or 'gpu_cuda')
    """
    config = get_config()
    return config.get_backend().value


def set_aerodynamics_backend(backend: str) -> None:
    """Switch aerodynamics backend at runtime.

    Changes take effect for new computations. Already-compiled Numba functions are not
    recompiled.

    Parameters:     backend: Target backend ('serial_cpu', 'parallel_cpu', or
    'gpu_cuda')

    Raises:     ValueError: If backend invalid or GPU not available for GPU backend
    """
    try:
        config = get_config()
        config.set_backend(backend)
    except Exception as e:
        raise ValueError(
            f"Failed to switch backend to '{backend}': {e}\n"
            "Valid options: 'serial_cpu', 'parallel_cpu', 'gpu_cuda'"
        )


def get_backend_info() -> dict:
    """Get detailed information about backend configuration.

    Useful for debugging and understanding which optimizations are enabled.

    Returns:     dict: Configuration state including backend, GPU availability, and
    flags
    """
    config = get_config()
    return config.get_config_summary()


# ==============================================================================
# Public API Documentation
# ==============================================================================
__all__ = [
    # Main API functions (public)
    "collapsed_velocities_from_ring_vortices",
    "collapsed_velocities_from_ring_vortices_chordwise_segments",
    "expanded_velocities_from_ring_vortices",
    "collapsed_velocities_from_horseshoe_vortices",
    "expanded_velocities_from_horseshoe_vortices",
    # Backend control (new public API)
    "get_aerodynamics_backend",
    "set_aerodynamics_backend",
    "get_backend_info",
]
