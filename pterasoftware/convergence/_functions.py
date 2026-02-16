"""Contains shared utilities for the convergence analysis subpackage.

**Contains the following classes:**

None

**Contains the following functions:**

_check_coefficient_convergence: Checks per coefficient convergence using an absolute
plus relative tolerance.

_validate_non_trapezoidal_problem: Validate that the UnsteadyProblem is suitable for
non-trapezoidal analysis.

_verify_panel_aspect_ratio: Verify that the airplane's mesh achieves the target panel
aspect ratio.

_visualize_wing_mesh: Visualize an Airplane's wing mesh for verification.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv

from .. import _logging, geometry, problems

convergence_logger = _logging.get_logger("convergence")

_COEFFICIENT_LABELS = ("cFX", "cFY", "cFZ", "cMX", "cMY", "cMZ")
_LOAD_LABELS = ("FX", "FY", "FZ", "MX", "MY", "MZ")
_LOAD_UNITS = ("N", "N", "N", "N*m", "N*m", "N*m")


def _check_coefficient_convergence(
    current_coefficients: np.ndarray,
    coarser_coefficients: np.ndarray,
    rtol: float,
    atol: float,
    mask: np.ndarray | None = None,
) -> tuple[bool, float, np.ndarray, np.ndarray, np.ndarray]:
    """Checks per coefficient convergence using an absolute plus relative tolerance.

    For each of the 6 coefficients, the error is defined as abs(current - coarser) and
    the tolerance is defined as atol + rtol * max(abs(current), abs(coarser)). A
    coefficient is converged when its error is less than or equal to its tolerance. The
    metric for each coefficient is a percentage indicating how close the coefficient is
    to converging, capped at 100.0.

    :param current_coefficients: A (6,) ndarray of floats representing the current
        (finer resolution) coefficients.
    :param coarser_coefficients: A (6,) ndarray of floats representing the previous
        (coarser resolution) coefficients.
    :param rtol: A float representing the relative tolerance. Must be positive.
    :param atol: A float representing the absolute tolerance. Must be positive.
    :param mask: A (6,) ndarray of bools that determines which coefficients are checked
        for convergence. If None, all 6 coefficients are checked. The default is None.
    :return: A tuple of (all_converged, min_metric, errors, tolerances, metrics) where
        all_converged is a bool indicating whether the masked coefficients are
        converged, min_metric is a float representing the minimum metric across the
        masked coefficients, errors is a (6,) ndarray of floats representing the
        absolute errors, tolerances is a (6,) ndarray of floats representing the
        computed tolerances, and metrics is a (6,) ndarray of floats representing the
        convergence metrics (percentages).
    """
    if mask is None:
        mask = np.ones(6, dtype=bool)

    errors = np.abs(current_coefficients - coarser_coefficients)
    tolerances = atol + rtol * np.maximum(
        np.abs(current_coefficients), np.abs(coarser_coefficients)
    )
    converged = errors <= tolerances

    metrics = np.zeros(6, dtype=float)
    for i in range(6):
        if errors[i] == 0.0:
            metrics[i] = 100.0
        else:
            metrics[i] = 100.0 * min(1.0, tolerances[i] / errors[i])

    all_converged = bool(np.all(converged[mask]))
    min_metric = float(np.min(metrics[mask]))

    return all_converged, min_metric, errors, tolerances, metrics


def _validate_non_trapezoidal_problem(
    ref_problem: problems.UnsteadyProblem,
) -> None:
    """Validate that the UnsteadyProblem is suitable for non-trapezoidal analysis.

    Non-trapezoidal convergence analysis requires: 1. Exactly one Airplane in the
    problem. 2. All Wings have WingCrossSections with num_spanwise_panels=1 (except the
    last    WingCrossSection of each Wing, which must have num_spanwise_panels=None).

    :param ref_problem: The UnsteadyProblem to validate.
    :raises ValueError: If the problem doesn't meet the requirements.
    """
    # Get the list of airplane movements from the movement
    ref_movement = ref_problem.movement
    ref_airplane_movements = ref_movement.airplane_movements

    # Check 1: Exactly one airplane
    if len(ref_airplane_movements) != 1:
        raise ValueError(
            f"analyze_unsteady_convergence_non_trapezoidal only supports "
            f"UnsteadyProblems with exactly one Airplane. "
            f"Found {len(ref_airplane_movements)} Airplane(s)."
        )

    # Check 2: All wings use num_spanwise_panels=1 (except last WCS)
    ref_airplane_movement = ref_airplane_movements[0]
    ref_wing_movements = ref_airplane_movement.wing_movements

    for wing_id, ref_wing_movement in enumerate(ref_wing_movements):
        ref_wcs_movements = ref_wing_movement.wing_cross_section_movements
        num_wcs = len(ref_wcs_movements)

        for wcs_id, ref_wcs_movement in enumerate(ref_wcs_movements):
            ref_wcs = ref_wcs_movement.base_wing_cross_section
            expected_panels = 1 if wcs_id < (num_wcs - 1) else None

            if ref_wcs.num_spanwise_panels != expected_panels:
                raise ValueError(
                    f"Wing {wing_id}, WingCrossSection {wcs_id} has "
                    f"num_spanwise_panels={ref_wcs.num_spanwise_panels}, "
                    f"expected {expected_panels}. All WingCrossSections must have "
                    f"num_spanwise_panels=1 (except the last, which must be None) "
                    f"for non-trapezoidal convergence analysis."
                )


def _verify_panel_aspect_ratio(
    airplane: geometry.airplane.Airplane,
    target_panel_ar: int,
    tolerance: float = 0.5,
) -> tuple[bool, float]:
    """Verify that the airplane's mesh achieves the target panel aspect ratio.

    Calculates the average panel aspect ratio across all wings and checks if it's within
    the specified tolerance of the target.

    :param airplane: The Airplane to check.
    :param target_panel_ar: The target panel aspect ratio.
    :param tolerance: Acceptable absolute deviation from target. Default is 0.5.
    :return: Tuple of (passed, actual_average_ar). passed is True if the actual average
        AR is within tolerance of the target.
    """
    total_ar = 0.0
    num_wings = 0

    for wing in airplane.wings:
        if wing.average_panel_aspect_ratio is not None:
            total_ar += wing.average_panel_aspect_ratio
            num_wings += 1

    actual_avg_ar = total_ar / num_wings if num_wings > 0 else 0.0
    passed = abs(actual_avg_ar - target_panel_ar) <= tolerance

    return passed, actual_avg_ar


def _visualize_wing_mesh(
    airplane: geometry.airplane.Airplane,
    title: str = "Wing Mesh Visualization",
    show: bool = True,
    save_path: str | None = None,
) -> None:
    """Visualize an Airplane's wing mesh for verification.

    Creates a 3D visualization of the wing mesh showing panel boundaries. Reports mesh
    statistics including number of panels and average panel aspect ratio.

    :param airplane: The Airplane whose mesh to visualize.
    :param title: Title for the visualization window. Default is "Wing Mesh
        Visualization".
    :param show: If True, display the visualization interactively. Default is True.
    :param save_path: If provided, save the visualization to this path as a PNG. Default
        is None.
    """
    # Create the plotter
    plotter = pv.Plotter(off_screen=not show, window_size=[1024, 768])
    plotter.set_background("black")  # type: ignore[arg-type]

    # Build panel surfaces
    panel_vertices = np.empty((0, 3), dtype=float)
    panel_faces = np.empty(0, dtype=int)
    panel_num = 0
    total_panels = 0

    for wing in airplane.wings:
        _panels = wing.panels
        if _panels is None:
            continue

        panels = np.ravel(_panels)
        total_panels += len(panels)

        for panel in panels:
            # Arrange this Panel's vertices and faces into ndarrays
            # Use _G_Cg coordinates (set during meshing) rather than _GP1_CgP1
            # coordinates (set during problem creation) so visualization works
            # before a problem is created.
            panel_vertices_to_add = np.vstack(
                (
                    panel.Flpp_G_Cg,
                    panel.Frpp_G_Cg,
                    panel.Brpp_G_Cg,
                    panel.Blpp_G_Cg,
                )
            )
            panel_face_to_add = np.array(
                [
                    4,
                    (panel_num * 4),
                    (panel_num * 4) + 1,
                    (panel_num * 4) + 2,
                    (panel_num * 4) + 3,
                ],
                dtype=int,
            )

            panel_vertices = np.vstack((panel_vertices, panel_vertices_to_add))
            panel_faces = np.hstack((panel_faces, panel_face_to_add))
            panel_num += 1

    # Convert to PyVista axes (swap Y and Z, negate new Z)
    panel_vertices_pv = panel_vertices.copy()
    panel_vertices_pv[:, 1] = panel_vertices[:, 2]
    panel_vertices_pv[:, 2] = -panel_vertices[:, 1]

    # Create PolyData and add to plotter
    if len(panel_vertices_pv) > 0:
        mesh = pv.PolyData(panel_vertices_pv, panel_faces)
        plotter.add_mesh(
            mesh,
            show_edges=True,
            edge_color="white",
            color="chartreuse",
            smooth_shading=False,
        )

    # Add title with statistics
    _, avg_ar = _verify_panel_aspect_ratio(airplane, 1)  # Get actual AR
    stats_text = f"{title}\nPanels: {total_panels}, Avg AR: {avg_ar:.2f}"
    plotter.add_text(stats_text, position="upper_left", font_size=10, color="white")

    # Set camera view (top-down)
    plotter.view_xz()  # type: ignore[call-arg]
    plotter.camera.roll -= 90
    plotter.camera.zoom(1.2)

    # Log statistics
    convergence_logger.info(
        f"Mesh visualization: {total_panels} panels, AR={avg_ar:.2f}"
    )

    # Show and/or save
    if show:
        plotter.show(screenshot=save_path)
    else:
        plotter.screenshot(save_path)
        plotter.close()

    if save_path is not None:
        convergence_logger.info(f"Saved mesh visualization to {save_path}")
