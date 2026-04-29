"""A minimal flapping-wing free-flight example for ``origin/feature/free_flight``.

This example uses the coupled UVLM + MuJoCo path from the feature branch to run a
free-flight simulation of a simple flapping aircraft in air. Unlike the ground-effect
example, it does not define an image surface or a MuJoCo ground plane.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

# Import the local checkout's package, not any site-packages installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pterasoftware as ps
from pterasoftware import _transformations

if not hasattr(ps.operating_point, "CoupledOperatingPoint"):
    raise RuntimeError(
        "This example requires the coupled free-flight API from "
        "`origin/feature/free_flight`."
    )


AIR_DENSITY = 1.225
KINEMATIC_VISCOSITY = 15.06e-6
GRAVITY_E = (0.0, 0.0, 9.80665)

FLAPPING_FREQUENCY_HZ = 8.0
FLAPPING_PERIOD_S = 1.0 / FLAPPING_FREQUENCY_HZ
DEFAULT_STEPS_PER_FLAP = 24

INITIAL_SPEED_MPS = 2.5
INITIAL_ALPHA_DEG = 5.0
INITIAL_YAW_DEG = 90.0

# This is the total aircraft weight used by the rigid-body model, not just the wings'
# weight. The aerodynamic geometry is wing-only, but the dynamics already treat this as
# the whole flapping aircraft's weight.
TOTAL_AIRCRAFT_WEIGHT_N = 0.18

# A simple body-drag model used to represent rigid-body drag that is missing from the
# wing-only UVLM geometry. The linear term acts as a small regularizing velocity
# damper, while the quadratic term behaves like a crude fuselage/body parasitic drag.
BODY_DRAG_LINEAR_N_PER_MPS = 0.003
BODY_DRAG_CD_AREA_M2 = 0.002

FIX_PITCH_ROTATION = True
FIXED_PITCH_DEG = INITIAL_ALPHA_DEG

# A simple diagonal inertia guess for a light flapping aircraft. These numbers are not
# tuned; they just provide a reasonable starting point for this branch example.
INERTIA_BP1_CGP1 = np.array(
    [
        [2.5e-5, 0.0, 0.0],
        [0.0, 4.5e-5, 0.0],
        [0.0, 0.0, 3.0e-5],
    ],
    dtype=float,
)

WING_ROOT_Y_M = 0.015
SEMI_SPAN_M = 0.11
ROOT_CHORD_M = 0.04
TIP_CHORD_M = 0.03

STROKE_AMPLITUDE_DEG = 30.0
FEATHER_AMPLITUDE_DEG = 12.0
FEATHER_PHASE_DEG = 110.0
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "output" / "free_flight_long_run"
)


def build_wing(*, name: str, mirror_only: bool) -> ps.geometry.wing.Wing:
    """Build one rigid rectangular wing."""
    symmetry_normal = (0.0, 1.0, 0.0) if mirror_only else None
    symmetry_point = (0.0, 0.0, 0.0) if mirror_only else None

    return ps.geometry.wing.Wing(
        wing_cross_sections=[
            ps.geometry.wing_cross_section.WingCrossSection(
                airfoil=ps.geometry.airfoil.Airfoil(name="naca0012"),
                chord=ROOT_CHORD_M,
                num_spanwise_panels=4,
                spanwise_spacing="uniform",
            ),
            ps.geometry.wing_cross_section.WingCrossSection(
                airfoil=ps.geometry.airfoil.Airfoil(name="naca0012"),
                chord=TIP_CHORD_M,
                Lp_Wcsp_Lpp=(0.0, SEMI_SPAN_M, 0.0),
                num_spanwise_panels=None,
            ),
        ],
        name=name,
        Ler_Gs_Cgs=(0.0, WING_ROOT_Y_M, 0.0),
        angles_Gs_to_Wn_ixyz=(0.0, 0.0, 0.0),
        symmetric=False,
        mirror_only=mirror_only,
        symmetryNormal_G=symmetry_normal,
        symmetryPoint_G_Cg=symmetry_point,
        num_chordwise_panels=4,
        chordwise_spacing="uniform",
    )


def build_airplane() -> ps.geometry.airplane.Airplane:
    """Build the simple two-wing aircraft."""
    return ps.geometry.airplane.Airplane(
        wings=[
            build_wing(name="Left Wing", mirror_only=False),
            build_wing(name="Right Wing", mirror_only=True),
        ],
        name="Flapping Free-Flight Example",
        weight=TOTAL_AIRCRAFT_WEIGHT_N,
    )


def rigid_body_drag_model(
    coupled_operating_point: ps.operating_point.CoupledOperatingPoint,
    airplane: ps.geometry.airplane.Airplane,
) -> tuple[np.ndarray, np.ndarray]:
    """Return simple rigid-body drag loads in wind axes.

    The coupled solver already applies the aircraft's full weight, so this helper only
    adds missing rigid-body effects that the wing-only aerodynamic geometry does not
    model.
    """
    del airplane

    speed_mps = coupled_operating_point.vCg__E
    rho = coupled_operating_point.rho

    linear_drag_n = BODY_DRAG_LINEAR_N_PER_MPS * speed_mps
    quadratic_drag_n = 0.5 * rho * BODY_DRAG_CD_AREA_M2 * speed_mps**2
    total_drag_n = linear_drag_n + quadratic_drag_n

    return (
        np.array([-total_drag_n, 0.0, 0.0], dtype=float),
        np.zeros(3, dtype=float),
    )


def extract_izyx_angles_deg(R_pas_E_to_BP1: np.ndarray) -> np.ndarray:
    """Extract intrinsic z-y'-x'' Euler angles from a passive rotation matrix."""
    sin_angleY = np.clip(-R_pas_E_to_BP1[0, 2], -1.0, 1.0)
    angleY = np.rad2deg(np.arcsin(sin_angleY))

    if np.abs(sin_angleY) > 0.99999:
        angleX = 0.0
        angleZ = np.rad2deg(np.arctan2(-R_pas_E_to_BP1[1, 0], R_pas_E_to_BP1[1, 1]))
    else:
        angleX = np.rad2deg(np.arctan2(R_pas_E_to_BP1[1, 2], R_pas_E_to_BP1[2, 2]))
        angleZ = np.rad2deg(np.arctan2(R_pas_E_to_BP1[0, 1], R_pas_E_to_BP1[0, 0]))

    return np.array([angleX, angleY, angleZ], dtype=float)


def clamp_mujoco_pitch(mujoco_model: object, target_pitch_deg: float) -> None:
    """Clamp the MuJoCo rigid-body state back to a fixed pitch angle."""
    R_pas_BP1_to_E = mujoco_model.data.xmat[mujoco_model.body_id].reshape(3, 3)
    current_angles_deg = extract_izyx_angles_deg(R_pas_BP1_to_E.T)
    clamped_angles_deg = current_angles_deg.copy()
    clamped_angles_deg[1] = target_pitch_deg

    clamped_T_pas_E_to_BP1 = _transformations.generate_rot_T(
        angles=clamped_angles_deg,
        passive=True,
        intrinsic=True,
        order="zyx",
    )
    clamped_R_pas_BP1_to_E = clamped_T_pas_E_to_BP1[:3, :3].T
    clamped_quat_act_E_to_BP1_wxyz = _transformations.R_to_quat_wxyz(
        clamped_R_pas_BP1_to_E
    )

    mujoco_model.data.qpos[3:7] = clamped_quat_act_E_to_BP1_wxyz
    mujoco_model.data.qvel[4] = 0.0
    mujoco.mj_forward(mujoco_model.model, mujoco_model.data)


def install_fixed_pitch_rotation(
    coupled_problem: ps.problems.CoupledUnsteadyProblem,
    target_pitch_deg: float,
) -> None:
    """Patch the example's MuJoCo step so pitch is clamped after each update."""
    mujoco_model = coupled_problem.mujoco_model
    clamp_mujoco_pitch(mujoco_model=mujoco_model, target_pitch_deg=target_pitch_deg)
    original_step = mujoco_model.step

    def step_with_fixed_pitch() -> None:
        original_step()
        clamp_mujoco_pitch(mujoco_model=mujoco_model, target_pitch_deg=target_pitch_deg)

    mujoco_model.step = step_with_fixed_pitch


def build_wing_cross_section_movements(
    wing: ps.geometry.wing.Wing,
) -> list[ps.movements.wing_cross_section_movement.WingCrossSectionMovement]:
    """Create static cross-section movements for a rigid wing."""
    return [
        ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=wing.wing_cross_sections[0]
        ),
        ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=wing.wing_cross_sections[1]
        ),
    ]


def build_wing_movement(
    wing: ps.geometry.wing.Wing,
) -> ps.movements.wing_movement.WingMovement:
    """Create one flapping wing movement."""
    flapping_period = 1.0 / FLAPPING_FREQUENCY_HZ

    return ps.movements.wing_movement.WingMovement(
        base_wing=wing,
        wing_cross_section_movements=build_wing_cross_section_movements(wing),
        rotationPointOffset_Gs_Ler=(0.25 * ROOT_CHORD_M, 0.0, 0.0),
        ampAngles_Gs_to_Wn_ixyz=(
            STROKE_AMPLITUDE_DEG,
            FEATHER_AMPLITUDE_DEG,
            0.0,
        ),
        periodAngles_Gs_to_Wn_ixyz=(flapping_period, flapping_period, 0.0),
        spacingAngles_Gs_to_Wn_ixyz=("sine", "sine", "sine"),
        phaseAngles_Gs_to_Wn_ixyz=(0.0, FEATHER_PHASE_DEG, 0.0),
    )


def build_airplane_movement(
    airplane: ps.geometry.airplane.Airplane,
) -> ps.movements.airplane_movement.AirplaneMovement:
    """Create the aircraft's prescribed internal flapping motion."""
    return ps.movements.airplane_movement.AirplaneMovement(
        base_airplane=airplane,
        wing_movements=[
            build_wing_movement(airplane.wings[0]),
            build_wing_movement(airplane.wings[1]),
        ],
    )


def build_problem(
    prescribed_num_steps: int,
    free_num_steps: int,
    steps_per_flap: int,
    fix_pitch_rotation: bool = FIX_PITCH_ROTATION,
    fixed_pitch_deg: float = FIXED_PITCH_DEG,
) -> tuple[
    ps.problems.CoupledUnsteadyProblem,
    ps.coupled_unsteady_ring_vortex_lattice_method.CoupledUnsteadyRingVortexLatticeMethodSolver,
]:
    """Build the coupled problem and solver."""
    delta_time = FLAPPING_PERIOD_S / steps_per_flap
    airplane = build_airplane()
    airplane_movement = build_airplane_movement(airplane)

    initial_coupled_operating_point = ps.operating_point.CoupledOperatingPoint(
        rho=AIR_DENSITY,
        vCg__E=INITIAL_SPEED_MPS,
        alpha=INITIAL_ALPHA_DEG,
        beta=0.0,
        angles_E_to_BP1_izyx=(0.0, INITIAL_ALPHA_DEG, INITIAL_YAW_DEG),
        externalFX_W=0.0,
        nu=KINEMATIC_VISCOSITY,
        g_E=GRAVITY_E,
    )

    coupled_movement = ps.movements.movement.CoupledMovement(
        airplane_movement=airplane_movement,
        initial_coupled_operating_point=initial_coupled_operating_point,
        delta_time=delta_time,
        prescribed_num_steps=prescribed_num_steps,
        free_num_steps=free_num_steps,
    )

    coupled_problem = ps.problems.CoupledUnsteadyProblem(
        coupled_movement=coupled_movement,
        I_BP1_CgP1=INERTIA_BP1_CGP1,
        external_forces_fn=rigid_body_drag_model,
    )

    coupled_solver = ps.coupled_unsteady_ring_vortex_lattice_method.CoupledUnsteadyRingVortexLatticeMethodSolver(
        coupled_unsteady_problem=coupled_problem
    )

    if fix_pitch_rotation:
        install_fixed_pitch_rotation(
            coupled_problem=coupled_problem,
            target_pitch_deg=fixed_pitch_deg,
        )

    return coupled_problem, coupled_solver


def get_history_arrays(
    coupled_problem: ps.problems.CoupledUnsteadyProblem,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract time, normalized time, velocity, speed, alpha, and Euler-angle histories."""
    operating_points = coupled_problem.coupled_movement.coupled_operating_points
    num_steps = len(operating_points)
    times_s = np.arange(num_steps) * coupled_problem.delta_time
    times_over_period = times_s / FLAPPING_PERIOD_S
    velocities_E__E = np.array(
        [operating_point.vCg_E__E for operating_point in operating_points],
        dtype=float,
    )
    speeds_mps = np.array(
        [operating_point.vCg__E for operating_point in operating_points],
        dtype=float,
    )
    alphas_deg = np.array(
        [operating_point.alpha for operating_point in operating_points],
        dtype=float,
    )
    euler_angles_deg = np.array(
        [operating_point.angles_E_to_BP1_izyx for operating_point in operating_points],
        dtype=float,
    )
    return (
        times_s,
        times_over_period,
        velocities_E__E,
        speeds_mps,
        alphas_deg,
        euler_angles_deg,
    )


def plot_results(
    coupled_problem: ps.problems.CoupledUnsteadyProblem,
    coupled_solver: ps.coupled_unsteady_ring_vortex_lattice_method.CoupledUnsteadyRingVortexLatticeMethodSolver,
) -> None:
    """Plot a few basic free-flight time histories."""
    _, times_over_period, _, speeds_mps, alphas_deg, euler_angles_deg = (
        get_history_arrays(coupled_problem)
    )

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(times_over_period, coupled_solver.stackPosition_E_E[:, 0], label="x")
    axes[0].plot(times_over_period, coupled_solver.stackPosition_E_E[:, 2], label="z")
    axes[0].set_ylabel("Position (m)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(times_over_period, speeds_mps)
    axes[1].set_ylabel("Speed (m/s)")
    axes[1].grid(True)

    axes[2].plot(times_over_period, alphas_deg)
    axes[2].set_ylabel("Alpha (deg)")
    axes[2].grid(True)

    axes[3].plot(times_over_period, euler_angles_deg[:, 0], label="Roll")
    axes[3].plot(times_over_period, euler_angles_deg[:, 1], label="Pitch")
    axes[3].plot(times_over_period, euler_angles_deg[:, 2], label="Yaw")
    axes[3].set_ylabel("Euler (deg)")
    axes[3].set_xlabel("Time / Flapping Period")
    axes[3].grid(True)
    axes[3].legend()

    fig.suptitle("Flapping-Wing Free-Flight Example")
    fig.tight_layout()
    plt.show()


def save_velocity_history_plot(
    coupled_problem: ps.problems.CoupledUnsteadyProblem,
    save_path: Path,
) -> Path:
    """Save the velocity and Euler-angle histories as a PNG."""
    (
        _,
        times_over_period,
        velocities_E__E,
        speeds_mps,
        _,
        euler_angles_deg,
    ) = get_history_arrays(coupled_problem)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(times_over_period, velocities_E__E[:, 0], label="Vx")
    axes[0].plot(times_over_period, velocities_E__E[:, 1], label="Vy")
    axes[0].plot(times_over_period, velocities_E__E[:, 2], label="Vz")
    axes[0].set_ylabel("Velocity (m/s)")
    axes[0].set_title("CG Velocity Components in Earth Axes")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(times_over_period, speeds_mps, color="black")
    axes[1].set_ylabel("Speed (m/s)")
    axes[1].set_title("CG Speed Magnitude")
    axes[1].grid(True)

    axes[2].plot(times_over_period, euler_angles_deg[:, 0], label="Roll")
    axes[2].plot(times_over_period, euler_angles_deg[:, 1], label="Pitch")
    axes[2].plot(times_over_period, euler_angles_deg[:, 2], label="Yaw")
    axes[2].set_ylabel("Euler (deg)")
    axes[2].set_xlabel("Time / Flapping Period")
    axes[2].set_title("Euler Angles")
    axes[2].grid(True)
    axes[2].legend()

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def run_example(
    prescribed_num_steps: int,
    free_num_steps: int,
    steps_per_flap: int,
    fix_pitch_rotation: bool,
    fixed_pitch_deg: float,
    show_progress: bool,
    plot: bool,
    animate: bool,
    show_wake_vortices: bool,
    velocity_plot_path: Path | None = None,
) -> tuple[
    ps.problems.CoupledUnsteadyProblem,
    ps.coupled_unsteady_ring_vortex_lattice_method.CoupledUnsteadyRingVortexLatticeMethodSolver,
]:
    """Run the free-flight example."""
    coupled_problem, coupled_solver = build_problem(
        prescribed_num_steps=prescribed_num_steps,
        free_num_steps=free_num_steps,
        steps_per_flap=steps_per_flap,
        fix_pitch_rotation=fix_pitch_rotation,
        fixed_pitch_deg=fixed_pitch_deg,
    )

    coupled_solver.run(
        prescribed_wake=True,
        show_progress=show_progress,
    )

    final_state = coupled_problem.mujoco_model.get_state()
    final_position_E_E = final_state["position_E_E"]
    final_velocity_E__E = final_state["velocity_E__E"]

    print(
        "Final CG position in Earth axes (m): "
        f"{np.asarray(final_position_E_E, dtype=float)}"
    )
    print(
        "Final CG velocity in Earth axes (m/s): "
        f"{np.asarray(final_velocity_E__E, dtype=float)}"
    )
    print(f"Flapping period (s): {FLAPPING_PERIOD_S}")
    print(f"Total aircraft weight (N): {TOTAL_AIRCRAFT_WEIGHT_N}")
    print(f"Body drag linear coefficient (N per m/s): {BODY_DRAG_LINEAR_N_PER_MPS}")
    print(f"Body drag CdA (m^2): {BODY_DRAG_CD_AREA_M2}")
    print(f"Initial yaw (deg): {INITIAL_YAW_DEG}")
    print(f"Pitch rotation fixed: {fix_pitch_rotation}")
    print(f"Fixed pitch target (deg): {fixed_pitch_deg}")

    if velocity_plot_path is not None:
        saved_path = save_velocity_history_plot(
            coupled_problem=coupled_problem,
            save_path=velocity_plot_path,
        )
        print(f"Saved velocity plot to: {saved_path}")

    if plot:
        plot_results(coupled_problem=coupled_problem, coupled_solver=coupled_solver)

    if animate:
        ps.output.animate_free_flight(
            coupled_solver=coupled_solver,
            scalar_type="lift",
            show_wake_vortices=show_wake_vortices,
        )

    return coupled_problem, coupled_solver


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a minimal flapping-wing free-flight example on the "
            "feature/free_flight branch."
        )
    )
    parser.add_argument(
        "--prescribed-steps",
        type=int,
        default=DEFAULT_STEPS_PER_FLAP,
        help="Number of prescribed steps used to build the wake before free flight.",
    )
    parser.add_argument(
        "--free-steps",
        type=int,
        default=DEFAULT_STEPS_PER_FLAP,
        help="Number of coupled free-flight steps to run after the prescribed phase.",
    )
    parser.add_argument(
        "--steps-per-flap",
        type=int,
        default=DEFAULT_STEPS_PER_FLAP,
        help="Temporal resolution in solver steps per flapping period.",
    )
    parser.add_argument(
        "--fix-pitch-rotation",
        action=argparse.BooleanOptionalAction,
        default=FIX_PITCH_ROTATION,
        help="Clamp rigid-body pitch angle after each MuJoCo step.",
    )
    parser.add_argument(
        "--fixed-pitch-deg",
        type=float,
        default=FIXED_PITCH_DEG,
        help="Pitch angle to hold when pitch rotation is fixed.",
    )
    parser.add_argument(
        "--show-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the TQDM progress bar.",
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot a few basic time histories after the solve.",
    )
    parser.add_argument(
        "--animate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Open the free-flight animation after the solve.",
    )
    parser.add_argument(
        "--show-wake-vortices",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show shed wake vortex rings in the animation.",
    )
    parser.add_argument(
        "--velocity-plot-path",
        type=Path,
        default=None,
        help="Optional PNG path for saving a velocity-over-time plot.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the script."""
    args = parse_args()
    run_example(
        prescribed_num_steps=args.prescribed_steps,
        free_num_steps=args.free_steps,
        steps_per_flap=args.steps_per_flap,
        fix_pitch_rotation=args.fix_pitch_rotation,
        fixed_pitch_deg=args.fixed_pitch_deg,
        show_progress=args.show_progress,
        plot=args.plot,
        animate=args.animate,
        show_wake_vortices=args.show_wake_vortices,
        velocity_plot_path=args.velocity_plot_path,
    )


if __name__ == "__main__":
    main()
