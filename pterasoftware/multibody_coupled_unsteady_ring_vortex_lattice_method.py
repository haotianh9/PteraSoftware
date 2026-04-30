"""Contains the multibody coupled unsteady ring-vortex solver.

This module implements the first working multibody free-flight slice: multiple rigid
bodies, one MuJoCo freejoint per body, and a shared Earth-frame aerodynamic solve.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
from tqdm import tqdm

from . import (
    _aerodynamics_functions,
    _functions,
    _logging,
    _panel,
    _parameter_validation,
    _transformations,
    _vortices,
    geometry,
    operating_point,
    problems,
)

_logger = _logging.get_logger("multibody_coupled_unsteady_ring_vortex_lattice_method")


class MultiBodyCoupledUnsteadyRingVortexLatticeMethodSolver:
    """Solve a multibody coupled free-flight problem with ring-vortex aerodynamics.

    The current implementation intentionally targets the first multibody free-flight
    milestone:

    - at least two rigid bodies - shared atmosphere across bodies - prescribed or free
    wake - no image-surface support yet
    """

    def __init__(
        self,
        coupled_unsteady_problem: problems.MultiBodyCoupledUnsteadyProblem,
    ) -> None:
        if not isinstance(
            coupled_unsteady_problem, problems.MultiBodyCoupledUnsteadyProblem
        ):
            raise TypeError(
                "coupled_unsteady_problem must be a MultiBodyCoupledUnsteadyProblem."
            )
        self.coupled_unsteady_problem = coupled_unsteady_problem
        self.mujoco_model = self.coupled_unsteady_problem.mujoco_model

        self.num_steps = self.coupled_unsteady_problem.num_steps
        self.delta_time = self.coupled_unsteady_problem.delta_time
        self._current_step = 0
        self._prescribed_wake = True
        self._history_stride = 1
        self._save_every_n_steps: int | None = None
        self._history_save_dir: Path | None = None
        self._store_full_history = True
        self.full_history_available = True

        self.multi_body_coupled_steady_problems = (
            self.coupled_unsteady_problem.multi_body_coupled_steady_problems
        )
        first_problem = self.multi_body_coupled_steady_problems[0]
        self.current_airplanes: tuple[geometry.airplane.Airplane, ...] = (
            first_problem.airplanes
        )
        self.current_coupled_operating_points: tuple[
            operating_point.CoupledOperatingPoint, ...
        ] = first_problem.coupled_operating_points
        self.current_positions_E_E: tuple[np.ndarray, ...] = first_problem.positions_E_E
        self.num_airplanes = len(self.current_airplanes)
        self.num_bodies = self.num_airplanes

        self._validate_shared_environment(self.current_coupled_operating_points)
        self._shared_rho = self.current_coupled_operating_points[0].rho
        self._shared_nu = self.current_coupled_operating_points[0].nu
        self._shared_g_E = self.current_coupled_operating_points[0].g_E.copy()

        self.num_panels = sum(
            airplane.num_panels for airplane in self.current_airplanes
        )

        self._current_body_positions_E_E = np.zeros((self.num_bodies, 3), dtype=float)
        self._current_body_velocities_E__E = np.zeros((self.num_bodies, 3), dtype=float)
        self._current_body_omegas_E__E = np.zeros((self.num_bodies, 3), dtype=float)
        self._current_body_R_pas_GP_to_Es = np.zeros(
            (self.num_bodies, 3, 3), dtype=float
        )
        self._last_body_positions_E_E = np.zeros((self.num_bodies, 3), dtype=float)
        self._last_body_R_pas_GP_to_Es = np.zeros((self.num_bodies, 3, 3), dtype=float)

        self._currentStackFreestreamWingInfluences__E = np.empty(0, dtype=float)
        self._currentGridWingWingInfluences__E = np.empty(0, dtype=float)
        self._currentStackWakeWingInfluences__E = np.empty(0, dtype=float)
        self._current_bound_vortex_strengths = np.empty(0, dtype=float)
        self._last_bound_vortex_strengths = np.empty(0, dtype=float)

        self.panels: np.ndarray = np.empty(0, dtype=object)
        self.stackUnitNormals_GP1 = np.empty(0, dtype=float)
        self.panel_areas = np.empty(0, dtype=float)
        self.panel_airplane_indices = np.empty(0, dtype=int)
        self.panel_is_trailing_edge = np.empty(0, dtype=bool)
        self.panel_is_leading_edge = np.empty(0, dtype=bool)
        self.panel_is_left_edge = np.empty(0, dtype=bool)
        self.panel_is_right_edge = np.empty(0, dtype=bool)
        self.stackSeedPoints_GP1_CgP1 = np.empty((0, 3), dtype=float)

        self.stackCpp_GP1_CgP1 = np.empty(0, dtype=float)
        self.stackBrbrvp_GP1_CgP1 = np.empty(0, dtype=float)
        self.stackFrbrvp_GP1_CgP1 = np.empty(0, dtype=float)
        self.stackFlbrvp_GP1_CgP1 = np.empty(0, dtype=float)
        self.stackBlbrvp_GP1_CgP1 = np.empty(0, dtype=float)
        self.stackCblvpr_GP1_CgP1 = np.empty(0, dtype=float)
        self.stackCblvpf_GP1_CgP1 = np.empty(0, dtype=float)
        self.stackCblvpl_GP1_CgP1 = np.empty(0, dtype=float)
        self.stackCblvpb_GP1_CgP1 = np.empty(0, dtype=float)
        self.stackRbrv_GP1 = np.empty(0, dtype=float)
        self.stackFbrv_GP1 = np.empty(0, dtype=float)
        self.stackLbrv_GP1 = np.empty(0, dtype=float)
        self.stackBbrv_GP1 = np.empty(0, dtype=float)
        self._lastStackCpp_GP1_CgP1 = np.empty(0, dtype=float)
        self._lastStackCblvpr_GP1_CgP1 = np.empty(0, dtype=float)
        self._lastStackCblvpf_GP1_CgP1 = np.empty(0, dtype=float)
        self._lastStackCblvpl_GP1_CgP1 = np.empty(0, dtype=float)
        self._lastStackCblvpb_GP1_CgP1 = np.empty(0, dtype=float)

        self._current_wake_vortex_strengths = np.empty(0, dtype=float)
        self._current_wake_vortex_ages = np.empty(0, dtype=float)
        self._currentStackBrwrvp_GP1_CgP1 = np.empty(0, dtype=float)
        self._currentStackFrwrvp_GP1_CgP1 = np.empty(0, dtype=float)
        self._currentStackFlwrvp_GP1_CgP1 = np.empty(0, dtype=float)
        self._currentStackBlwrvp_GP1_CgP1 = np.empty(0, dtype=float)
        self._currentStackBoundRc0s = np.empty(0, dtype=float)
        self._currentStackWakeRc0s = np.empty(0, dtype=float)

        self.list_num_wake_vortices: list[int] = []
        self._list_wake_vortex_strengths: list[np.ndarray | None] = []
        self._list_wake_vortex_ages: list[np.ndarray | None] = []
        self._list_wake_rc0s: list[np.ndarray | None] = []
        self.listStackBrwrvp_GP1_CgP1: list[np.ndarray | None] = []
        self.listStackFrwrvp_GP1_CgP1: list[np.ndarray | None] = []
        self.listStackFlwrvp_GP1_CgP1: list[np.ndarray | None] = []
        self.listStackBlwrvp_GP1_CgP1: list[np.ndarray | None] = []

        self.stackPositions_E_E = np.zeros(
            (self.num_steps, self.num_bodies, 3), dtype=float
        )
        self.stackR_pas_E_to_BPs = np.zeros(
            (self.num_steps, self.num_bodies, 3, 3), dtype=float
        )

        self._next_positions_E_E = np.empty((0, 3), dtype=float)
        self._next_R_pas_E_to_BPs = np.empty((0, 3, 3), dtype=float)
        self._next_velocities_E__E = np.empty((0, 3), dtype=float)
        self._next_omegas_BPs__E = np.empty((0, 3), dtype=float)

        self._current_total_forces_E = np.zeros((self.num_bodies, 3), dtype=float)
        self._current_total_moments_E_Cg = np.zeros((self.num_bodies, 3), dtype=float)

        self.ran = False

    def run(
        self,
        prescribed_wake: bool | np.bool_ = False,
        show_progress: bool | np.bool_ = True,
        history_stride: int | np.integer = 1,
        save_every_n_steps: int | np.integer | None = None,
        history_save_dir: str | Path | None = None,
    ) -> None:
        """Run the multibody coupled simulation."""
        self._prescribed_wake = _parameter_validation.boolLike_return_bool(
            prescribed_wake, "prescribed_wake"
        )
        show_progress = _parameter_validation.boolLike_return_bool(
            show_progress, "show_progress"
        )
        self._history_stride = _parameter_validation.int_in_range_return_int(
            history_stride,
            "history_stride",
            min_val=1,
            min_inclusive=True,
        )
        if save_every_n_steps is None:
            self._save_every_n_steps = None
        else:
            self._save_every_n_steps = _parameter_validation.int_in_range_return_int(
                save_every_n_steps,
                "save_every_n_steps",
                min_val=1,
                min_inclusive=True,
            )
        if history_save_dir is None:
            self._history_save_dir = None
        else:
            if not isinstance(history_save_dir, (str, Path)):
                raise TypeError("history_save_dir must be a str, Path, or None.")
            self._history_save_dir = Path(history_save_dir)
            self._history_save_dir.mkdir(parents=True, exist_ok=True)
        self._store_full_history = self._history_stride == 1
        self.full_history_available = self._store_full_history

        self._preallocate_wake_arrays()
        initial_states = self.mujoco_model.get_states()
        self.stackPositions_E_E[0] = cast(np.ndarray, initial_states["positions_E_E"])
        self.stackR_pas_E_to_BPs[0] = cast(np.ndarray, initial_states["R_pas_E_to_BPs"])

        with tqdm(
            total=self.num_steps,
            desc="Simulating",
            disable=not show_progress,
            ncols=100,
        ) as bar:
            for step in range(self.num_steps):
                self._current_step = step
                current_problem = self.multi_body_coupled_steady_problems[step]
                self.current_airplanes = current_problem.airplanes
                self.current_coupled_operating_points = (
                    current_problem.coupled_operating_points
                )
                self.current_positions_E_E = current_problem.positions_E_E
                self._validate_shared_environment(self.current_coupled_operating_points)
                self._update_current_body_kinematics()

                self._allocate_step_arrays(step=step)
                if step == 0:
                    self._initialize_panel_vortices(step=0)

                self._collapse_geometry()
                self._calculate_wing_wing_influences()
                self._calculate_freestream_wing_influences()
                self._calculate_wake_wing_influences()
                self._calculate_vortex_strengths()
                self._calculate_loads()
                self._pass_loads_to_mujoco()
                self.mujoco_model.step()
                self._process_new_states_from_mujoco()
                self._create_next_coupled_steady_problem()
                if step < self.num_steps - 1:
                    self._initialize_panel_vortices(step + 1)
                self._populate_next_airplanes_wake()

                if self._save_every_n_steps is not None:
                    if (
                        step % self._save_every_n_steps == 0
                        or step == self.num_steps - 1
                    ):
                        self._save_history_snapshot(step=step)

                if not self._store_full_history and step > 1:
                    self._prune_old_step_history(step=step)

                bar.update(1)

        self.ran = True

    def _preallocate_wake_arrays(self) -> None:
        """Preallocate wake-history arrays for every time step."""
        total_spanwise_panels = 0
        for airplane in self.current_airplanes:
            for wing in airplane.wings:
                _num_spanwise = wing.num_spanwise_panels
                assert _num_spanwise is not None
                total_spanwise_panels += _num_spanwise

        for step in range(self.num_steps):
            this_num_wake_ring_vortices = step * total_spanwise_panels
            keep_step_history = self._store_full_history or self._should_retain_step(
                step
            )
            self.list_num_wake_vortices.append(this_num_wake_ring_vortices)
            if keep_step_history:
                self._list_wake_vortex_strengths.append(
                    np.zeros(this_num_wake_ring_vortices, dtype=float)
                )
                self._list_wake_vortex_ages.append(
                    np.zeros(this_num_wake_ring_vortices, dtype=float)
                )
                self._list_wake_rc0s.append(
                    np.zeros(this_num_wake_ring_vortices, dtype=float)
                )
                self.listStackBrwrvp_GP1_CgP1.append(
                    np.zeros((this_num_wake_ring_vortices, 3), dtype=float)
                )
                self.listStackFrwrvp_GP1_CgP1.append(
                    np.zeros((this_num_wake_ring_vortices, 3), dtype=float)
                )
                self.listStackFlwrvp_GP1_CgP1.append(
                    np.zeros((this_num_wake_ring_vortices, 3), dtype=float)
                )
                self.listStackBlwrvp_GP1_CgP1.append(
                    np.zeros((this_num_wake_ring_vortices, 3), dtype=float)
                )
            else:
                self._list_wake_vortex_strengths.append(None)
                self._list_wake_vortex_ages.append(None)
                self._list_wake_rc0s.append(None)
                self.listStackBrwrvp_GP1_CgP1.append(None)
                self.listStackFrwrvp_GP1_CgP1.append(None)
                self.listStackFlwrvp_GP1_CgP1.append(None)
                self.listStackBlwrvp_GP1_CgP1.append(None)

    def _allocate_step_arrays(self, step: int) -> None:
        """Allocate or reset per-step arrays."""
        self._currentStackFreestreamWingInfluences__E = np.zeros(
            self.num_panels, dtype=float
        )
        self._currentGridWingWingInfluences__E = np.zeros(
            (self.num_panels, self.num_panels), dtype=float
        )
        self._currentStackWakeWingInfluences__E = np.zeros(self.num_panels, dtype=float)
        self._current_bound_vortex_strengths = np.ones(self.num_panels, dtype=float)
        if step == 0:
            self._last_bound_vortex_strengths = np.zeros(self.num_panels, dtype=float)

        self.panels = np.empty(self.num_panels, dtype=object)
        self.stackUnitNormals_GP1 = np.zeros((self.num_panels, 3), dtype=float)
        self.panel_areas = np.zeros(self.num_panels, dtype=float)
        self.panel_airplane_indices = np.zeros(self.num_panels, dtype=int)
        self.panel_is_trailing_edge = np.zeros(self.num_panels, dtype=bool)
        self.panel_is_leading_edge = np.zeros(self.num_panels, dtype=bool)
        self.panel_is_left_edge = np.zeros(self.num_panels, dtype=bool)
        self.panel_is_right_edge = np.zeros(self.num_panels, dtype=bool)
        self.stackSeedPoints_GP1_CgP1 = np.zeros((0, 3), dtype=float)

        self.stackCpp_GP1_CgP1 = np.zeros((self.num_panels, 3), dtype=float)
        self.stackBrbrvp_GP1_CgP1 = np.zeros((self.num_panels, 3), dtype=float)
        self.stackFrbrvp_GP1_CgP1 = np.zeros((self.num_panels, 3), dtype=float)
        self.stackFlbrvp_GP1_CgP1 = np.zeros((self.num_panels, 3), dtype=float)
        self.stackBlbrvp_GP1_CgP1 = np.zeros((self.num_panels, 3), dtype=float)
        self.stackCblvpr_GP1_CgP1 = np.zeros((self.num_panels, 3), dtype=float)
        self.stackCblvpf_GP1_CgP1 = np.zeros((self.num_panels, 3), dtype=float)
        self.stackCblvpl_GP1_CgP1 = np.zeros((self.num_panels, 3), dtype=float)
        self.stackCblvpb_GP1_CgP1 = np.zeros((self.num_panels, 3), dtype=float)
        self.stackRbrv_GP1 = np.zeros((self.num_panels, 3), dtype=float)
        self.stackFbrv_GP1 = np.zeros((self.num_panels, 3), dtype=float)
        self.stackLbrv_GP1 = np.zeros((self.num_panels, 3), dtype=float)
        self.stackBbrv_GP1 = np.zeros((self.num_panels, 3), dtype=float)
        self._lastStackCpp_GP1_CgP1 = np.zeros((self.num_panels, 3), dtype=float)
        self._lastStackCblvpr_GP1_CgP1 = np.zeros((self.num_panels, 3), dtype=float)
        self._lastStackCblvpf_GP1_CgP1 = np.zeros((self.num_panels, 3), dtype=float)
        self._lastStackCblvpl_GP1_CgP1 = np.zeros((self.num_panels, 3), dtype=float)
        self._lastStackCblvpb_GP1_CgP1 = np.zeros((self.num_panels, 3), dtype=float)

        num_wake_ring_vortices = self.list_num_wake_vortices[step]
        wake_strengths = self._list_wake_vortex_strengths[step]
        self._current_wake_vortex_strengths = (
            np.zeros(num_wake_ring_vortices, dtype=float)
            if wake_strengths is None
            else wake_strengths
        )
        wake_ages = self._list_wake_vortex_ages[step]
        self._current_wake_vortex_ages = (
            np.zeros(num_wake_ring_vortices, dtype=float)
            if wake_ages is None
            else wake_ages
        )
        wake_br = self.listStackBrwrvp_GP1_CgP1[step]
        self._currentStackBrwrvp_GP1_CgP1 = (
            np.zeros((num_wake_ring_vortices, 3), dtype=float)
            if wake_br is None
            else wake_br
        )
        wake_fr = self.listStackFrwrvp_GP1_CgP1[step]
        self._currentStackFrwrvp_GP1_CgP1 = (
            np.zeros((num_wake_ring_vortices, 3), dtype=float)
            if wake_fr is None
            else wake_fr
        )
        wake_fl = self.listStackFlwrvp_GP1_CgP1[step]
        self._currentStackFlwrvp_GP1_CgP1 = (
            np.zeros((num_wake_ring_vortices, 3), dtype=float)
            if wake_fl is None
            else wake_fl
        )
        wake_bl = self.listStackBlwrvp_GP1_CgP1[step]
        self._currentStackBlwrvp_GP1_CgP1 = (
            np.zeros((num_wake_ring_vortices, 3), dtype=float)
            if wake_bl is None
            else wake_bl
        )
        self._currentStackBoundRc0s = np.zeros(self.num_panels, dtype=float)
        wake_rc0s = self._list_wake_rc0s[step]
        self._currentStackWakeRc0s = (
            np.zeros(num_wake_ring_vortices, dtype=float)
            if wake_rc0s is None
            else wake_rc0s
        )
        self._current_total_forces_E[:] = 0.0
        self._current_total_moments_E_Cg[:] = 0.0

    def _should_retain_step(self, step: int) -> bool:
        """Return whether this time step should be retained in memory."""
        return (
            step == 0 or step == self.num_steps - 1 or step % self._history_stride == 0
        )

    def _save_history_snapshot(self, step: int) -> None:
        """Persist one multibody solver snapshot to disk."""
        if self._history_save_dir is None:
            return

        snapshot_path = self._history_save_dir / f"step_{step:06d}.npz"
        np.savez_compressed(
            snapshot_path,
            step=np.array([step], dtype=int),
            positions_E_E=self._next_positions_E_E.copy(),
            R_pas_E_to_BPs=self._next_R_pas_E_to_BPs.copy(),
            velocities_E__E=self._next_velocities_E__E.copy(),
            omegas_BPs__E=self._next_omegas_BPs__E.copy(),
            wake_strengths=self._current_wake_vortex_strengths.copy(),
            wake_ages=self._current_wake_vortex_ages.copy(),
            wake_rc0s=self._currentStackWakeRc0s.copy(),
            wake_br=self._currentStackBrwrvp_GP1_CgP1.copy(),
            wake_fr=self._currentStackFrwrvp_GP1_CgP1.copy(),
            wake_fl=self._currentStackFlwrvp_GP1_CgP1.copy(),
            wake_bl=self._currentStackBlwrvp_GP1_CgP1.copy(),
        )

    def _prune_old_step_history(self, step: int) -> None:
        """Drop heavy history objects that are no longer needed for the solve."""
        prune_step = step - 1
        if prune_step <= 0 or self._should_retain_step(prune_step):
            return

        self._list_wake_vortex_strengths[prune_step] = None
        self._list_wake_vortex_ages[prune_step] = None
        self._list_wake_rc0s[prune_step] = None
        self.listStackBrwrvp_GP1_CgP1[prune_step] = None
        self.listStackFrwrvp_GP1_CgP1[prune_step] = None
        self.listStackFlwrvp_GP1_CgP1[prune_step] = None
        self.listStackBlwrvp_GP1_CgP1[prune_step] = None

        if prune_step < len(self.multi_body_coupled_steady_problems):
            self.multi_body_coupled_steady_problems[prune_step] = None  # type: ignore[assignment]

    def _update_current_body_kinematics(self) -> None:
        """Refresh per-body position, velocity, and Earth-frame angular velocity."""
        for body_index, (coupled_operating_point, body_position_E_E) in enumerate(
            zip(
                self.current_coupled_operating_points,
                self.current_positions_E_E,
                strict=True,
            )
        ):
            self._current_body_positions_E_E[body_index] = body_position_E_E
            self._current_body_velocities_E__E[body_index] = (
                coupled_operating_point.vCg_E__E
            )
            omegas_BP = coupled_operating_point.omegas_BP1__E
            self._current_body_omegas_E__E[body_index] = (
                _transformations.apply_T_to_vectors(
                    coupled_operating_point.T_pas_BP1_CgP1_to_E_CgP1,
                    omegas_BP,
                    has_point=False,
                )
            )
            self._current_body_R_pas_GP_to_Es[body_index] = (
                coupled_operating_point.T_pas_GP1_CgP1_to_E_CgP1[:3, :3]
            )

    def _initialize_panel_vortices(self, step: int) -> None:
        """Initialize bound ring vortices for one multibody time step."""
        this_problem = self.multi_body_coupled_steady_problems[step]

        for airplane_index, airplane in enumerate(this_problem.airplanes):
            for wing_index, wing in enumerate(airplane.wings):
                _num_spanwise = wing.num_spanwise_panels
                assert _num_spanwise is not None

                for chordwise_position in range(wing.num_chordwise_panels):
                    for spanwise_position in range(_num_spanwise):
                        _panels = wing.panels
                        assert _panels is not None
                        panel: _panel.Panel = _panels[
                            chordwise_position, spanwise_position
                        ]

                        _Flbvp = panel.Flbvp_GP1_CgP1
                        _Frbvp = panel.Frbvp_GP1_CgP1
                        assert _Flbvp is not None
                        assert _Frbvp is not None
                        Flrvp = _Flbvp
                        Frrvp = _Frbvp

                        if not panel.is_trailing_edge:
                            next_panel: _panel.Panel = _panels[
                                chordwise_position + 1, spanwise_position
                            ]
                            _next_fleft = next_panel.Flbvp_GP1_CgP1
                            _next_fright = next_panel.Frbvp_GP1_CgP1
                            assert _next_fleft is not None
                            assert _next_fright is not None
                            Blrvp = _next_fleft
                            Brrvp = _next_fright
                        else:
                            _Blpp = panel.Blpp_GP1_CgP1
                            _Brpp = panel.Brpp_GP1_CgP1
                            assert _Blpp is not None
                            assert _Brpp is not None

                            current_points_E = np.vstack([_Blpp, _Brpp])
                            airplane_indices = np.array(
                                [airplane_index, airplane_index], dtype=int
                            )
                            if step == 0:
                                current_body_states = (
                                    self._get_body_state_arrays_for_step(step)
                                )
                                apparent_velocities = self._calculate_surface_apparent_velocities_from_body_states(
                                    points_E=current_points_E,
                                    airplane_indices=airplane_indices,
                                    last_points_E=None,
                                    current_positions_E_E=current_body_states[0],
                                    current_velocities_E__E=current_body_states[1],
                                    current_omegas_E__E=current_body_states[2],
                                    current_R_pas_GP_to_Es=current_body_states[3],
                                )
                            else:
                                last_problem = self.multi_body_coupled_steady_problems[
                                    step - 1
                                ]
                                last_panel = (
                                    last_problem.airplanes[airplane_index]
                                    .wings[wing_index]
                                    .panels[chordwise_position, spanwise_position]
                                )
                                assert last_panel.Blpp_GP1_CgP1 is not None
                                assert last_panel.Brpp_GP1_CgP1 is not None
                                last_points_E = np.vstack(
                                    [
                                        last_panel.Blpp_GP1_CgP1,
                                        last_panel.Brpp_GP1_CgP1,
                                    ]
                                )
                                current_body_states = (
                                    self._get_body_state_arrays_for_step(step)
                                )
                                last_body_states = self._get_body_state_arrays_for_step(
                                    step - 1
                                )
                                apparent_velocities = self._calculate_surface_apparent_velocities_from_body_states(
                                    points_E=current_points_E,
                                    airplane_indices=airplane_indices,
                                    last_points_E=last_points_E,
                                    current_positions_E_E=current_body_states[0],
                                    current_velocities_E__E=current_body_states[1],
                                    current_omegas_E__E=current_body_states[2],
                                    current_R_pas_GP_to_Es=current_body_states[3],
                                    last_positions_E_E=last_body_states[0],
                                    last_R_pas_GP_to_Es=last_body_states[3],
                                    delta_time=self.delta_time,
                                )

                            Blrvp = (
                                _Blpp + apparent_velocities[0] * self.delta_time * 0.25
                            )
                            Brrvp = (
                                _Brpp + apparent_velocities[1] * self.delta_time * 0.25
                            )

                        panel.ring_vortex = _vortices.ring_vortex.RingVortex(
                            Flrvp_GP1_CgP1=Flrvp,
                            Frrvp_GP1_CgP1=Frrvp,
                            Blrvp_GP1_CgP1=Blrvp,
                            Brrvp_GP1_CgP1=Brrvp,
                            strength=1.0,
                        )

    def _collapse_geometry(self) -> None:
        """Collapse the current multibody geometry into vectorized arrays."""
        global_panel_position = 0
        global_wake_position = 0

        for airplane_index, airplane in enumerate(self.current_airplanes):
            for wing in airplane.wings:
                _standard_mean_chord = wing.standard_mean_chord
                assert _standard_mean_chord is not None
                wing_r_c0 = 0.03 * _standard_mean_chord

                _panels = wing.panels
                _wake_ring_vortices = wing.wake_ring_vortices
                assert _panels is not None
                assert _wake_ring_vortices is not None

                for panel in np.ravel(_panels):
                    _functions.update_ring_vortex_solvers_panel_attributes(
                        ring_vortex_solver=self,
                        global_panel_position=global_panel_position,
                        panel=panel,
                    )
                    self._currentStackBoundRc0s[global_panel_position] = wing_r_c0
                    self.panel_airplane_indices[global_panel_position] = airplane_index
                    global_panel_position += 1

                for wake_ring_vortex in np.ravel(_wake_ring_vortices):
                    self._current_wake_vortex_strengths[global_wake_position] = (
                        wake_ring_vortex.strength
                    )
                    self._current_wake_vortex_ages[global_wake_position] = (
                        wake_ring_vortex.age
                    )
                    self._currentStackFrwrvp_GP1_CgP1[global_wake_position] = (
                        wake_ring_vortex.Frrvp_GP1_CgP1
                    )
                    self._currentStackFlwrvp_GP1_CgP1[global_wake_position] = (
                        wake_ring_vortex.Flrvp_GP1_CgP1
                    )
                    self._currentStackBlwrvp_GP1_CgP1[global_wake_position] = (
                        wake_ring_vortex.Blrvp_GP1_CgP1
                    )
                    self._currentStackBrwrvp_GP1_CgP1[global_wake_position] = (
                        wake_ring_vortex.Brrvp_GP1_CgP1
                    )
                    self._currentStackWakeRc0s[global_wake_position] = wing_r_c0
                    global_wake_position += 1

        if self._current_step > 0:
            last_problem = self.multi_body_coupled_steady_problems[
                self._current_step - 1
            ]
            for body_index, (
                last_coupled_operating_point,
                last_position_E_E,
            ) in enumerate(
                zip(
                    last_problem.coupled_operating_points,
                    last_problem.positions_E_E,
                    strict=True,
                )
            ):
                self._last_body_positions_E_E[body_index] = last_position_E_E
                self._last_body_R_pas_GP_to_Es[body_index] = (
                    last_coupled_operating_point.T_pas_GP1_CgP1_to_E_CgP1[:3, :3]
                )

            global_panel_position = 0
            for last_airplane in last_problem.airplanes:
                for last_wing in last_airplane.wings:
                    _last_panels = last_wing.panels
                    assert _last_panels is not None
                    for last_panel in np.ravel(_last_panels):
                        last_ring_vortex = last_panel.ring_vortex
                        assert last_ring_vortex is not None
                        self._lastStackCpp_GP1_CgP1[global_panel_position, :] = (
                            last_panel.Cpp_GP1_CgP1
                        )
                        self._lastStackCblvpr_GP1_CgP1[global_panel_position, :] = (
                            last_ring_vortex.right_leg.Clvp_GP1_CgP1
                        )
                        self._lastStackCblvpf_GP1_CgP1[global_panel_position, :] = (
                            last_ring_vortex.front_leg.Clvp_GP1_CgP1
                        )
                        self._lastStackCblvpl_GP1_CgP1[global_panel_position, :] = (
                            last_ring_vortex.left_leg.Clvp_GP1_CgP1
                        )
                        self._lastStackCblvpb_GP1_CgP1[global_panel_position, :] = (
                            last_ring_vortex.back_leg.Clvp_GP1_CgP1
                        )
                        self._last_bound_vortex_strengths[global_panel_position] = (
                            last_ring_vortex.strength
                        )
                        global_panel_position += 1

    def _calculate_wing_wing_influences(self) -> None:
        """Calculate the current bound-vortex influence matrix."""
        singularity_counts = np.zeros(4, dtype=np.int64)
        grid_norm_v_ind = (
            _aerodynamics_functions.expanded_velocities_from_ring_vortices(
                stackP_GP1_CgP1=self.stackCpp_GP1_CgP1,
                stackBrrvp_GP1_CgP1=self.stackBrbrvp_GP1_CgP1,
                stackFrrvp_GP1_CgP1=self.stackFrbrvp_GP1_CgP1,
                stackFlrvp_GP1_CgP1=self.stackFlbrvp_GP1_CgP1,
                stackBlrvp_GP1_CgP1=self.stackBlbrvp_GP1_CgP1,
                strengths=self._current_bound_vortex_strengths,
                r_c0s=self._currentStackBoundRc0s,
                singularity_counts=singularity_counts,
                ages=None,
                nu=self._shared_nu,
            )
        )
        _functions.log_unexpected_singularity_counts(
            _logger,
            logging.ERROR,
            "_calculate_wing_wing_influences",
            np.copy(singularity_counts),
        )
        self._currentGridWingWingInfluences__E = np.einsum(
            "...k,...k->...",
            grid_norm_v_ind,
            np.expand_dims(self.stackUnitNormals_GP1, axis=1),
        )

    def _calculate_freestream_wing_influences(self) -> None:
        """Calculate the normal apparent-flow influence at each collocation point."""
        apparent_velocities = self._calculate_surface_apparent_velocities_at_points(
            points_E=self.stackCpp_GP1_CgP1,
            airplane_indices=self.panel_airplane_indices,
            last_points_E=self._lastStackCpp_GP1_CgP1,
        )
        self._currentStackFreestreamWingInfluences__E = np.einsum(
            "ij,ij->i", self.stackUnitNormals_GP1, apparent_velocities
        )

    def _calculate_wake_wing_influences(self) -> None:
        """Calculate wake influence coefficients at each collocation point."""
        if self._current_step == 0:
            self._currentStackWakeWingInfluences__E = np.zeros(
                self.num_panels, dtype=float
            )
            return

        singularity_counts = np.zeros(4, dtype=np.int64)
        wake_velocities = (
            _aerodynamics_functions.collapsed_velocities_from_ring_vortices(
                stackP_GP1_CgP1=self.stackCpp_GP1_CgP1,
                stackBrrvp_GP1_CgP1=self._currentStackBrwrvp_GP1_CgP1,
                stackFrrvp_GP1_CgP1=self._currentStackFrwrvp_GP1_CgP1,
                stackFlrvp_GP1_CgP1=self._currentStackFlwrvp_GP1_CgP1,
                stackBlrvp_GP1_CgP1=self._currentStackBlwrvp_GP1_CgP1,
                strengths=self._current_wake_vortex_strengths,
                r_c0s=self._currentStackWakeRc0s,
                singularity_counts=singularity_counts,
                ages=self._current_wake_vortex_ages,
                nu=self._shared_nu,
            )
        )
        _functions.log_unexpected_singularity_counts(
            _logger,
            logging.INFO,
            "_calculate_wake_wing_influences",
            np.copy(singularity_counts),
        )
        self._currentStackWakeWingInfluences__E = np.einsum(
            "ij,ij->i", wake_velocities, self.stackUnitNormals_GP1
        )

    def _calculate_vortex_strengths(self) -> None:
        """Solve for the current bound-vortex strengths."""
        self._current_bound_vortex_strengths = np.linalg.solve(
            self._currentGridWingWingInfluences__E,
            -self._currentStackWakeWingInfluences__E
            - self._currentStackFreestreamWingInfluences__E,
        )
        for panel_index, panel in enumerate(self.panels):
            this_ring_vortex = panel.ring_vortex
            assert this_ring_vortex is not None
            this_ring_vortex.strength = self._current_bound_vortex_strengths[
                panel_index
            ]

    def calculate_solution_velocity(
        self,
        stackP_GP1_CgP1: np.ndarray | Sequence[Sequence[float | int]],
        bound_singularity_counts: np.ndarray | None = None,
        wake_singularity_counts: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the induced fluid velocity at the requested Earth-frame points."""
        stackP = (
            _parameter_validation.arrayLike_of_threeD_number_vectorLikes_return_float(
                stackP_GP1_CgP1, "stackP_GP1_CgP1"
            )
        )
        if bound_singularity_counts is None:
            bound_singularity_counts = np.zeros(4, dtype=np.int64)
        if wake_singularity_counts is None:
            wake_singularity_counts = np.zeros(4, dtype=np.int64)

        bound_velocity = (
            _aerodynamics_functions.collapsed_velocities_from_ring_vortices(
                stackP_GP1_CgP1=stackP,
                stackBrrvp_GP1_CgP1=self.stackBrbrvp_GP1_CgP1,
                stackFrrvp_GP1_CgP1=self.stackFrbrvp_GP1_CgP1,
                stackFlrvp_GP1_CgP1=self.stackFlbrvp_GP1_CgP1,
                stackBlrvp_GP1_CgP1=self.stackBlbrvp_GP1_CgP1,
                strengths=self._current_bound_vortex_strengths,
                r_c0s=self._currentStackBoundRc0s,
                singularity_counts=bound_singularity_counts,
                ages=None,
                nu=self._shared_nu,
            )
        )
        wake_velocity = _aerodynamics_functions.collapsed_velocities_from_ring_vortices(
            stackP_GP1_CgP1=stackP,
            stackBrrvp_GP1_CgP1=self._currentStackBrwrvp_GP1_CgP1,
            stackFrrvp_GP1_CgP1=self._currentStackFrwrvp_GP1_CgP1,
            stackFlrvp_GP1_CgP1=self._currentStackFlwrvp_GP1_CgP1,
            stackBlrvp_GP1_CgP1=self._currentStackBlwrvp_GP1_CgP1,
            strengths=self._current_wake_vortex_strengths,
            r_c0s=self._currentStackWakeRc0s,
            singularity_counts=wake_singularity_counts,
            ages=self._current_wake_vortex_ages,
            nu=self._shared_nu,
        )
        return cast(np.ndarray, bound_velocity + wake_velocity)

    def _calculate_loads(self) -> None:
        """Calculate panel loads and reduce them to one net load per body."""
        global_panel_position = 0
        effective_right = np.zeros(self.num_panels, dtype=float)
        effective_front = np.zeros(self.num_panels, dtype=float)
        effective_left = np.zeros(self.num_panels, dtype=float)
        effective_back = np.zeros(self.num_panels, dtype=float)

        for airplane in self.current_airplanes:
            for wing in airplane.wings:
                _panels = wing.panels
                assert _panels is not None
                for panel in np.ravel(_panels):
                    chordwise = panel.local_chordwise_position
                    spanwise = panel.local_spanwise_position
                    assert chordwise is not None
                    assert spanwise is not None

                    if panel.is_right_edge:
                        effective_right[global_panel_position] = (
                            self._current_bound_vortex_strengths[global_panel_position]
                        )
                    else:
                        panel_to_right = _panels[chordwise, spanwise + 1]
                        ring_vortex_to_right = panel_to_right.ring_vortex
                        assert ring_vortex_to_right is not None
                        effective_right[global_panel_position] = (
                            self._current_bound_vortex_strengths[global_panel_position]
                            - ring_vortex_to_right.strength
                        ) / 2

                    if panel.is_leading_edge:
                        effective_front[global_panel_position] = (
                            self._current_bound_vortex_strengths[global_panel_position]
                        )
                    else:
                        panel_to_front = _panels[chordwise - 1, spanwise]
                        ring_vortex_to_front = panel_to_front.ring_vortex
                        assert ring_vortex_to_front is not None
                        effective_front[global_panel_position] = (
                            self._current_bound_vortex_strengths[global_panel_position]
                            - ring_vortex_to_front.strength
                        ) / 2

                    if panel.is_left_edge:
                        effective_left[global_panel_position] = (
                            self._current_bound_vortex_strengths[global_panel_position]
                        )
                    else:
                        panel_to_left = _panels[chordwise, spanwise - 1]
                        ring_vortex_to_left = panel_to_left.ring_vortex
                        assert ring_vortex_to_left is not None
                        effective_left[global_panel_position] = (
                            self._current_bound_vortex_strengths[global_panel_position]
                            - ring_vortex_to_left.strength
                        ) / 2

                    if panel.is_trailing_edge:
                        if self._current_step == 0:
                            effective_back[global_panel_position] = (
                                self._current_bound_vortex_strengths[
                                    global_panel_position
                                ]
                            )
                        else:
                            effective_back[global_panel_position] = (
                                self._current_bound_vortex_strengths[
                                    global_panel_position
                                ]
                                - self._last_bound_vortex_strengths[
                                    global_panel_position
                                ]
                            )
                    else:
                        panel_to_back = _panels[chordwise + 1, spanwise]
                        ring_vortex_to_back = panel_to_back.ring_vortex
                        assert ring_vortex_to_back is not None
                        effective_back[global_panel_position] = (
                            self._current_bound_vortex_strengths[global_panel_position]
                            - ring_vortex_to_back.strength
                        ) / 2

                    global_panel_position += 1

        bound_singularity_counts = np.zeros(4, dtype=np.int64)
        wake_singularity_counts = np.zeros(4, dtype=np.int64)
        velocity_right = self.calculate_solution_velocity(
            self.stackCblvpr_GP1_CgP1,
            bound_singularity_counts=bound_singularity_counts,
            wake_singularity_counts=wake_singularity_counts,
        ) + self._calculate_surface_apparent_velocities_at_points(
            self.stackCblvpr_GP1_CgP1,
            self.panel_airplane_indices,
            self._lastStackCblvpr_GP1_CgP1,
        )
        velocity_front = self.calculate_solution_velocity(
            self.stackCblvpf_GP1_CgP1,
            bound_singularity_counts=bound_singularity_counts,
            wake_singularity_counts=wake_singularity_counts,
        ) + self._calculate_surface_apparent_velocities_at_points(
            self.stackCblvpf_GP1_CgP1,
            self.panel_airplane_indices,
            self._lastStackCblvpf_GP1_CgP1,
        )
        velocity_left = self.calculate_solution_velocity(
            self.stackCblvpl_GP1_CgP1,
            bound_singularity_counts=bound_singularity_counts,
            wake_singularity_counts=wake_singularity_counts,
        ) + self._calculate_surface_apparent_velocities_at_points(
            self.stackCblvpl_GP1_CgP1,
            self.panel_airplane_indices,
            self._lastStackCblvpl_GP1_CgP1,
        )
        velocity_back = self.calculate_solution_velocity(
            self.stackCblvpb_GP1_CgP1,
            bound_singularity_counts=bound_singularity_counts,
            wake_singularity_counts=wake_singularity_counts,
        ) + self._calculate_surface_apparent_velocities_at_points(
            self.stackCblvpb_GP1_CgP1,
            self.panel_airplane_indices,
            self._lastStackCblvpb_GP1_CgP1,
        )

        expected_bound_collinearity = 0
        expected_wake_collinearity = 0
        for airplane in self.current_airplanes:
            for wing in airplane.wings:
                num_chordwise = wing.num_chordwise_panels
                num_spanwise = wing.num_spanwise_panels
                assert num_spanwise is not None
                n = num_chordwise * num_spanwise
                expected_bound_collinearity += (
                    8 * n - 2 * num_chordwise - 2 * num_spanwise
                )
                if self._current_step > 0:
                    expected_wake_collinearity += num_spanwise

        unexpected_bound_singularity_counts = np.copy(bound_singularity_counts)
        unexpected_wake_singularity_counts = np.copy(wake_singularity_counts)
        unexpected_bound_singularity_counts[3] -= expected_bound_collinearity
        unexpected_wake_singularity_counts[3] -= expected_wake_collinearity

        _functions.log_unexpected_singularity_counts(
            _logger,
            logging.ERROR,
            "_calculate_loads (bound)",
            unexpected_bound_singularity_counts,
        )
        _functions.log_unexpected_singularity_counts(
            _logger,
            logging.INFO,
            "_calculate_loads (wake)",
            unexpected_wake_singularity_counts,
        )

        right_forces = (
            self._shared_rho
            * np.expand_dims(effective_right, axis=1)
            * _functions.numba_1d_explicit_cross(velocity_right, self.stackRbrv_GP1)
        )
        front_forces = (
            self._shared_rho
            * np.expand_dims(effective_front, axis=1)
            * _functions.numba_1d_explicit_cross(velocity_front, self.stackFbrv_GP1)
        )
        left_forces = (
            self._shared_rho
            * np.expand_dims(effective_left, axis=1)
            * _functions.numba_1d_explicit_cross(velocity_left, self.stackLbrv_GP1)
        )
        back_forces = (
            self._shared_rho
            * np.expand_dims(effective_back, axis=1)
            * _functions.numba_1d_explicit_cross(velocity_back, self.stackBbrv_GP1)
        )
        unsteady_forces = -(
            self._shared_rho
            * np.expand_dims(
                self._current_bound_vortex_strengths
                - self._last_bound_vortex_strengths,
                axis=1,
            )
            * np.expand_dims(self.panel_areas, axis=1)
            * self.stackUnitNormals_GP1
            / self.delta_time
        )
        forces_E = (
            right_forces + front_forces + left_forces + back_forces + unsteady_forces
        )

        right_moments_origin = _functions.numba_1d_explicit_cross(
            self.stackCblvpr_GP1_CgP1, right_forces
        )
        front_moments_origin = _functions.numba_1d_explicit_cross(
            self.stackCblvpf_GP1_CgP1, front_forces
        )
        left_moments_origin = _functions.numba_1d_explicit_cross(
            self.stackCblvpl_GP1_CgP1, left_forces
        )
        back_moments_origin = _functions.numba_1d_explicit_cross(
            self.stackCblvpb_GP1_CgP1, back_forces
        )
        unsteady_moments_origin = _functions.numba_1d_explicit_cross(
            self.stackCpp_GP1_CgP1, unsteady_forces
        )
        moments_origin = (
            right_moments_origin
            + front_moments_origin
            + left_moments_origin
            + back_moments_origin
            + unsteady_moments_origin
        )

        self._process_panel_loads(forces_E=forces_E, moments_origin_E=moments_origin)

    def _process_panel_loads(
        self,
        forces_E: np.ndarray,
        moments_origin_E: np.ndarray,
    ) -> None:
        """Reduce panel loads to one net force/moment per airplane body."""
        self._current_total_forces_E[:] = 0.0
        self._current_total_moments_E_Cg[:] = 0.0

        for panel_index, panel in enumerate(self.panels):
            airplane_index = self.panel_airplane_indices[panel_index]
            body_cg_E = self._current_body_positions_E_E[airplane_index]

            panel.forces_GP1 = forces_E[panel_index]
            panel.moments_GP1_CgP1 = moments_origin_E[panel_index] - np.cross(
                body_cg_E, forces_E[panel_index]
            )

            self._current_total_forces_E[airplane_index] += forces_E[panel_index]
            self._current_total_moments_E_Cg[airplane_index] += panel.moments_GP1_CgP1

        forces_W = np.zeros((self.num_bodies, 3), dtype=float)
        moments_W_Cg = np.zeros((self.num_bodies, 3), dtype=float)
        force_coefficients_W = np.zeros((self.num_bodies, 3), dtype=float)
        moment_coefficients_W_Cg = np.zeros((self.num_bodies, 3), dtype=float)

        for body_index, (airplane, coupled_operating_point) in enumerate(
            zip(
                self.current_airplanes,
                self.current_coupled_operating_points,
                strict=True,
            )
        ):
            total_force_W = _transformations.apply_T_to_vectors(
                coupled_operating_point.T_pas_E_CgP1_to_W_CgP1,
                self._current_total_forces_E[body_index],
                has_point=False,
            )
            total_moment_W = _transformations.apply_T_to_vectors(
                coupled_operating_point.T_pas_E_CgP1_to_W_CgP1,
                self._current_total_moments_E_Cg[body_index],
                has_point=True,
            )

            airplane.forces_W = total_force_W
            airplane.moments_W_CgP1 = total_moment_W
            airplane.forceCoefficients_W = np.array(
                [
                    total_force_W[0] / coupled_operating_point.qInf__E / airplane.s_ref,
                    total_force_W[1] / coupled_operating_point.qInf__E / airplane.s_ref,
                    total_force_W[2] / coupled_operating_point.qInf__E / airplane.s_ref,
                ],
                dtype=float,
            )
            airplane.momentCoefficients_W_CgP1 = np.array(
                [
                    total_moment_W[0]
                    / coupled_operating_point.qInf__E
                    / airplane.s_ref
                    / airplane.b_ref,
                    total_moment_W[1]
                    / coupled_operating_point.qInf__E
                    / airplane.s_ref
                    / airplane.c_ref,
                    total_moment_W[2]
                    / coupled_operating_point.qInf__E
                    / airplane.s_ref
                    / airplane.b_ref,
                ],
                dtype=float,
            )

            forces_W[body_index] = airplane.forces_W
            moments_W_Cg[body_index] = airplane.moments_W_CgP1
            force_coefficients_W[body_index] = airplane.forceCoefficients_W
            moment_coefficients_W_Cg[body_index] = airplane.momentCoefficients_W_CgP1

        self.coupled_unsteady_problem.forces_W.append(forces_W.copy())
        self.coupled_unsteady_problem.forceCoefficients_W.append(
            force_coefficients_W.copy()
        )
        self.coupled_unsteady_problem.moments_W_Cg.append(moments_W_Cg.copy())
        self.coupled_unsteady_problem.momentCoefficients_W_Cg.append(
            moment_coefficients_W_Cg.copy()
        )

    def _pass_loads_to_mujoco(self) -> None:
        """Pass the net aerodynamic and extra loads to MuJoCo."""
        forces_E = self._current_total_forces_E.copy()
        moments_E_Cg = self._current_total_moments_E_Cg.copy()

        for body_index, (airplane, coupled_operating_point) in enumerate(
            zip(
                self.current_airplanes,
                self.current_coupled_operating_points,
                strict=True,
            )
        ):
            forces_E[body_index] += airplane.weight * (
                coupled_operating_point.g_E
                / np.linalg.norm(coupled_operating_point.g_E)
            )
            if self.coupled_unsteady_problem.external_forces_fn is not None:
                extra_forces_W, extra_moments_W_Cg = (
                    self.coupled_unsteady_problem.external_forces_fn(
                        body_index, coupled_operating_point, airplane
                    )
                )
                forces_E[body_index] += _transformations.apply_T_to_vectors(
                    coupled_operating_point.T_pas_W_CgP1_to_E_CgP1,
                    extra_forces_W,
                    has_point=False,
                )
                moments_E_Cg[body_index] += _transformations.apply_T_to_vectors(
                    coupled_operating_point.T_pas_W_CgP1_to_E_CgP1,
                    extra_moments_W_Cg,
                    has_point=True,
                )

        if (
            self._current_step
            >= self.coupled_unsteady_problem.coupled_movement.prescribed_num_steps
        ):
            self.mujoco_model.apply_loads(forces_E=forces_E, moments_E_Cg=moments_E_Cg)
        else:
            self.mujoco_model.apply_loads(
                forces_E=np.zeros_like(forces_E),
                moments_E_Cg=np.zeros_like(moments_E_Cg),
            )

    def _process_new_states_from_mujoco(self) -> None:
        """Read the new MuJoCo states and create the next coupled operating points."""
        states = self.mujoco_model.get_states()
        self._next_positions_E_E = cast(np.ndarray, states["positions_E_E"])
        self._next_R_pas_E_to_BPs = cast(np.ndarray, states["R_pas_E_to_BPs"])
        self._next_velocities_E__E = cast(np.ndarray, states["velocities_E__E"])
        self._next_omegas_BPs__E = cast(np.ndarray, states["omegas_BPs__E"])

        if self._current_step < self.num_steps - 1:
            self.stackPositions_E_E[self._current_step + 1] = (
                self._next_positions_E_E.copy()
            )
            self.stackR_pas_E_to_BPs[self._current_step + 1] = (
                self._next_R_pas_E_to_BPs.copy()
            )

        next_operating_points: list[operating_point.CoupledOperatingPoint] = []
        for body_index, previous_operating_point in enumerate(
            self.current_coupled_operating_points
        ):
            R_pas_E_to_BP = self._next_R_pas_E_to_BPs[body_index]
            angles_E_to_BP_izyx = self._extract_euler_angles_deg(R_pas_E_to_BP)
            velocity_E = self._next_velocities_E__E[body_index]
            speed = float(np.linalg.norm(velocity_E))
            vInf_E = -velocity_E
            vInf_BP = R_pas_E_to_BP @ vInf_E
            u, v, w = vInf_BP
            alpha = np.rad2deg(np.arctan2(-w, -u))
            v_normalized = np.clip(v / (speed + 1e-12), -1.0, 1.0)
            beta = np.rad2deg(np.arcsin(v_normalized))

            next_operating_points.append(
                operating_point.CoupledOperatingPoint(
                    rho=previous_operating_point.rho,
                    vCg__E=speed,
                    omegas_BP1__E=self._next_omegas_BPs__E[body_index],
                    angles_E_to_BP1_izyx=angles_E_to_BP_izyx,
                    alpha=alpha,
                    beta=beta,
                    externalFX_W=previous_operating_point.externalFX_W,
                    nu=previous_operating_point.nu,
                    g_E=previous_operating_point.g_E,
                )
            )

        self.coupled_unsteady_problem.coupled_movement.coupled_operating_points.append(
            tuple(next_operating_points)
        )
        self.coupled_unsteady_problem.coupled_movement.positions_E_E.append(
            tuple(np.copy(position_E_E) for position_E_E in self._next_positions_E_E)
        )

    def _create_next_coupled_steady_problem(self) -> None:
        """Create the next time step's multibody coupled steady problem."""
        if self._current_step >= self.num_steps - 1:
            return

        next_step_id = self._current_step + 1
        prescribed_airplanes = self.coupled_unsteady_problem.coupled_movement.airplanes[
            next_step_id
        ]
        next_airplanes = tuple(
            cast(geometry.airplane.Airplane, airplane.__deepcopy__({}))
            for airplane in prescribed_airplanes
        )
        next_coupled_operating_points = (
            self.coupled_unsteady_problem.coupled_movement.coupled_operating_points[
                next_step_id
            ]
        )
        next_positions_E_E = (
            self.coupled_unsteady_problem.coupled_movement.positions_E_E[next_step_id]
        )
        self.multi_body_coupled_steady_problems.append(
            problems.MultiBodyCoupledSteadyProblem(
                airplanes=next_airplanes,
                coupled_operating_points=next_coupled_operating_points,
                positions_E_E=next_positions_E_E,
            )
        )

    def _populate_next_airplanes_wake(self) -> None:
        """Update the next step's wake state."""
        self._populate_next_airplanes_wake_vortex_points()
        self._populate_next_airplanes_wake_vortices()

    def _populate_next_airplanes_wake_vortex_points(self) -> None:
        """Populate the next step's wake points for the prescribed-wake model."""
        if self._current_step >= self.num_steps - 1:
            return

        next_problem = self.multi_body_coupled_steady_problems[self._current_step + 1]
        next_airplanes = next_problem.airplanes

        for airplane_index, next_airplane in enumerate(next_airplanes):
            this_airplane = self.current_airplanes[airplane_index]
            current_operating_point = self.current_coupled_operating_points[
                airplane_index
            ]
            next_operating_point = next_problem.coupled_operating_points[airplane_index]
            current_position_E_E = self.current_positions_E_E[airplane_index]
            next_position_E_E = next_problem.positions_E_E[airplane_index]
            current_vInf_GP1__E = current_operating_point.vInf_GP1__E
            for wing_index, next_wing in enumerate(next_airplane.wings):
                this_wing = this_airplane.wings[wing_index]

                if self._current_step == 0:
                    num_spanwise_panels = this_wing.num_spanwise_panels
                    assert num_spanwise_panels is not None
                    chordwise_panel_id = this_wing.num_chordwise_panels - 1
                    new_row = np.zeros((1, num_spanwise_panels + 1, 3), dtype=float)

                    for spanwise_panel_id in range(num_spanwise_panels):
                        _next_panels = next_wing.panels
                        assert _next_panels is not None
                        next_panel = _next_panels[chordwise_panel_id, spanwise_panel_id]
                        next_ring_vortex = next_panel.ring_vortex
                        assert next_ring_vortex is not None
                        new_row[0, spanwise_panel_id] = next_ring_vortex.Blrvp_GP1_CgP1
                        if spanwise_panel_id == num_spanwise_panels - 1:
                            new_row[0, spanwise_panel_id + 1] = (
                                next_ring_vortex.Brrvp_GP1_CgP1
                            )

                    next_wing.gridWrvp_GP1_CgP1 = np.copy(new_row)
                    second_row = np.zeros_like(new_row)
                    for spanwise_point_id in range(num_spanwise_panels + 1):
                        first_row_point_E = next_wing.gridWrvp_GP1_CgP1[
                            0, spanwise_point_id
                        ]
                        if self._prescribed_wake:
                            first_row_point_next_GP1 = (
                                _transformations.apply_T_to_vectors(
                                    next_operating_point.T_pas_E_CgP1_to_GP1_CgP1,
                                    first_row_point_E - next_position_E_E,
                                    has_point=True,
                                )
                            )
                            second_row_point_next_GP1 = (
                                first_row_point_next_GP1
                                + current_vInf_GP1__E * self.delta_time
                            )
                            second_row[0, spanwise_point_id] = next_position_E_E + (
                                _transformations.apply_T_to_vectors(
                                    next_operating_point.T_pas_GP1_CgP1_to_E_CgP1,
                                    second_row_point_next_GP1,
                                    has_point=True,
                                )
                            )
                        else:
                            induced_velocity_E = np.squeeze(
                                self.calculate_solution_velocity(
                                    np.expand_dims(first_row_point_E, axis=0)
                                )
                            )
                            surface_apparent_velocity_E = np.squeeze(
                                self._calculate_rigid_body_apparent_velocities_at_points(
                                    points_E=np.expand_dims(first_row_point_E, axis=0),
                                    airplane_indices=np.array(
                                        [airplane_index], dtype=int
                                    ),
                                )
                            )
                            second_row[0, spanwise_point_id] = (
                                first_row_point_E
                                + (surface_apparent_velocity_E + induced_velocity_E)
                                * self.delta_time
                            )
                    next_wing.gridWrvp_GP1_CgP1 = np.vstack(
                        (next_wing.gridWrvp_GP1_CgP1, second_row)
                    )
                else:
                    _this_grid = this_wing.gridWrvp_GP1_CgP1
                    assert _this_grid is not None
                    next_grid = np.zeros_like(_this_grid)

                    if self._prescribed_wake:
                        for chordwise_point_id in range(_this_grid.shape[0]):
                            for spanwise_point_id in range(_this_grid.shape[1]):
                                wake_point_current_E = _this_grid[
                                    chordwise_point_id, spanwise_point_id
                                ]
                                wake_point_current_GP1 = _transformations.apply_T_to_vectors(
                                    current_operating_point.T_pas_E_CgP1_to_GP1_CgP1,
                                    wake_point_current_E - current_position_E_E,
                                    has_point=True,
                                )
                                wake_point_next_GP1 = (
                                    wake_point_current_GP1
                                    + current_vInf_GP1__E * self.delta_time
                                )
                                next_grid[chordwise_point_id, spanwise_point_id] = (
                                    next_position_E_E
                                    + _transformations.apply_T_to_vectors(
                                        next_operating_point.T_pas_GP1_CgP1_to_E_CgP1,
                                        wake_point_next_GP1,
                                        has_point=True,
                                    )
                                )
                    else:
                        next_grid[:] = np.copy(_this_grid)
                        num_chordwise_points = next_grid.shape[0]
                        num_spanwise_points = next_grid.shape[1]
                        for chordwise_point_id in range(num_chordwise_points):
                            for spanwise_point_id in range(num_spanwise_points):
                                wake_point_E = next_grid[
                                    chordwise_point_id, spanwise_point_id
                                ]
                                induced_velocity_E = np.squeeze(
                                    self.calculate_solution_velocity(
                                        np.expand_dims(wake_point_E, axis=0)
                                    )
                                )
                                next_grid[chordwise_point_id, spanwise_point_id] += (
                                    induced_velocity_E * self.delta_time
                                )

                    next_wing.gridWrvp_GP1_CgP1 = next_grid

                    chordwise_panel_id = this_wing.num_chordwise_panels - 1
                    _num_spanwise_panels = this_wing.num_spanwise_panels
                    assert _num_spanwise_panels is not None
                    new_row = np.zeros((1, _num_spanwise_panels + 1, 3), dtype=float)

                    for spanwise_panel_id in range(_num_spanwise_panels):
                        _next_panels = next_wing.panels
                        assert _next_panels is not None
                        next_panel = _next_panels[chordwise_panel_id, spanwise_panel_id]
                        next_ring_vortex = next_panel.ring_vortex
                        assert next_ring_vortex is not None
                        new_row[0, spanwise_panel_id] = next_ring_vortex.Blrvp_GP1_CgP1
                        if spanwise_panel_id == _num_spanwise_panels - 1:
                            new_row[0, spanwise_panel_id + 1] = (
                                next_ring_vortex.Brrvp_GP1_CgP1
                            )

                    next_wing.gridWrvp_GP1_CgP1 = np.vstack(
                        (new_row, next_wing.gridWrvp_GP1_CgP1)
                    )

    def _populate_next_airplanes_wake_vortices(self) -> None:
        """Populate the next step's wake ring-vortex objects."""
        if self._current_step >= self.num_steps - 1:
            return

        next_problem = self.multi_body_coupled_steady_problems[self._current_step + 1]
        next_airplanes = next_problem.airplanes

        for airplane_index, next_airplane in enumerate(next_airplanes):
            for wing_index, this_wing in enumerate(
                self.current_airplanes[airplane_index].wings
            ):
                next_wing = next_airplane.wings[wing_index]
                next_grid = next_wing.gridWrvp_GP1_CgP1
                assert next_grid is not None

                num_chordwise_points = next_grid.shape[0]
                num_spanwise_points = next_grid.shape[1]
                this_wake_ring_vortices = (
                    self.current_airplanes[airplane_index]
                    .wings[wing_index]
                    .wake_ring_vortices
                )
                assert this_wake_ring_vortices is not None

                new_row_of_wake_ring_vortices = np.empty(
                    (1, num_spanwise_points - 1), dtype=object
                )
                next_wing.wake_ring_vortices = np.vstack(
                    (new_row_of_wake_ring_vortices, this_wake_ring_vortices)
                )

                for chordwise_id in range(num_chordwise_points - 1):
                    for spanwise_id in range(num_spanwise_points - 1):
                        front_left = next_grid[chordwise_id, spanwise_id]
                        front_right = next_grid[chordwise_id, spanwise_id + 1]
                        back_left = next_grid[chordwise_id + 1, spanwise_id]
                        back_right = next_grid[chordwise_id + 1, spanwise_id + 1]

                        wake_age: float | None = None
                        if chordwise_id == 0:
                            strength = this_wing.panels[
                                -1, spanwise_id
                            ].ring_vortex.strength
                        else:
                            old_wake_ring_vortex = this_wake_ring_vortices[
                                chordwise_id - 1, spanwise_id
                            ]
                            strength = old_wake_ring_vortex.strength
                            wake_age = old_wake_ring_vortex.age + self.delta_time

                        next_wing.wake_ring_vortices[chordwise_id, spanwise_id] = (
                            _vortices.ring_vortex.RingVortex(
                                Flrvp_GP1_CgP1=front_left,
                                Frrvp_GP1_CgP1=front_right,
                                Blrvp_GP1_CgP1=back_left,
                                Brrvp_GP1_CgP1=back_right,
                                strength=strength,
                            )
                        )
                        if wake_age is not None:
                            next_wing.wake_ring_vortices[
                                chordwise_id, spanwise_id
                            ].age = wake_age

    def _get_body_state_arrays_for_step(
        self,
        step: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return per-body positions, velocities, angular rates, and GP-to-E
        rotations."""
        problem = self.multi_body_coupled_steady_problems[step]
        positions_E_E = np.vstack(problem.positions_E_E)
        velocities_E__E = np.zeros((self.num_bodies, 3), dtype=float)
        omegas_E__E = np.zeros((self.num_bodies, 3), dtype=float)
        R_pas_GP_to_Es = np.zeros((self.num_bodies, 3, 3), dtype=float)

        for body_index, coupled_operating_point in enumerate(
            problem.coupled_operating_points
        ):
            velocities_E__E[body_index] = coupled_operating_point.vCg_E__E
            omegas_E__E[body_index] = _transformations.apply_T_to_vectors(
                coupled_operating_point.T_pas_BP1_CgP1_to_E_CgP1,
                coupled_operating_point.omegas_BP1__E,
                has_point=False,
            )
            R_pas_GP_to_Es[body_index] = (
                coupled_operating_point.T_pas_GP1_CgP1_to_E_CgP1[:3, :3]
            )

        return positions_E_E, velocities_E__E, omegas_E__E, R_pas_GP_to_Es

    @staticmethod
    def _calculate_surface_apparent_velocities_from_body_states(
        points_E: np.ndarray,
        airplane_indices: np.ndarray,
        last_points_E: np.ndarray | None,
        current_positions_E_E: np.ndarray,
        current_velocities_E__E: np.ndarray,
        current_omegas_E__E: np.ndarray,
        current_R_pas_GP_to_Es: np.ndarray,
        last_positions_E_E: np.ndarray | None = None,
        last_R_pas_GP_to_Es: np.ndarray | None = None,
        delta_time: float | None = None,
    ) -> np.ndarray:
        """Return apparent velocities from current rigid motion plus internal motion."""
        apparent_velocities = np.zeros_like(points_E, dtype=float)
        include_internal_motion = last_points_E is not None

        if include_internal_motion and (
            last_positions_E_E is None
            or last_R_pas_GP_to_Es is None
            or delta_time is None
        ):
            raise ValueError(
                "last body states and delta_time are required when last_points_E is "
                "provided."
            )

        for row_index, airplane_index in enumerate(airplane_indices):
            current_position_E_E = current_positions_E_E[airplane_index]
            current_r_E = points_E[row_index] - current_position_E_E
            current_omega_E_rad = np.deg2rad(current_omegas_E__E[airplane_index])
            current_point_velocity_E = current_velocities_E__E[
                airplane_index
            ] + np.cross(current_omega_E_rad, current_r_E)
            apparent_velocities[row_index] = -current_point_velocity_E

            if include_internal_motion:
                assert last_positions_E_E is not None
                assert last_R_pas_GP_to_Es is not None
                assert delta_time is not None
                last_r_E = last_points_E[row_index] - last_positions_E_E[airplane_index]
                current_R_pas_GP_to_E = current_R_pas_GP_to_Es[airplane_index]
                last_R_pas_GP_to_E = last_R_pas_GP_to_Es[airplane_index]
                last_r_mapped_to_current_attitude_E = (
                    current_R_pas_GP_to_E @ last_R_pas_GP_to_E.T @ last_r_E
                )
                internal_point_velocity_E = (
                    current_r_E - last_r_mapped_to_current_attitude_E
                ) / delta_time
                apparent_velocities[row_index] -= internal_point_velocity_E

        return apparent_velocities

    def _calculate_rigid_body_apparent_velocities_at_points(
        self,
        points_E: np.ndarray,
        airplane_indices: np.ndarray,
    ) -> np.ndarray:
        """Return apparent fluid velocities at Earth-frame points on the bodies."""
        return self._calculate_surface_apparent_velocities_from_body_states(
            points_E=points_E,
            airplane_indices=airplane_indices,
            last_points_E=None,
            current_positions_E_E=self._current_body_positions_E_E,
            current_velocities_E__E=self._current_body_velocities_E__E,
            current_omegas_E__E=self._current_body_omegas_E__E,
            current_R_pas_GP_to_Es=self._current_body_R_pas_GP_to_Es,
        )

    def _calculate_surface_apparent_velocities_at_points(
        self,
        points_E: np.ndarray,
        airplane_indices: np.ndarray,
        last_points_E: np.ndarray,
    ) -> np.ndarray:
        """Return apparent velocities including prescribed internal body motion."""
        if self._current_step < 1:
            return self._calculate_rigid_body_apparent_velocities_at_points(
                points_E=points_E,
                airplane_indices=airplane_indices,
            )
        return self._calculate_surface_apparent_velocities_from_body_states(
            points_E=points_E,
            airplane_indices=airplane_indices,
            last_points_E=last_points_E,
            current_positions_E_E=self._current_body_positions_E_E,
            current_velocities_E__E=self._current_body_velocities_E__E,
            current_omegas_E__E=self._current_body_omegas_E__E,
            current_R_pas_GP_to_Es=self._current_body_R_pas_GP_to_Es,
            last_positions_E_E=self._last_body_positions_E_E,
            last_R_pas_GP_to_Es=self._last_body_R_pas_GP_to_Es,
            delta_time=self.delta_time,
        )

    @staticmethod
    def _extract_euler_angles_deg(R_pas_E_to_BP: np.ndarray) -> np.ndarray:
        """Extract intrinsic zyx Euler angles in degrees from a passive rotation."""

        def _wrap_angle_deg(angle_deg: float) -> float:
            """Wrap an angle into the validator's accepted interval (-180, 180]."""
            wrapped_angle_deg = ((angle_deg + 180.0) % 360.0) - 180.0
            if wrapped_angle_deg <= -180.0:
                wrapped_angle_deg += 360.0
            return float(wrapped_angle_deg)

        sin_angle_y = np.clip(-R_pas_E_to_BP[0, 2], -1.0, 1.0)
        angle_y = np.rad2deg(np.arcsin(sin_angle_y))
        if abs(sin_angle_y) > 0.99999:
            angle_x = 0.0
            angle_z = np.rad2deg(np.arctan2(-R_pas_E_to_BP[1, 0], R_pas_E_to_BP[1, 1]))
        else:
            angle_x = np.rad2deg(np.arctan2(R_pas_E_to_BP[1, 2], R_pas_E_to_BP[2, 2]))
            angle_z = np.rad2deg(np.arctan2(R_pas_E_to_BP[0, 1], R_pas_E_to_BP[0, 0]))
        angle_x = _wrap_angle_deg(angle_x)
        angle_y = _wrap_angle_deg(angle_y)
        angle_z = _wrap_angle_deg(angle_z)
        return np.array([angle_x, angle_y, angle_z], dtype=float)

    @staticmethod
    def _validate_shared_environment(
        coupled_operating_points: Sequence[operating_point.CoupledOperatingPoint],
    ) -> None:
        """Validate the current restricted shared-atmosphere multibody assumptions."""
        first_coupled_operating_point = coupled_operating_points[0]
        if (
            first_coupled_operating_point.surfaceNormal_E is not None
            or first_coupled_operating_point.surfacePoint_E_Eo is not None
        ):
            raise NotImplementedError(
                "MultiBodyCoupledUnsteadyRingVortexLatticeMethodSolver does not "
                "yet support image-surface modeling."
            )
        for coupled_operating_point in coupled_operating_points[1:]:
            if coupled_operating_point.rho != first_coupled_operating_point.rho:
                raise ValueError("All bodies must currently share the same rho.")
            if coupled_operating_point.nu != first_coupled_operating_point.nu:
                raise ValueError("All bodies must currently share the same nu.")
            if not np.allclose(
                coupled_operating_point.g_E, first_coupled_operating_point.g_E
            ):
                raise ValueError("All bodies must currently share the same gravity.")
            if (
                coupled_operating_point.surfaceNormal_E is not None
                or coupled_operating_point.surfacePoint_E_Eo is not None
            ):
                raise NotImplementedError(
                    "MultiBodyCoupledUnsteadyRingVortexLatticeMethodSolver does not "
                    "yet support image-surface modeling."
                )
