"""Contains classes for aerodynamic problems.

**Contains the following classes:**

SteadyProblem: A class used to contain steady aerodynamics problems.

UnsteadyProblem: A class used to contain unsteady aerodynamics problems.

CoupledSteadyProblem: A class used to contain steady aerodynamics problems that
characterize each time step of a coupled unsteady simulation.

CoupledUnsteadyProblem: A class used to contain unsteady aerodynamics problems that will
be used for coupled unsteady simulations.

**Contains the following functions:**

None
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

from . import (
    _mujoco_model,
    _parameter_validation,
    _transformations,
    geometry,
    movements,
)
from . import operating_point as operating_point_mod


class SteadyProblem:
    """A class used to contain steady aerodynamics problems.

    **Contains the following methods:**

    reynolds_numbers: A tuple of Reynolds numbers, one for each Airplane in the
    SteadyProblem.
    """

    def __init__(
        self,
        airplanes: list[geometry.airplane.Airplane],
        operating_point: operating_point_mod.OperatingPoint,
    ) -> None:
        """The initialization method.

        :param airplanes: The list of the Airplanes for this SteadyProblem.
        :param operating_point: The OperatingPoint for this SteadyProblem.
        :return: None
        """
        # Validate and store immutable attributes.
        if not isinstance(airplanes, list):
            raise TypeError("airplanes must be a list.")
        if len(airplanes) < 1:
            raise ValueError("airplanes must have at least one element.")
        for airplane in airplanes:
            if not isinstance(airplane, geometry.airplane.Airplane):
                raise TypeError("Every element in airplanes must be an Airplane.")
        # Store as tuple to prevent external mutation via .append(), .pop(), etc.
        self._airplanes: tuple[geometry.airplane.Airplane, ...] = tuple(airplanes)

        if not isinstance(operating_point, operating_point_mod.OperatingPoint):
            raise TypeError("operating_point must be an OperatingPoint.")
        self._operating_point = operating_point

        # Initialize the caches for the properties derived from the immutable
        # attributes.
        self._reynolds_numbers: tuple[float, ...] | None = None

        # Validate that the first Airplane has Cg_GP1_CgP1 set to zeros.
        self._airplanes[0].validate_first_airplane_constraints()

        # Populate GP1_CgP1 coordinates for all Airplanes' Panels. This finds the
        # Panels' positions in the first Airplane's geometry axes, relative to the
        # first Airplane's CG based on their locally defined positions.
        for airplane in self._airplanes:
            # Compute the passive transformation matrix from this Airplane's local
            # geometry axes, relative to its CG, to the first Airplane's geometry axes,
            # relative to the first Airplane's CG.
            T_pas_G_Cg_to_GP1_CgP1 = airplane.T_pas_G_Cg_to_GP1_CgP1

            for wing in airplane.wings:
                assert wing.panels is not None

                for panel in np.ravel(wing.panels):
                    panel.Frpp_GP1_CgP1 = _transformations.apply_T_to_vectors(
                        T_pas_G_Cg_to_GP1_CgP1, panel.Frpp_G_Cg, has_point=True
                    )
                    panel.Flpp_GP1_CgP1 = _transformations.apply_T_to_vectors(
                        T_pas_G_Cg_to_GP1_CgP1, panel.Flpp_G_Cg, has_point=True
                    )
                    panel.Blpp_GP1_CgP1 = _transformations.apply_T_to_vectors(
                        T_pas_G_Cg_to_GP1_CgP1, panel.Blpp_G_Cg, has_point=True
                    )
                    panel.Brpp_GP1_CgP1 = _transformations.apply_T_to_vectors(
                        T_pas_G_Cg_to_GP1_CgP1, panel.Brpp_G_Cg, has_point=True
                    )

    # --- Immutable: read only properties ---
    @property
    def airplanes(self) -> tuple[geometry.airplane.Airplane, ...]:
        return self._airplanes

    @property
    def operating_point(self) -> operating_point_mod.OperatingPoint:
        return self._operating_point

    # --- Immutable derived: manual lazy caching ---
    @property
    def reynolds_numbers(self) -> tuple[float, ...]:
        """A tuple of Reynolds numbers, one for each Airplane in the SteadyProblem.

        **Notes:**

        The Reynolds number is calculated as: Re = (V x L) / nu, where V is the
        freestream speed, observed from the Earth frame (vCg__E from OperatingPoint,
        m/s), L is the characteristic length (c_ref from Airplane, m), and nu is the
        kinematic viscosity (nu from OperatingPoint, m^2/s).

        These Reynolds numbers only consider the freestream speed, not any apparent
        velocity due to prescribed motion, so be careful interpreting it for cases where
        this SteadyProblem corresponds to one time step in an UnsteadyProblem.

        :return: A tuple of Reynolds numbers, one for each Airplane.
        """
        if self._reynolds_numbers is None:
            v = self._operating_point.vCg__E
            nu = self._operating_point.nu

            reynolds_list = []
            for airplane in self._airplanes:
                c_ref = airplane.c_ref
                assert c_ref is not None, "Airplane c_ref must be set to calculate Re"
                re = (v * c_ref) / nu
                reynolds_list.append(re)

            # Store as tuple to prevent external mutation.
            self._reynolds_numbers = tuple(reynolds_list)
        return self._reynolds_numbers


class UnsteadyProblem:
    """A class used to contain unsteady aerodynamics problems.

    **Contains the following methods:**

    None
    """

    def __init__(
        self,
        movement: movements.movement.Movement,
        only_final_results: bool | np.bool_ = False,
    ) -> None:
        """The initialization method.

        :param movement: The Movement that contains this UnsteadyProblem's
            OperatingPointMovement and AirplaneMovements.
        :param only_final_results: Determines whether the Solver will only calculate
            loads for the final time step (for static Movements) or (for non static
            Movements) for will only calculate loads for the time steps in the final
            complete motion cycle (of the Movement's sub Movement with the longest
            period), which increases simulation speed. Can be a bool or a numpy bool and
            will be converted internally to a bool. The default is False.
        :return: None
        """
        # Validate and store immutable attributes.
        if not isinstance(movement, movements.movement.Movement):
            raise TypeError("movement must be a Movement.")
        self._movement = movement
        self._only_final_results = _parameter_validation.boolLike_return_bool(
            only_final_results, "only_final_results"
        )

        self._num_steps: int = self._movement.num_steps
        self._delta_time: float = self._movement.delta_time
        self._max_wake_rows: int | None = self._movement.max_wake_rows

        # For UnsteadyProblems with a static Movement, we are typically interested in
        # the final time step's forces and moments, which, assuming convergence, will be
        # the most accurate. For UnsteadyProblems with cyclic movement, (e.g. flapping
        # wings) we are typically interested in the forces and moments averaged over the
        # last cycle simulated. Use the LCM of all motion periods to ensure we average
        # over a complete cycle of all motions.
        _movement_lcm_period = self._movement.lcm_period
        self._first_averaging_step: int
        if _movement_lcm_period == 0:
            self._first_averaging_step = self._num_steps - 1
        else:
            self._first_averaging_step = max(
                0,
                math.floor(self._num_steps - (_movement_lcm_period / self._delta_time)),
            )

        # If we only wants to calculate forces and moments for the final cycle (for a
        # cyclic Movement) or for the final time step (for a static Movement) set the
        # first step to calculate results to the first averaging step. Otherwise, set it
        # to the zero, which is the first time step.
        self._first_results_step: int
        if self._only_final_results:
            self._first_results_step = self._first_averaging_step
        else:
            self._first_results_step = 0

        # Initialize empty lists to hold the final loads and load coefficients each
        # Airplane experiences. These will only be populated if this UnsteadyProblem's
        # Movement is static. These are mutable and populated by the solver.
        self.finalForces_W: list[np.ndarray] = []
        self.finalForceCoefficients_W: list[np.ndarray] = []
        self.finalMoments_W_CgP1: list[np.ndarray] = []
        self.finalMomentCoefficients_W_CgP1: list[np.ndarray] = []

        # Initialize empty lists to hold the final cycle-averaged loads and load
        # coefficients each Airplane experiences. These will only be populated if this
        # UnsteadyProblem's Movement is cyclic. These are mutable and populated by the
        # solver.
        self.finalMeanForces_W: list[np.ndarray] = []
        self.finalMeanForceCoefficients_W: list[np.ndarray] = []
        self.finalMeanMoments_W_CgP1: list[np.ndarray] = []
        self.finalMeanMomentCoefficients_W_CgP1: list[np.ndarray] = []

        # Initialize empty lists to hold the final cycle-root-mean-squared loads and
        # load coefficients each airplane object experiences. These will only be
        # populated for variable geometry problems. These are mutable and populated by
        # the solver.
        self.finalRmsForces_W: list[np.ndarray] = []
        self.finalRmsForceCoefficients_W: list[np.ndarray] = []
        self.finalRmsMoments_W_CgP1: list[np.ndarray] = []
        self.finalRmsMomentCoefficients_W_CgP1: list[np.ndarray] = []

        # Initialize an empty list to hold the SteadyProblems as they are generated.
        steady_problems_temp: list[SteadyProblem] = []

        # Iterate through the UnsteadyProblem's time steps.
        for step_id in range(self._num_steps):

            # Get the Airplanes and the OperatingPoint associated with this time step.
            these_airplanes = []
            for this_base_airplane in movement.airplanes:
                these_airplanes.append(this_base_airplane[step_id])
            this_operating_point = movement.operating_points[step_id]

            # Initialize the SteadyProblem at this time step.
            this_steady_problem = SteadyProblem(
                airplanes=these_airplanes, operating_point=this_operating_point
            )

            # Append this SteadyProblem to the temporary list.
            steady_problems_temp.append(this_steady_problem)

        # Store as tuple to prevent external mutation via .append(), .pop(), etc.
        self._steady_problems: tuple[SteadyProblem, ...] = tuple(steady_problems_temp)

    # --- Immutable: read only properties ---
    @property
    def movement(self) -> movements.movement.Movement:
        return self._movement

    @property
    def only_final_results(self) -> bool:
        return self._only_final_results

    @property
    def num_steps(self) -> int:
        return self._num_steps

    @property
    def delta_time(self) -> float:
        return self._delta_time

    @property
    def first_averaging_step(self) -> int:
        return self._first_averaging_step

    @property
    def first_results_step(self) -> int:
        return self._first_results_step

    @property
    def max_wake_rows(self) -> int | None:
        return self._max_wake_rows

    @property
    def steady_problems(self) -> tuple[SteadyProblem, ...]:
        return self._steady_problems


class CoupledSteadyProblem:
    """A class used to contain steady aerodynamics problems that characterize each time
    step of a coupled unsteady simulation.

    **Contains the following methods:**

    None
    """

    def __init__(
        self,
        airplane: geometry.airplane.Airplane,
        coupled_operating_point: operating_point_mod.CoupledOperatingPoint,
    ) -> None:
        """The initialization method.

        :param airplane: The Airplane for this CoupledSteadyProblem.
        :param coupled_operating_point: The CoupledOperatingPoint for this
            CoupledSteadyProblem.
        :return: None
        """
        if not isinstance(airplane, geometry.airplane.Airplane):
            raise TypeError("airplane must be an Airplane.")
        self._airplane = airplane

        if not isinstance(
            coupled_operating_point, operating_point_mod.CoupledOperatingPoint
        ):
            raise TypeError("coupled_operating_point must be a CoupledOperatingPoint.")
        self._coupled_operating_point = coupled_operating_point

        # As CoupledSteadyProblems can only have one Airplane, they must have
        # Cg_GP1_CgP1 set to zeros.
        self._airplane.validate_first_airplane_constraints()

        # Populate the GP1_CgP1 coordinates for the Airplane's Panels.
        T_pas_G_Cg_to_GP1_CgP1 = airplane.T_pas_G_Cg_to_GP1_CgP1
        for wing in airplane.wings:
            _panels = wing.panels
            assert _panels is not None

            for panel in np.ravel(_panels):
                panel.Frpp_GP1_CgP1 = _transformations.apply_T_to_vectors(
                    T_pas_G_Cg_to_GP1_CgP1, panel.Frpp_G_Cg, has_point=True
                )
                panel.Flpp_GP1_CgP1 = _transformations.apply_T_to_vectors(
                    T_pas_G_Cg_to_GP1_CgP1, panel.Flpp_G_Cg, has_point=True
                )
                panel.Blpp_GP1_CgP1 = _transformations.apply_T_to_vectors(
                    T_pas_G_Cg_to_GP1_CgP1, panel.Blpp_G_Cg, has_point=True
                )
                panel.Brpp_GP1_CgP1 = _transformations.apply_T_to_vectors(
                    T_pas_G_Cg_to_GP1_CgP1, panel.Brpp_G_Cg, has_point=True
                )

    # --- Immutable: read only properties ---
    @property
    def airplane(self) -> geometry.airplane.Airplane:
        return self._airplane

    @property
    def coupled_operating_point(self) -> operating_point_mod.CoupledOperatingPoint:
        return self._coupled_operating_point


class MultiBodyCoupledSteadyProblem:
    """A steady multibody aerodynamic problem for one coupled time step.

    Each airplane carries its own CoupledOperatingPoint. The current implementation
    expresses all panel coordinates in a shared Earth-frame solve space so multiple
    freely moving rigid bodies can be solved together aerodynamically.
    """

    def __init__(
        self,
        airplanes: Sequence[geometry.airplane.Airplane],
        coupled_operating_points: Sequence[operating_point_mod.CoupledOperatingPoint],
        positions_E_E: Sequence[np.ndarray | Sequence[float | int]],
    ) -> None:
        """Initialize the multibody coupled steady problem."""
        if not isinstance(airplanes, Sequence):
            raise TypeError("airplanes must be a sequence.")
        if not isinstance(coupled_operating_points, Sequence):
            raise TypeError("coupled_operating_points must be a sequence.")
        if not isinstance(positions_E_E, Sequence):
            raise TypeError("positions_E_E must be a sequence.")
        if len(airplanes) < 2:
            raise ValueError("airplanes must contain at least two elements.")
        if not (len(airplanes) == len(coupled_operating_points) == len(positions_E_E)):
            raise ValueError(
                "airplanes, coupled_operating_points, and positions_E_E must have the "
                "same length."
            )

        validated_airplanes: list[geometry.airplane.Airplane] = []
        validated_coupled_operating_points: list[
            operating_point_mod.CoupledOperatingPoint
        ] = []
        validated_positions_E_E: list[np.ndarray] = []
        for body_index, (airplane, coupled_operating_point, position_E_E) in enumerate(
            zip(airplanes, coupled_operating_points, positions_E_E, strict=True)
        ):
            if not isinstance(airplane, geometry.airplane.Airplane):
                raise TypeError("Every element in airplanes must be an Airplane.")
            if not isinstance(
                coupled_operating_point, operating_point_mod.CoupledOperatingPoint
            ):
                raise TypeError(
                    "Every element in coupled_operating_points must be a "
                    "CoupledOperatingPoint."
                )
            validated_airplanes.append(airplane)
            validated_coupled_operating_points.append(coupled_operating_point)
            validated_positions_E_E.append(
                _parameter_validation.threeD_number_vectorLike_return_float(
                    position_E_E, f"positions_E_E[{body_index}]"
                )
            )

        self._airplanes = tuple(validated_airplanes)
        self._coupled_operating_points = tuple(validated_coupled_operating_points)
        self._positions_E_E = tuple(validated_positions_E_E)

        for airplane, coupled_operating_point, body_position_E_E in zip(
            self._airplanes,
            self._coupled_operating_points,
            self._positions_E_E,
            strict=True,
        ):
            T_pas_G_Cg_to_E_Cg = coupled_operating_point.T_pas_GP1_CgP1_to_E_CgP1

            for wing in airplane.wings:
                _panels = wing.panels
                assert _panels is not None

                for panel in np.ravel(_panels):
                    panel.Frpp_GP1_CgP1 = (
                        _transformations.apply_T_to_vectors(
                            T_pas_G_Cg_to_E_Cg, panel.Frpp_G_Cg, has_point=True
                        )
                        + body_position_E_E
                    )
                    panel.Flpp_GP1_CgP1 = (
                        _transformations.apply_T_to_vectors(
                            T_pas_G_Cg_to_E_Cg, panel.Flpp_G_Cg, has_point=True
                        )
                        + body_position_E_E
                    )
                    panel.Blpp_GP1_CgP1 = (
                        _transformations.apply_T_to_vectors(
                            T_pas_G_Cg_to_E_Cg, panel.Blpp_G_Cg, has_point=True
                        )
                        + body_position_E_E
                    )
                    panel.Brpp_GP1_CgP1 = (
                        _transformations.apply_T_to_vectors(
                            T_pas_G_Cg_to_E_Cg, panel.Brpp_G_Cg, has_point=True
                        )
                        + body_position_E_E
                    )

    @property
    def airplanes(self) -> tuple[geometry.airplane.Airplane, ...]:
        return self._airplanes

    @property
    def coupled_operating_points(
        self,
    ) -> tuple[operating_point_mod.CoupledOperatingPoint, ...]:
        return self._coupled_operating_points

    @property
    def positions_E_E(self) -> tuple[np.ndarray, ...]:
        return self._positions_E_E


class CoupledUnsteadyProblem:
    """A class used to contain unsteady aerodynamics problems that will be used for
    coupled unsteady simulations.

    **Contains the following methods:**

    None
    """

    def __init__(
        self,
        coupled_movement: movements.movement.CoupledMovement,
        I_BP1_CgP1: np.ndarray | Sequence[Sequence[float | int]],
        external_forces_fn: (
            Callable[
                [
                    operating_point_mod.CoupledOperatingPoint,
                    geometry.airplane.Airplane,
                ],
                tuple[np.ndarray, np.ndarray],
            ]
            | None
        ) = None,
        extra_xml: dict[str, str] | None = None,
        mujoco_assets: dict[str, bytes] | None = None,
    ) -> None:
        """The initialization method.

        :param coupled_movement: The CoupledMovement that contains this
            CoupledUnsteadyProblem's CoupledOperatingPoints and AirplaneMovements.
        :param I_BP1_CgP1: An array-like object of numbers (ints or floats) with shape
            (3,3) for the inertia matrix of the airplane represented by
            coupled_movement's AirplaneMovement. It is in the first Airplane's body
            axes, relative to the first Airplane's CG. It can be a tuple, list, or
            ndarray. Values will be converted internally to floats. Its units are in
            kilogram square meters.
        :param external_forces_fn: A callable that computes additional forces and
            moments to apply to the Airplane during the coupled simulation. It takes a
            CoupledOperatingPoint and an Airplane and returns a tuple of two (3,)
            ndarrays of floats: the additional force (in wind axes, in Newtons) and the
            additional moment (in wind axes, relative to the first Airplane's CG, in
            Newton meters). Setting this to None applies no additional forces. The
            default is None.
        :param extra_xml: A dict mapping injection point names to XML fragment strings
            to inject into the MuJoCo model's XML. Supported keys are "default",
            "asset", "visual", "worldbody", and "body". Setting this to None injects no
            extra XML. The default is None.
        :param mujoco_assets: A dict mapping virtual filenames to their binary contents
            for the MuJoCo model. Setting this to None provides no extra assets. The
            default is None.
        :return: None
        """
        if not isinstance(coupled_movement, movements.movement.CoupledMovement):
            raise TypeError("coupled_movement must be a CoupledMovement.")
        self._coupled_movement = coupled_movement

        I_BP1_CgP1 = _parameter_validation.m_by_n_number_arrayLike_return_float(
            I_BP1_CgP1, "I_BP1_CgP1", 3, 3
        )
        if not np.allclose(I_BP1_CgP1, I_BP1_CgP1.T):
            raise ValueError("I_BP1_CgP1 must be symmetric.")
        self._I_BP1_CgP1 = I_BP1_CgP1
        self._I_BP1_CgP1.flags.writeable = False

        if external_forces_fn is not None and not callable(external_forces_fn):
            raise TypeError("external_forces_fn must be callable or None.")
        self._external_forces_fn = external_forces_fn

        self._num_steps: int = self._coupled_movement.num_steps
        self._delta_time: float = self._coupled_movement.delta_time

        # Initialize empty lists to hold the loads and load coefficients experienced by
        # each time step's Airplane.
        self.forces_W: list[np.ndarray] = []
        self.forceCoefficients_W: list[np.ndarray] = []
        self.moments_W_Cg: list[np.ndarray] = []
        self.momentCoefficients_W_Cg: list[np.ndarray] = []

        # Get the tuple representing the Airplane at each time step.
        self._airplanes = self._coupled_movement.airplanes

        # Initialize a list with the first time step's CoupledSteadyProblem. The
        # CoupledUnsteadyRingVortexLatticeMethodSolver will append each subsequent time
        # step's CoupledSteadyProblem to this list.
        self.coupled_steady_problems = [
            CoupledSteadyProblem(
                airplane=self._airplanes[0],
                coupled_operating_point=self._coupled_movement.coupled_operating_points[
                    0
                ],
            )
        ]

        self._mujoco_model = _mujoco_model.MuJoCoModel(
            coupled_movement=self._coupled_movement,
            I_BP1_CgP1=self._I_BP1_CgP1,
            extra_xml=extra_xml,
            mujoco_assets=mujoco_assets,
        )

    # --- Immutable: read only properties ---
    @property
    def coupled_movement(self) -> movements.movement.CoupledMovement:
        return self._coupled_movement

    @property
    def I_BP1_CgP1(self) -> np.ndarray:
        return self._I_BP1_CgP1

    @property
    def external_forces_fn(
        self,
    ) -> (
        Callable[
            [
                operating_point_mod.CoupledOperatingPoint,
                geometry.airplane.Airplane,
            ],
            tuple[np.ndarray, np.ndarray],
        ]
        | None
    ):
        return self._external_forces_fn

    @property
    def num_steps(self) -> int:
        return self._num_steps

    @property
    def delta_time(self) -> float:
        return self._delta_time

    @property
    def airplanes(self) -> tuple[geometry.airplane.Airplane, ...]:
        return self._airplanes

    @property
    def mujoco_model(self) -> _mujoco_model.MuJoCoModel:
        return self._mujoco_model


class MultiBodyCoupledUnsteadyProblem:
    """An unsteady coupled free-flight problem for multiple rigid bodies.

    The current implementation is intentionally scoped to the first practical multibody
    milestone: multiple static-geometry gliding wings coupled to MuJoCo.
    """

    def __init__(
        self,
        coupled_movement: movements.movement.MultiBodyCoupledMovement,
        I_BP1_CgP1s: Sequence[np.ndarray | Sequence[Sequence[float | int]]],
        external_forces_fn: (
            Callable[
                [
                    int,
                    operating_point_mod.CoupledOperatingPoint,
                    geometry.airplane.Airplane,
                ],
                tuple[np.ndarray, np.ndarray],
            ]
            | None
        ) = None,
        extra_xml: dict[str, str] | None = None,
        mujoco_assets: dict[str, bytes] | None = None,
    ) -> None:
        """Initialize the multibody coupled unsteady problem."""
        if not isinstance(
            coupled_movement, movements.movement.MultiBodyCoupledMovement
        ):
            raise TypeError("coupled_movement must be a MultiBodyCoupledMovement.")
        self._coupled_movement = coupled_movement

        if not isinstance(I_BP1_CgP1s, Sequence):
            raise TypeError("I_BP1_CgP1s must be a sequence of inertia matrices.")
        if len(I_BP1_CgP1s) != len(self._coupled_movement.airplanes[0]):
            raise ValueError("I_BP1_CgP1s must have one inertia matrix per rigid body.")

        validated_inertias: list[np.ndarray] = []
        for body_index, inertia_matrix in enumerate(I_BP1_CgP1s):
            validated_inertia = (
                _parameter_validation.m_by_n_number_arrayLike_return_float(
                    inertia_matrix,
                    f"I_BP1_CgP1s[{body_index}]",
                    3,
                    3,
                )
            )
            if not np.allclose(validated_inertia, validated_inertia.T):
                raise ValueError(f"I_BP1_CgP1s[{body_index}] must be symmetric.")
            validated_inertia.flags.writeable = False
            validated_inertias.append(validated_inertia)
        self._I_BP1_CgP1s = tuple(validated_inertias)

        if external_forces_fn is not None and not callable(external_forces_fn):
            raise TypeError("external_forces_fn must be callable or None.")
        self._external_forces_fn = external_forces_fn

        self._num_steps = self._coupled_movement.num_steps
        self._delta_time = self._coupled_movement.delta_time

        self.forces_W: list[np.ndarray] = []
        self.forceCoefficients_W: list[np.ndarray] = []
        self.moments_W_Cg: list[np.ndarray] = []
        self.momentCoefficients_W_Cg: list[np.ndarray] = []

        self._airplanes = self._coupled_movement.airplanes

        initial_operating_points = self._coupled_movement.coupled_operating_points[0]
        initial_positions_E_E = self._coupled_movement.positions_E_E[0]
        self.multi_body_coupled_steady_problems = [
            MultiBodyCoupledSteadyProblem(
                airplanes=self._airplanes[0],
                coupled_operating_points=initial_operating_points,
                positions_E_E=initial_positions_E_E,
            )
        ]

        self._mujoco_model = _mujoco_model.MultiBodyMuJoCoModel(
            airplanes=self._airplanes[0],
            coupled_operating_points=initial_operating_points,
            I_BP1_CgP1s=self._I_BP1_CgP1s,
            initial_positions_E_E=initial_positions_E_E,
            delta_time=self._delta_time,
            extra_xml=extra_xml,
            mujoco_assets=mujoco_assets,
        )

    @property
    def coupled_movement(self) -> movements.movement.MultiBodyCoupledMovement:
        return self._coupled_movement

    @property
    def I_BP1_CgP1s(self) -> tuple[np.ndarray, ...]:
        return self._I_BP1_CgP1s

    @property
    def external_forces_fn(
        self,
    ) -> (
        Callable[
            [
                int,
                operating_point_mod.CoupledOperatingPoint,
                geometry.airplane.Airplane,
            ],
            tuple[np.ndarray, np.ndarray],
        ]
        | None
    ):
        return self._external_forces_fn

    @property
    def num_steps(self) -> int:
        return self._num_steps

    @property
    def delta_time(self) -> float:
        return self._delta_time

    @property
    def airplanes(self) -> tuple[tuple[geometry.airplane.Airplane, ...], ...]:
        return self._airplanes

    @property
    def mujoco_model(self) -> _mujoco_model.MultiBodyMuJoCoModel:
        return self._mujoco_model
