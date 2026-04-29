"""Contains class definitions for interfacing with MuJoCo for free-flight
simulations."""

from __future__ import annotations

from collections.abc import Sequence

import mujoco
import numpy as np

from . import (
    _parameter_validation,
    _transformations,
    geometry,
    movements,
    operating_point,
)


def _make_unique_mujoco_names(base_names: Sequence[str]) -> tuple[str, ...]:
    """Return MuJoCo-safe unique names while preserving input order.

    MuJoCo requires unique names for bodies and joints. This helper appends a numeric
    suffix to repeated names while leaving the first occurrence unchanged.
    """
    counts: dict[str, int] = {}
    unique_names: list[str] = []
    for base_name in base_names:
        count = counts.get(base_name, 0)
        counts[base_name] = count + 1
        if count == 0:
            unique_names.append(base_name)
        else:
            unique_names.append(f"{base_name}_{count + 1}")
    return tuple(unique_names)


def _body_state_from_airplane_and_operating_point(
    airplane: geometry.airplane.Airplane,
    coupled_operating_point: operating_point.CoupledOperatingPoint,
    inertia_BP1_CgP1: np.ndarray,
    initial_position_E_E: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Build validated MuJoCo body state data from codebase objects."""
    if not isinstance(airplane, geometry.airplane.Airplane):
        raise TypeError("airplane must be an Airplane.")
    if not isinstance(coupled_operating_point, operating_point.CoupledOperatingPoint):
        raise TypeError("coupled_operating_point must be a CoupledOperatingPoint.")

    inertia_BP1_CgP1 = _parameter_validation.m_by_n_number_arrayLike_return_float(
        inertia_BP1_CgP1, "inertia_BP1_CgP1", 3, 3
    )
    initial_position_E_E = _parameter_validation.threeD_number_vectorLike_return_float(
        initial_position_E_E, "initial_position_E_E"
    )

    g_E = coupled_operating_point.g_E
    g_norm = np.linalg.norm(g_E)
    if g_norm == 0.0:
        raise ValueError("g_E must have non-zero magnitude.")

    mass = airplane.weight / g_norm
    if mass <= 0.0:
        raise ValueError("Each airplane must have positive weight for MuJoCo mass.")

    omegas_rad_BP1__E = np.deg2rad(coupled_operating_point.omegas_BP1__E)
    T_pas_BP1_CgP1_to_E_CgP1 = coupled_operating_point.T_pas_BP1_CgP1_to_E_CgP1
    vCg_E__E = coupled_operating_point.vCg_E__E

    R_pas_BP1_to_E = T_pas_BP1_CgP1_to_E_CgP1[:3, :3]
    R_act_BP1_to_E = np.linalg.inv(R_pas_BP1_to_E)
    R_act_E_to_BP1 = R_act_BP1_to_E.T
    quat_act_E_to_BP1_wxyz = _transformations.R_to_quat_wxyz(R_act_E_to_BP1)

    IXX_BP1_CgP1, IXY_BP1_CgP1, IXZ_BP1_CgP1 = inertia_BP1_CgP1[0]
    IYX_BP1_CgP1, IYY_BP1_CgP1, IYZ_BP1_CgP1 = inertia_BP1_CgP1[1]
    IZX_BP1_CgP1, IZY_BP1_CgP1, IZZ_BP1_CgP1 = inertia_BP1_CgP1[2]

    full_inertia_BP1_CgP1 = np.array(
        [
            IXX_BP1_CgP1,
            IYY_BP1_CgP1,
            IZZ_BP1_CgP1,
            (IXY_BP1_CgP1 + IYX_BP1_CgP1) / 2,
            (IXZ_BP1_CgP1 + IZX_BP1_CgP1) / 2,
            (IYZ_BP1_CgP1 + IZY_BP1_CgP1) / 2,
        ],
        dtype=float,
    )

    qpos = np.hstack([initial_position_E_E, quat_act_E_to_BP1_wxyz]).astype(float)
    qvel = np.hstack([vCg_E__E, omegas_rad_BP1__E]).astype(float)

    return {
        "mass": float(mass),
        "full_inertia_BP1_CgP1": full_inertia_BP1_CgP1,
        "qpos": qpos,
        "qvel": qvel,
    }


# TEST: Add unit tests for this class's initialization.
class MuJoCoModel:
    """A class that wraps MuJoCo models and data objects to provide a clean interface
    for free flight simulations.

    Provides methods for applying aerodynamic loads to the first Airplane, advancing the
    MuJoCo simulation, and extracting the current state of the first Airplane.

    **Contains the following methods:**

    apply_loads: Applies loads to the model.

    step: Advances the MuJoCo simulation by one time step.

    get_state: Extracts the current position, orientation, velocity, and angular
    velocity from the model.

    reset: Resets the model's state to the initial conditions, time to zero seconds, and
    removes any applied loads.
    """

    def __init__(
        self,
        coupled_movement: movements.movement.CoupledMovement,
        I_BP1_CgP1: np.ndarray,
        extra_xml: dict[str, str] | None = None,
        mujoco_assets: dict[str, bytes] | None = None,
    ) -> None:
        """The initialization method.

        :param coupled_movement: The CoupledMovement this model is associated with.
        :param I_BP1_CgP1: A (3,3) ndarray of floats representing the inertia matrix of
            the airplane represented by coupled_movement's AirplaneMovement. It is in
            the first Airplane's body axes, relative to the first Airplane's CG.
        :param extra_xml: A dict mapping injection point names to XML fragment strings
            to inject into the generated MuJoCo XML. Supported keys are "default",
            "asset", and "visual" (inserted as top level elements), "worldbody"
            (inserted inside the worldbody element, before the body), and "body"
            (inserted inside the body element, after the inertial element). The default
            is None, which injects no extra XML.
        :param mujoco_assets: A dict mapping virtual filenames to their binary contents.
            These are passed to MuJoCo's from_xml_string as the assets parameter,
            allowing meshes and other binary files to be loaded without writing to disk.
            The default is None, which provides no extra assets.
        :return: None
        """
        start_key_frame_name: str = "start"

        initial_airplane = coupled_movement.airplanes[0]
        initial_coupled_operating_point = coupled_movement.coupled_operating_points[0]
        delta_time = coupled_movement.delta_time

        name = initial_airplane.name
        weight = initial_airplane.weight
        omegasRad_BP1__E = np.deg2rad(initial_coupled_operating_point.omegas_BP1__E)
        g_E = initial_coupled_operating_point.g_E
        T_pas_BP1_CgP1_to_E_CgP1 = (
            initial_coupled_operating_point.T_pas_BP1_CgP1_to_E_CgP1
        )
        vCg_E__E = initial_coupled_operating_point.vCg_E__E

        mass = weight / np.linalg.norm(g_E)
        omegaXRad_BP1__E, omegaYRad_BP1__E, omegaZRad_BP1__E = omegasRad_BP1__E[:]

        R_pas_BP1_to_E = T_pas_BP1_CgP1_to_E_CgP1[:3, :3]

        R_act_BP1_to_E = np.linalg.inv(R_pas_BP1_to_E)

        R_act_E_to_BP1 = R_act_BP1_to_E.T

        # REFACTOR: Add section on quaternions to ANGLES_VECTORS_AND_TRANSFORMATIONS.md.
        quat_act_E_to_BP1_wxyz = _transformations.R_to_quat_wxyz(R_act_E_to_BP1)

        IXX_BP1_CgP1, IXY_BP1_CgP1, IXZ_BP1_CgP1 = I_BP1_CgP1[0]
        IYX_BP1_CgP1, IYY_BP1_CgP1, IYZ_BP1_CgP1 = I_BP1_CgP1[1]
        IZX_BP1_CgP1, IZY_BP1_CgP1, IZZ_BP1_CgP1 = I_BP1_CgP1[2]

        IXY_BP1_CgP1 = (IXY_BP1_CgP1 + IYX_BP1_CgP1) / 2
        IXZ_BP1_CgP1 = (IXZ_BP1_CgP1 + IZX_BP1_CgP1) / 2
        IYZ_BP1_CgP1 = (IYZ_BP1_CgP1 + IZY_BP1_CgP1) / 2

        (
            quatW_act_E_to_BP1,
            quatX_act_E_to_BP1,
            quatY_act_E_to_BP1,
            quatZ_act_E_to_BP1,
        ) = quat_act_E_to_BP1_wxyz[:]

        vCgX_E__E, vCgY_E__E, vCgZ_E__E = vCg_E__E[:]

        # Gravity in the MuJoCo model is turned off as it is applied by the
        # CoupledUnsteadyRingVortexLatticeMethodSolver.
        gravity_str = f"0.0 0.0 0.0"
        inertia_str = (
            f"{IXX_BP1_CgP1} {IYY_BP1_CgP1} {IZZ_BP1_CgP1} {IXY_BP1_CgP1} "
            f"{IXZ_BP1_CgP1} {IYZ_BP1_CgP1}"
        )
        qpos_str = (
            f"0.0 0.0 0.0 {quatW_act_E_to_BP1} {quatX_act_E_to_BP1} {quatY_act_E_to_BP1} "
            f"{quatZ_act_E_to_BP1}"
        )
        qvel_str = (
            f"{vCgX_E__E} {vCgY_E__E} {vCgZ_E__E} {omegaXRad_BP1__E} "
            f"{omegaYRad_BP1__E} {omegaZRad_BP1__E}"
        )

        # Build the extra XML fragments to inject. If extra_xml is None, use an empty
        # dict so the .get calls below return empty strings.
        _extra = extra_xml if extra_xml is not None else {}
        extra_default = _extra.get("default", "")
        extra_asset = _extra.get("asset", "")
        extra_visual = _extra.get("visual", "")
        extra_worldbody = _extra.get("worldbody", "")
        extra_body = _extra.get("body", "")

        self.xml_str = f"""
        <mujoco model="{name}">
          <option timestep="{delta_time}" integrator="RK4" gravity="{gravity_str}"/>

          {extra_default}
          {extra_asset}
          {extra_visual}

          <worldbody>
            {extra_worldbody}
            <body name="{name}" pos="0.0 0.0 0.0" >
              <freejoint/>
              <inertial pos="0.0 0.0 0.0" mass="{mass}" fullinertia="{inertia_str}"/>
              {extra_body}
            </body>
          </worldbody>

          <keyframe>
            <key name="{start_key_frame_name}" qpos="{qpos_str}" qvel="{qvel_str}"/>
          </keyframe>
        </mujoco>
        """

        # Create the internal MuJoCo model object from the XML str. If mujoco_assets
        # is provided, pass it so MuJoCo can resolve virtual filenames (e.g., STL
        # meshes) without writing them to disk.
        # noinspection PyArgumentList
        if mujoco_assets is not None:
            self.model = mujoco.MjModel.from_xml_string(
                self.xml_str, assets=mujoco_assets
            )
        else:
            self.model = mujoco.MjModel.from_xml_string(self.xml_str)

        # Set the internal model's time step to be the same as CoupledMovement's.
        self.model.opt.timestep = delta_time

        # Create the MuJoCo data structure.
        self.data: mujoco.MjData = mujoco.MjData(self.model)

        # Get and store the body ID and the initial conditions key frame ID.
        self.body_id: int = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, name
        )
        self.initial_key_frame_id: int = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, start_key_frame_name
        )

        # Set the internal model's state to the initial conditions.
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.initial_key_frame_id)

        # Run forward kinematics to compute derived quantities (xmat, xpos, etc.)
        # from the initial qpos/qvel. Without this, xmat would be zeros until the
        # first call to mj_step.
        mujoco.mj_forward(self.model, self.data)

        # Store initial state for reset functionality.
        self._initial_qpos: np.ndarray = np.copy(self.data.qpos)
        self._initial_qvel: np.ndarray = np.copy(self.data.qvel)

    # TEST: Add unit tests for this method.
    def apply_loads(
        self,
        forces_E: np.ndarray | Sequence[float | int],
        moments_E_CgP1: np.ndarray | Sequence[float | int],
    ) -> None:
        """Applies loads to the model.

        **Notes:**

        xfrc_applied[0:3] = forces_E: The current force applied to the first Airplane's
        CG (in Earth axes) in Newtons.

        xfrc_applied[3:6] = moments_E_CgP1: The current moment applied to the first
        Airplane's CG (in Earth axes, relative to the first Airplane's CG) in Newton
        meters.

        The loads will persist until the next call to apply_loads or until they are
        explicitly cleared.

        :param forces_E: A (3,) array-like object of numbers (int or float) representing
            the forces (in Earth axes) to apply to the first Airplane at the first
            Airplane's CG. Can be a tuple, list, or ndarray. Values are converted to
            floats internally. The units are in Newtons.
        :param moments_E_CgP1: A (3,) array-like object of numbers (int or float)
            representing the moments (in Earth axes, relative to the first Airplane's
            CG) to apply to the first Airplane at the first Airplane's CG. Can be a
            tuple, list, or ndarray. Values are converted to floats internally. The
            units are in Newton meters.
        :return: None
        """
        forces_E = _parameter_validation.threeD_number_vectorLike_return_float(
            forces_E, "forces_E"
        )
        moments_E_CgP1 = _parameter_validation.threeD_number_vectorLike_return_float(
            moments_E_CgP1, "moments_E_CgP1"
        )

        # Pack the force and moment into the model's 6-element xfrc_applied array.
        self.data.xfrc_applied[self.body_id][:] = np.hstack([forces_E, moments_E_CgP1])

    # TEST: Add unit tests for this method.
    def step(self) -> None:
        """Advances the MuJoCo simulation by one time step.

        Steps the equations of motion forward in time by one time step, taking into
        account all forces, moments, contacts, and constraints in the model.

        :return: None
        """
        mujoco.mj_step(self.model, self.data)

    # TEST: Add unit tests for this method.
    def get_state(self) -> dict[str, np.ndarray | float]:
        """Extracts the current position, orientation, velocity, and angular velocity of
        the model.

        **Notes:**

        qpos[0:3] = position_E_E: The current position of the first Airplane's CG (in
        Earth axes, relative to the Earth origin) in meters.

        qvel[0:3] = velocity_E__E: The current velocity of the first Airplane's CG (in
        Earth axes, observed from the Earth frame) in meters per second.

        np.rad2deg(qvel[3:6]) = omegas_BP1__E: The current angular velocity of the first
        Airplane's body axes (in the first Airplane's body axes, observed from the Earth
        frame) in degrees per second.

        xmat = R_pas_BP1_to_E: The current orientation of the first Airplane as a
        passive rotation matrix from the first Airplane's body axes to Earth axes.

        We define MuJoCo world coordinates to be identical to Ptera Software Earth axes.

        :return: A dictionary containing the following keys: ``position_E_E``, a (3,)
            ndarray of floats representing the current position of the first Airplane's
            CG (in Earth axes, relative to the Earth origin) in meters;
            ``R_pas_E_to_BP1``, a (3,3) ndarray of floats representing the current
            orientation of the first Airplane as a passive rotation matrix from Earth
            axes to first Airplane's body axes; ``velocity_E__E``, a (3,) ndarray of
            floats representing the current velocity of the first Airplane's CG (in
            Earth axes, observed from the Earth frame) in meters per second;
            ``omegas_BP1__E``, a (3,) ndarray of floats representing the current angular
            velocity of the first Airplane's body axes (in the first Airplane's body
            axes, observed from the Earth frame) in degrees per second; ``time``, a
            float representing the current simulation time in seconds.
        """
        # MuJoCo's xmat is R_pas_BP1_to_E: it transforms vectors from the first
        # Airplane's body axes to Earth axes. To get R_pas_E_to_BP1, we take the
        # transpose.
        R_pas_BP1_to_E = self.data.xmat[self.body_id].reshape(3, 3)
        # REFACTOR: Consider creating an invert_R_pas function in _transformations.py
        #  and calling it here.
        R_pas_E_to_BP1 = R_pas_BP1_to_E.T

        return {
            "position_E_E": np.copy(self.data.qpos[0:3]),
            "R_pas_E_to_BP1": np.copy(R_pas_E_to_BP1),
            "velocity_E__E": np.copy(self.data.qvel[0:3]),
            "omegas_BP1__E": np.rad2deg(np.copy(self.data.qvel[3:6])),
            "time": float(self.data.time),
        }

    # TEST: Add unit tests for this method.
    def reset(self) -> None:
        """Resets the model's state to the initial conditions, time to zero seconds, and
        removes any applied loads.

        :return: None
        """
        # Reset the model's state to the initial conditions.
        self.data.qpos[:] = self._initial_qpos
        self.data.qvel[:] = self._initial_qvel

        # Reset time to zero seconds.
        self.data.time = 0.0

        # Remove any applied loads.
        self.data.xfrc_applied[:] = 0.0

        # Run forward kinematics to update dependent quantities.
        mujoco.mj_forward(self.model, self.data)


class MultiBodyMuJoCoModel:
    """A MuJoCo wrapper for multiple rigid bodies in free flight.

    This class is the first true multibody MuJoCo wrapper in the codebase. It builds one
    MuJoCo body and one ``freejoint`` per airplane, stores per-body state/load
    addressing information, and exposes vectorized load/state methods with one row per
    body.
    """

    def __init__(
        self,
        airplanes: Sequence[geometry.airplane.Airplane],
        coupled_operating_points: Sequence[operating_point.CoupledOperatingPoint],
        I_BP1_CgP1s: Sequence[np.ndarray | Sequence[Sequence[float | int]]],
        initial_positions_E_E: Sequence[np.ndarray | Sequence[float | int]],
        delta_time: float | int,
        extra_xml: dict[str, str] | None = None,
        mujoco_assets: dict[str, bytes] | None = None,
    ) -> None:
        """Initialize the multibody MuJoCo model.

        :param airplanes: The airplanes whose rigid-body masses/names are used.
        :param coupled_operating_points: One coupled operating point per body.
        :param I_BP1_CgP1s: One body-axis inertia matrix per body.
        :param initial_positions_E_E: One initial Earth-frame position per body.
        :param delta_time: Simulation time step in seconds.
        :param extra_xml: Optional XML fragments to inject, using the same keys as the
            single-body wrapper.
        :param mujoco_assets: Optional in-memory MuJoCo assets.
        :return: None
        """
        if not isinstance(airplanes, Sequence):
            raise TypeError("airplanes must be a sequence of Airplanes.")
        if not isinstance(coupled_operating_points, Sequence):
            raise TypeError(
                "coupled_operating_points must be a sequence of CoupledOperatingPoints."
            )
        if not isinstance(I_BP1_CgP1s, Sequence):
            raise TypeError("I_BP1_CgP1s must be a sequence of inertia matrices.")
        if not isinstance(initial_positions_E_E, Sequence):
            raise TypeError(
                "initial_positions_E_E must be a sequence of 3D position vectors."
            )

        if len(airplanes) < 2:
            raise ValueError(
                "MultiBodyMuJoCoModel requires at least two bodies for multibody use."
            )
        if not (
            len(airplanes)
            == len(coupled_operating_points)
            == len(I_BP1_CgP1s)
            == len(initial_positions_E_E)
        ):
            raise ValueError(
                "airplanes, coupled_operating_points, I_BP1_CgP1s, and "
                "initial_positions_E_E must all have the same length."
            )

        delta_time = _parameter_validation.number_in_range_return_float(
            delta_time, "delta_time", min_val=0.0, min_inclusive=False
        )

        self.num_bodies = len(airplanes)
        self.body_names = _make_unique_mujoco_names(
            [airplane.name for airplane in airplanes]
        )
        self.joint_names = tuple(
            f"{body_name}_freejoint" for body_name in self.body_names
        )
        _extra = extra_xml if extra_xml is not None else {}
        extra_default = _extra.get("default", "")
        extra_asset = _extra.get("asset", "")
        extra_visual = _extra.get("visual", "")
        extra_worldbody = _extra.get("worldbody", "")
        extra_body = _extra.get("body", "")

        body_state_specs = [
            _body_state_from_airplane_and_operating_point(
                airplane=airplane,
                coupled_operating_point=coupled_operating_point,
                inertia_BP1_CgP1=np.array(inertia_BP1_CgP1, dtype=float, copy=True),
                initial_position_E_E=np.array(
                    initial_position_E_E, dtype=float, copy=True
                ),
            )
            for airplane, coupled_operating_point, inertia_BP1_CgP1, initial_position_E_E in zip(
                airplanes,
                coupled_operating_points,
                I_BP1_CgP1s,
                initial_positions_E_E,
                strict=True,
            )
        ]

        body_xml_fragments: list[str] = []
        for body_name, joint_name, body_state_spec in zip(
            self.body_names,
            self.joint_names,
            body_state_specs,
            strict=True,
        ):
            full_inertia = body_state_spec["full_inertia_BP1_CgP1"]
            assert isinstance(full_inertia, np.ndarray)
            inertia_str = " ".join(str(float(value)) for value in full_inertia)
            body_xml_fragments.append(
                (
                    f'<body name="{body_name}" pos="0.0 0.0 0.0">\n'
                    f'  <freejoint name="{joint_name}"/>\n'
                    f'  <inertial pos="0.0 0.0 0.0" mass="{float(body_state_spec["mass"])}" '
                    f'fullinertia="{inertia_str}"/>\n'
                    f"  {extra_body}\n"
                    f"</body>"
                )
            )

        qpos_str = " ".join(
            str(float(value))
            for body_state_spec in body_state_specs
            for value in np.asarray(body_state_spec["qpos"], dtype=float)
        )
        qvel_str = " ".join(
            str(float(value))
            for body_state_spec in body_state_specs
            for value in np.asarray(body_state_spec["qvel"], dtype=float)
        )

        self.xml_str = f"""
        <mujoco model="Multibody Free Flight">
          <option timestep="{delta_time}" integrator="RK4" gravity="0.0 0.0 0.0"/>

          {extra_default}
          {extra_asset}
          {extra_visual}

          <worldbody>
            {extra_worldbody}
            {"".join(body_xml_fragments)}
          </worldbody>

          <keyframe>
            <key name="start" qpos="{qpos_str}" qvel="{qvel_str}"/>
          </keyframe>
        </mujoco>
        """

        if mujoco_assets is not None:
            self.model = mujoco.MjModel.from_xml_string(
                self.xml_str, assets=mujoco_assets
            )
        else:
            self.model = mujoco.MjModel.from_xml_string(self.xml_str)

        self.model.opt.timestep = delta_time
        self.data: mujoco.MjData = mujoco.MjData(self.model)

        self.initial_key_frame_id: int = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "start"
        )
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.initial_key_frame_id)
        mujoco.mj_forward(self.model, self.data)

        self.body_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                for body_name in self.body_names
            ],
            dtype=int,
        )
        self.joint_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                for joint_name in self.joint_names
            ],
            dtype=int,
        )
        self.body_qposadrs = np.array(
            [self.model.jnt_qposadr[joint_id] for joint_id in self.joint_ids],
            dtype=int,
        )
        self.body_qveladrs = np.array(
            [self.model.jnt_dofadr[joint_id] for joint_id in self.joint_ids],
            dtype=int,
        )

        self._initial_qpos = np.copy(self.data.qpos)
        self._initial_qvel = np.copy(self.data.qvel)

    def apply_loads(
        self,
        forces_E: np.ndarray | Sequence[Sequence[float | int]],
        moments_E_Cg: np.ndarray | Sequence[Sequence[float | int]],
    ) -> None:
        """Apply one Earth-frame load vector and one moment vector per body."""
        forces_E = (
            _parameter_validation.arrayLike_of_threeD_number_vectorLikes_return_float(
                forces_E, "forces_E"
            )
        )
        moments_E_Cg = (
            _parameter_validation.arrayLike_of_threeD_number_vectorLikes_return_float(
                moments_E_Cg, "moments_E_Cg"
            )
        )

        if forces_E.shape != (self.num_bodies, 3):
            raise ValueError(
                f"forces_E must have shape ({self.num_bodies}, 3), got {forces_E.shape}."
            )
        if moments_E_Cg.shape != (self.num_bodies, 3):
            raise ValueError(
                f"moments_E_Cg must have shape ({self.num_bodies}, 3), got "
                f"{moments_E_Cg.shape}."
            )

        self.data.xfrc_applied[:] = 0.0
        for body_row, body_id in enumerate(self.body_ids):
            self.data.xfrc_applied[body_id][:] = np.hstack(
                [forces_E[body_row], moments_E_Cg[body_row]]
            )

    def step(self) -> None:
        """Advance the multibody MuJoCo simulation by one time step."""
        mujoco.mj_step(self.model, self.data)

    def get_states(self) -> dict[str, np.ndarray | float]:
        """Return the state of every body as per-body arrays."""
        positions_E_E = np.zeros((self.num_bodies, 3), dtype=float)
        R_pas_E_to_BPs = np.zeros((self.num_bodies, 3, 3), dtype=float)
        velocities_E__E = np.zeros((self.num_bodies, 3), dtype=float)
        omegas_BPs__E = np.zeros((self.num_bodies, 3), dtype=float)

        for body_row, (body_id, qpos_adr, qvel_adr) in enumerate(
            zip(
                self.body_ids,
                self.body_qposadrs,
                self.body_qveladrs,
                strict=True,
            )
        ):
            positions_E_E[body_row] = self.data.qpos[qpos_adr : qpos_adr + 3]
            velocities_E__E[body_row] = self.data.qvel[qvel_adr : qvel_adr + 3]
            omegas_BPs__E[body_row] = np.rad2deg(
                self.data.qvel[qvel_adr + 3 : qvel_adr + 6]
            )
            R_pas_BP_to_E = self.data.xmat[body_id].reshape(3, 3)
            R_pas_E_to_BPs[body_row] = R_pas_BP_to_E.T

        return {
            "positions_E_E": positions_E_E,
            "R_pas_E_to_BPs": R_pas_E_to_BPs,
            "velocities_E__E": velocities_E__E,
            "omegas_BPs__E": omegas_BPs__E,
            "time": float(self.data.time),
        }

    def get_state(self, body_index: int) -> dict[str, np.ndarray | float]:
        """Return the state of one body."""
        if not isinstance(body_index, (int, np.integer)):
            raise TypeError("body_index must be an int.")
        if body_index < 0 or body_index >= self.num_bodies:
            raise ValueError(
                f"body_index must be in [0, {self.num_bodies - 1}], got {body_index}."
            )

        states = self.get_states()
        positions_E_E = np.asarray(states["positions_E_E"], dtype=float)
        R_pas_E_to_BPs = np.asarray(states["R_pas_E_to_BPs"], dtype=float)
        velocities_E__E = np.asarray(states["velocities_E__E"], dtype=float)
        omegas_BPs__E = np.asarray(states["omegas_BPs__E"], dtype=float)
        return {
            "position_E_E": positions_E_E[body_index].copy(),
            "R_pas_E_to_BP": R_pas_E_to_BPs[body_index].copy(),
            "velocity_E__E": velocities_E__E[body_index].copy(),
            "omegas_BP__E": omegas_BPs__E[body_index].copy(),
            "time": float(states["time"]),
        }

    def reset(self) -> None:
        """Reset all bodies to their initial state and clear applied loads."""
        self.data.qpos[:] = self._initial_qpos
        self.data.qvel[:] = self._initial_qvel
        self.data.time = 0.0
        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
