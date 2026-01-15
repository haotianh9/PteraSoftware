# Katz Method for Force Calculations (Issue #80)

## Summary

Added an optional `force_method` parameter to `UnsteadyRingVortexLatticeMethodSolver.run()` to enable the Katz pressure integration method as an alternative to the current Joukowski (Kutta-Joukowski) method for aerodynamic force calculations.

## Status: Implemented

The Katz method is fully implemented and tested. See the implementation details below.

## Requirements

- **Scope**: Unsteady solver only
- **Parameter**: `force_method` in `run()` method
- **Default**: `"joukowski"` (backward compatible)
- **Alternative**: `"katz"`
- **Testing**: Unit tests + comparison tests (Katz vs Joukowski)

## Theoretical Background

### Current Joukowski Method

Located in `_calculate_loads_joukowski()`:
```
F = rho x Gamma x (V x L)  for each of 4 ring vortex legs
F_unsteady = -rho x (dGamma/dt) x A x n_hat
```

### Katz Method (Katz & Plotkin Section 13.12, Eq. 13.150-13.151)

```
delta_p = rho x [ V.tau_i x (Gamma_ij - Gamma_i-1,j) / c_ij
                + V.tau_j x (Gamma_ij - Gamma_i,j-1) / b_ij
                + dGamma_ij / dt ]

F = -(delta_p x S)_ij x n_hat_ij
```

Where:
- `V` = local velocity at Panel centroid
- `tau_i` = chordwise tangent vector (normalized)
- `tau_j` = spanwise tangent vector (normalized)
- `Gamma_ij` = circulation at Panel (i,j)
- `Gamma_i-1,j` = circulation at Panel in front (chordwise)
- `Gamma_i,j-1` = circulation at Panel to the left (spanwise)
- `c_ij` = Panel chord length
- `b_ij` = Panel span length
- `S_ij` = Panel area
- `n_hat_ij` = Panel unit normal

## Files Modified

| File                                                                          | Changes                                              |
|-------------------------------------------------------------------------------|------------------------------------------------------|
| `pterasoftware/unsteady_ring_vortex_lattice_method.py`                        | Added parameter, helper methods, Katz implementation |
| `tests/unit/test_unsteady_ring_vortex_lattice_method.py`                      | Unit tests for parameter validation                  |
| `tests/integration/test_unsteady_ring_vortex_lattice_method_force_methods.py` | Comparison tests between methods                     |

## Implementation Details

### Step 1: Add `force_method` Parameter to `run()`

**Location**: `pterasoftware/unsteady_ring_vortex_lattice_method.py`

Modify signature:
```python
def run(
    self,
    prescribed_wake: bool | np.bool_ = True,
    calculate_streamlines: bool | np.bool_ = True,
    show_progress: bool | np.bool_ = True,
    force_method: str = "joukowski",  # NEW
) -> None:
```

Add docstring entry (after show_progress description):
```python
:param force_method: The method to use for calculating aerodynamic forces. Valid
    options are "joukowski" (default) which uses the Kutta-Joukowski theorem on
    each RingVortex leg, and "katz" which uses the pressure integration method
    from Katz and Plotkin (Section 13.12, Eq. 13.150-13.151). Both methods
    include the unsteady force term from the unsteady Bernoulli equation. The
    default is "joukowski".
```

Add parameter validation (after existing validations):
```python
force_method = _parameter_validation.str_return_str(force_method, "force_method")
if force_method not in ("joukowski", "katz"):
    raise ValueError(
        f"force_method must be 'joukowski' or 'katz', got '{force_method}'."
    )
self._force_method = force_method
```

### Step 2: Add Instance Attribute in `__init__`

**Location**: `pterasoftware/unsteady_ring_vortex_lattice_method.py`

```python
self._force_method: str = "joukowski"
```

### Step 3: Add New Data Structure Attributes in `__init__`

**Location**: After existing attribute declarations in `__init__`

```python
# Katz method specific arrays.
self._panel_chord_lengths: np.ndarray = np.empty(0, dtype=float)
self._panel_span_lengths: np.ndarray = np.empty(0, dtype=float)
self._stackChordwiseTangent_GP1: np.ndarray = np.empty(0, dtype=float)
self._stackSpanwiseTangent_GP1: np.ndarray = np.empty(0, dtype=float)
self._stackCentroid_GP1_CgP1: np.ndarray = np.empty(0, dtype=float)
```

### Step 4: Rename `_calculate_loads()` to `_calculate_loads_joukowski()`

Rename the existing method and create a dispatcher:

```python
def _calculate_loads(self) -> None:
    """Dispatches to the appropriate force calculation method.

    :return: None
    """
    if self._force_method == "joukowski":
        self._calculate_loads_joukowski()
    else:
        self._calculate_loads_katz()

def _calculate_loads_joukowski(self) -> None:
    """Calculates the forces using the Kutta-Joukowski theorem.

    [Original docstring and implementation from _calculate_loads()]
    """
    # ... existing implementation ...
```

### Step 5: Add Helper Methods for Katz Method

#### 5.1 `_collapse_geometry_katz_data()`

Populates tangent vectors, chord/span lengths, and centroid positions for all Panels.

```python
def _collapse_geometry_katz_data(self) -> None:
    """Populates Katz method specific geometric data for all Panels.

    Calculates the tangent vectors, chord lengths, span lengths, and centroid
    positions needed for the Katz pressure integration method.

    :return: None
    """
    global_panel_position = 0

    for airplane in self.current_airplanes:
        for wing in airplane.wings:
            _panels = wing.panels
            assert _panels is not None

            panels = np.ravel(_panels)

            panel: _panel.Panel
            for panel in panels:
                # Get leg vectors.
                _rightLeg_GP1 = panel.rightLeg_GP1
                _frontLeg_GP1 = panel.frontLeg_GP1
                _leftLeg_GP1 = panel.leftLeg_GP1
                _backLeg_GP1 = panel.backLeg_GP1
                assert _rightLeg_GP1 is not None
                assert _frontLeg_GP1 is not None
                assert _leftLeg_GP1 is not None
                assert _backLeg_GP1 is not None

                # Chordwise tangent (average of right and left legs, normalized).
                chordwise_vec = (_rightLeg_GP1 - _leftLeg_GP1) / 2
                chordwise_length = float(np.linalg.norm(chordwise_vec))
                if chordwise_length > 0:
                    self._stackChordwiseTangent_GP1[global_panel_position, :] = (
                            chordwise_vec / chordwise_length)
                self._panel_chord_lengths[global_panel_position] = chordwise_length

                # Spanwise tangent (average of front and back legs, normalized).
                spanwise_vec = (_frontLeg_GP1 - _backLeg_GP1) / 2
                spanwise_length = float(np.linalg.norm(spanwise_vec))
                if spanwise_length > 0:
                    self._stackSpanwiseTangent_GP1[global_panel_position, :] = (
                            spanwise_vec / spanwise_length)
                self._panel_span_lengths[global_panel_position] = spanwise_length

                # Centroid (average of four corners).
                _Frpp = panel.Frpp_GP1_CgP1
                _Flpp = panel.Flpp_GP1_CgP1
                _Blpp = panel.Blpp_GP1_CgP1
                _Brpp = panel.Brpp_GP1_CgP1
                assert _Frpp is not None
                assert _Flpp is not None
                assert _Blpp is not None
                assert _Brpp is not None

                self._stackCentroid_GP1_CgP1[global_panel_position, :] = (
                                                                                 _Frpp + _Flpp + _Blpp + _Brpp) / 4

                global_panel_position += 1
```

#### 5.2 `_calculate_chordwise_vorticity_gradients()`

Calculates true vorticity gradients (dGamma/dx) using backward differencing for non leading edge Panels and one sided differencing for leading edge Panels.

**Key implementation details:**
- Returns gradients in units of meters per second (circulation / distance)
- Leading edge Panels: `Gamma / (chord/2)` assuming zero vorticity upstream
- Non leading edge Panels: `(Gamma_this - Gamma_front) / distance_between_centers`

#### 5.3 `_calculate_spanwise_vorticity_gradients()`

Calculates true vorticity gradients (dGamma/dy) using symmetric differencing to prevent spurious roll moments.

**Key implementation details:**
- Returns gradients in units of meters per second (circulation / distance)
- Left edge Panels: forward difference `(Gamma_right - Gamma_this) / distance`
- Right edge Panels: backward difference `(Gamma_this - Gamma_left) / distance`
- Interior Panels: central difference `(Gamma_right - Gamma_left) / distance`
- Single Panel spanwise: gradient is zero

This symmetric treatment was chosen to prevent asymmetric gradient calculations from introducing nonphysical roll moments.

#### 5.4 `_calculate_current_movement_velocities_at_centroids()`

Returns apparent velocities at Panel centroids due to prescribed motion.

**Key implementation details:**
- Uses `_stackLastCentroid_GP1_CgP1` attribute populated by `_populate_last_centroid_positions()`
- Returns negative of displacement velocity (apparent velocity is opposite to motion)
- Returns zeros for the first time step

#### 5.5 `_populate_last_centroid_positions()`

Populates the `_stackLastCentroid_GP1_CgP1` attribute with centroid positions from the previous time step. Called from `_collapse_geometry_katz_data()` when not at the first time step.

### Step 6: Implement `_calculate_loads_katz()`

The actual implementation differs from the original plan in several ways:

1. **Vorticity gradients are true gradients**: The gradient methods now return `dGamma/dx` (units: m/s), so `_calculate_loads_katz()` simply multiplies by velocity components without dividing by panel lengths again.

2. **Sign convention for unsteady term**: The implementation uses `- d_gamma_dt` instead of `+ d_gamma_dt` to account for Ptera Software's CCW vertex ordering convention (vs. Katz & Plotkin's CW ordering). See detailed comment in the code.

3. **Sign convention for force**: The implementation uses `F = +delta_p * S * n_hat` (positive sign) because `delta_p` is defined as `p_lower - p_upper` and the normal points upward.

**Actual implementation summary:**
```python
# Vorticity gradients already include division by distance
chord_term = chordwise_velocity_component * chordwise_vorticity_gradients
span_term = spanwise_velocity_component * spanwise_vorticity_gradients

# Sign convention adjusted for CCW vertex ordering
delta_p = rho * (chord_term + span_term - d_gamma_dt)

# Positive sign because delta_p = p_lower - p_upper
forces_GP1 = +delta_p * S * n_hat
```

### Step 7: Add Conditional Call in `run()` Time Step Loop

**Location**: After `_collapse_geometry()` call in the time step loop

```python
if self._force_method == "katz":
    self._collapse_geometry_katz_data()
```

Also need to initialize arrays at beginning of time step loop:

```python
# Katz method specific arrays.
self._panel_chord_lengths = np.zeros(self.num_panels, dtype=float)
self._panel_span_lengths = np.zeros(self.num_panels, dtype=float)
self._stackChordwiseTangent_GP1 = np.zeros((self.num_panels, 3), dtype=float)
self._stackSpanwiseTangent_GP1 = np.zeros((self.num_panels, 3), dtype=float)
self._stackCentroid_GP1_CgP1 = np.zeros((self.num_panels, 3), dtype=float)
```

### Step 8: Unit Tests (Implemented)

**File**: `tests/unit/test_unsteady_ring_vortex_lattice_method.py`

Implemented test cases:
- `test_force_method_parameter_default`: Verifies default is "joukowski"
- `test_force_method_parameter_joukowski`: Verifies "joukowski" is accepted and solver runs
- `test_force_method_parameter_katz`: Verifies "katz" is accepted and solver runs
- `test_force_method_parameter_invalid_string`: Verifies ValueError for invalid strings (including case-sensitive variants)
- `test_force_method_parameter_invalid_type`: Verifies TypeError for non-strings (int, float, None, bool, list, dict)

### Step 9: Comparison Tests (Implemented)

**File**: `tests/integration/test_unsteady_ring_vortex_lattice_method_force_methods.py`

Implemented test cases:
- `test_static_geometry_methods_produce_similar_lift`: Both methods within 25% for standard case
- `test_static_geometry_methods_produce_similar_drag`: Both methods within 100% (drag is most sensitive to method)
- `test_static_geometry_methods_produce_similar_moment`: Both methods within 50%
- `test_variable_geometry_joukowski_completes`: Joukowski method runs without errors on variable geometry
- `test_variable_geometry_katz_completes`: Katz method runs without errors on variable geometry

## Edge Cases (Actual Implementation)

| Case                  | Chordwise Gradient                        | Spanwise Gradient                          |
|-----------------------|-------------------------------------------|--------------------------------------------|
| Leading edge          | `Gamma / (chord/2)`                       | Forward/backward/central based on position |
| Left edge             | Backward difference                       | Forward difference                         |
| Right edge            | Backward difference                       | Backward difference                        |
| Interior              | Backward difference                       | Central difference                         |
| Single panel spanwise | Backward difference                       | Zero gradient                              |
| Trailing edge         | Backward difference                       | Forward/backward/central based on position |
| Zero length panel     | Returns zero (guarded)                    | Returns zero (guarded)                     |
| First time step       | `dGamma/dt` uses zeros for last strengths | Same                                       |

## Sign Convention (Actual Implementation)

The implementation uses:
- `delta_p = rho * (chord_term + span_term - d_gamma_dt)` (note the minus sign on unsteady term)
- `F = +delta_p * S * n_hat` (positive sign)

This differs from the theoretical formula (`F = -delta_p * S * n_hat` with `+d_gamma_dt`) due to:
1. Ptera Software uses CCW vertex ordering vs. Katz & Plotkin's CW ordering
2. `delta_p` is defined as `p_lower - p_upper` with normal pointing upward

The combined effect produces correct upward lift for positive angle of attack.

## Coordinate System

All calculations use the first Airplane's geometry axes (GP1) with variables following the naming conventions from `AXES_POINTS_AND_FRAMES.md`:
- `_stackChordwiseTangent_GP1`: Chordwise tangent vectors in GP1 axes
- `_stackSpanwiseTangent_GP1`: Spanwise tangent vectors in GP1 axes
- `_stackCentroid_GP1_CgP1`: Centroid positions in GP1 axes, relative to GP1 CG
- `_stackLastCentroid_GP1_CgP1`: Previous time step centroid positions
- `stackVelocityCentroid_GP1__E`: Velocities in GP1 axes, observed from Earth frame

## Open Questions / Future Work

The implementation includes REFACTOR comments noting areas for potential improvement:
1. Question about why leading and trailing edges are treated differently in chordwise gradient
2. Question about whether to assume zero vorticity off the wing for spanwise edge treatment (similar to chordwise leading edge)
3. Question about whether central distance formula is valid for nonuniform spacings
4. **Leading edge factor of 2 discrepancy**: The implementation uses `Gamma / (chord/2)` for leading edge Panels, but Katz and Plotkin Eq. 13.150 implies `Gamma / c` (dividing by full chord length). This is because Katz and Plotkin's formula `(Gamma_ij - Gamma_{i-1,j}) / c_ij` at the leading edge becomes `Gamma_ij / c_ij` when `Gamma_{i-1,j} = 0`. The current implementation effectively doubles the chordwise pressure contribution at the leading edge compared to Katz and Plotkin.
5. **Spanwise gradient differencing scheme**: Katz and Plotkin uses backward differencing `(Gamma_ij - Gamma_{i,j-1}) / b_ij` for all Panels, while the implementation uses central differencing for interior Panels and one sided differences at edges. At the left edge, Katz and Plotkin would implicitly use `Gamma_ij / b_ij` (assuming zero circulation off the wing), but the implementation uses forward differencing to the right neighbor instead. Perhaps we should stick with our central differencing approach but consider virtual zero vorticity Panels off the edges?

---

## Phase 2: Induced Drag Correction (In Progress)

### Overview

The current Katz implementation computes forces by projecting pressure forces onto wind axes, which Katz and Plotkin explicitly notes "does not account for the leading edge suction force" and will "overestimate" induced drag. This phase implements the proper induced drag calculation following Lambert (2015), which adapts Katz and Plotkin's approach for complex kinematics.

### Theoretical Foundation

#### The Problem with Pressure Based Drag

When we compute `F = delta_p * S * n_hat` and project onto the freestream direction, we get:
- **Correct lift** (perpendicular to freestream)
- **Overestimated drag** (parallel to freestream) because thin airfoil theory predicts a leading edge suction force that partially cancels the pressure drag

#### Katz and Plotkin's Solution (Eq. 13.152)

For straight line motion, Katz and Plotkin provides a specific induced drag formula:

```
D_ij = rho * { (w_ind + w_W)_ij * (Gamma_ij - Gamma_{i-1,j}) * b_ij
             + (dGamma_ij/dt) * S_ij * sin(alpha_ij) }
```

Where:
- `w_ind` = downwash induced by **chordwise vortex segments only** (computed via b_KL coefficients)
- `w_W` = wake induced downwash
- `alpha_ij` = Panel angle of attack relative to freestream

**Limitation**: Katz and Plotkin notes "the main difficulty in the induced drag calculation for a general motion lies in the identification of the force component that will be designated as drag."

#### Lambert's Extension for Complex Kinematics

Lambert (2015) solves the drag direction problem by defining lift and drag **relative to the local flow velocity at each Panel**:

**Lift** (Eq. 2.14):
```
delta_L_ij = delta_p_ij * S_ij * cos(alpha_ij)
```

**Drag** (Eq. 2.15):
```
delta_D_ij = rho * { (U_bc_ij + U_w_ij) dot (P_U_hat * n_hat_ij) * (Gamma_ij - Gamma_{i-1,j}) * delta_b_ij
                   + (dGamma_ij/dt) * S_ij * sin(alpha_ij) }
```

**Total Force** (Eq. 2.16):
```
F_ij = delta_D_ij * U_hat_ij + delta_L_ij * (P_U_hat * n_hat_ij) / |P_U_hat * n_hat_ij|
```

Where:
- `U_hat_ij` = unit vector of local Panel velocity (drag direction)
- `P_U_hat = I - U_hat (outer product) U_hat^T` = projection operator onto plane perpendicular to flow
- `P_U_hat * n_hat_ij` = Panel normal projected into lift plane (lift direction)
- `|P_U_hat * n_hat_ij| = cos(alpha_ij)` naturally emerges from the projection
- `n_hat_ij dot U_hat_ij = sin(alpha_ij)` naturally emerges from the dot product
- `U_bc_ij` = velocity from **bound chordwise vortices only** (equivalent to Katz and Plotkin's b_KL)

### Key Insight: Implicit Angle Calculation

The projection operator elegantly handles angle of attack without explicit calculation:
- `cos(alpha) = |P_U_hat * n_hat|` (magnitude of projected normal)
- `sin(alpha) = n_hat dot U_hat` (normal component along flow direction)

This works for any Panel orientation and any local flow direction.

### Implementation Plan

#### Step 10: Compute Chordwise Induced Velocity (Implemented)

**10.1 New Method: `_calculate_chordwise_induced_velocity()` (Implemented)**

Computes the velocity at collocation points induced by chordwise (streamwise) bound vortex segments only. This uses the existing `collapsed_velocities_from_ring_vortices_chordwise_segments` function with the solved vortex strengths.

**Note:** The original plan proposed pre-computing b_KL influence coefficients (an N x N matrix) and storing them for later matrix-vector multiplication. However, since the Biot-Savart law is linear in vortex strength, we can simply call the existing collapsed velocity function with the actual solved strengths to get the same result. This avoids storing an extra (num_panels x num_panels) matrix and uses existing, tested code.

```python
def _calculate_chordwise_induced_velocity(self) -> np.ndarray:
    """Computes velocity at collocation points from bound chordwise vortex segments.

    Returns the velocity induced at each collocation point by the chordwise
    (streamwise) segments of all bound RingVortices. This corresponds to U_bc in
    Lambert (2015) Eq. 2.15 and w_ind in Katz and Plotkin Eq. 13.152.

    :return: A (num_panels, 3) ndarray of floats for the induced velocity (in the
        first Airplane's geometry axes, observed from the Earth frame) at each
        collocation point. The units are meters per second.
    """
    return _aerodynamics.collapsed_velocities_from_ring_vortices_chordwise_segments(
        stackP_GP1_CgP1=self.stackCpp_GP1_CgP1,
        stackBrrvp_GP1_CgP1=self.stackBrbrvp_GP1_CgP1,
        stackFrrvp_GP1_CgP1=self.stackFrbrvp_GP1_CgP1,
        stackFlrvp_GP1_CgP1=self.stackFlbrvp_GP1_CgP1,
        stackBlrvp_GP1_CgP1=self.stackBlbrvp_GP1_CgP1,
        strengths=self._current_bound_vortex_strengths,
        ages=None,
        nu=self.current_operating_point.nu,
    )
```

**10.2 New Method: `_calculate_wake_induced_velocity()` (Implemented)**

Computes the velocity at collocation points induced by wake vortices. The wake-wing influences are already computed for the linear system RHS, but we need the full velocity vector (not just the normal component) for the induced drag calculation.

```python
def _calculate_wake_induced_velocity(self) -> np.ndarray:
    """Computes velocity at collocation points from wake vortices.

    Returns the velocity induced at each collocation point by all wake RingVortices.
    This corresponds to U_w in Lambert (2015) Eq. 2.15.

    :return: A (num_panels, 3) ndarray of floats for the wake induced velocity (in
        the first Airplane's geometry axes, observed from the Earth frame) at each
        collocation point. The units are meters per second.
    """
    if self._current_step < 1:
        return np.zeros((self.num_panels, 3), dtype=float)

    return _aerodynamics.collapsed_velocities_from_ring_vortices(
        stackP_GP1_CgP1=self.stackCpp_GP1_CgP1,
        stackBrrvp_GP1_CgP1=self._currentStackBrwrvp_GP1_CgP1,
        stackFrrvp_GP1_CgP1=self._currentStackFrwrvp_GP1_CgP1,
        stackFlrvp_GP1_CgP1=self._currentStackFlwrvp_GP1_CgP1,
        stackBlrvp_GP1_CgP1=self._currentStackBlwrvp_GP1_CgP1,
        strengths=self._current_wake_vortex_strengths,
        ages=self._current_wake_vortex_ages,
        nu=self.current_operating_point.nu,
    )
```

#### Step 11: Local Reference Frame Calculation (Implemented)

**11.1 New Method: `_calculate_local_flow_directions()` (Implemented)**

The actual implementation takes the local velocities as a parameter rather than reading from an instance attribute:

```python
def _calculate_local_flow_directions(
    self,
    stackLocalVelocity_GP1__E: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes local flow unit vectors and projection operators for each Panel.

    For each Panel, calculates U_hat (unit vector of local flow velocity, which is
    the drag direction), P_U_hat * n_hat (Panel normal projected perpendicular to
    flow, which is the lift direction), and sin(alpha) and cos(alpha) (implicit
    angle of attack components).

    :return: A tuple of three ndarrays: (1) stackFlowUnitVectors_GP1, a
        (num_panels, 3) ndarray of floats for the unit flow directions (in the
        first Airplane's geometry axes), (2) stackLiftDirections_GP1, a
        (num_panels, 3) ndarray of floats for the lift direction vectors (in the
        first Airplane's geometry axes), and (3) stackSinAlpha, a (num_panels,)
        ndarray of floats for the sine of the local angle of attack. The units for
        the direction vectors are unitless. The units for stackSinAlpha are
        unitless.
    """
    # Get local Panel velocities (already computed for Katz pressure calculation).
    # This is U_m in Lambert's notation.
    local_velocities = self._stackLocalVelocityCentroid_GP1__E

    # Compute unit flow vectors U_hat.
    flow_magnitudes = np.linalg.norm(local_velocities, axis=1, keepdims=True)
    flow_magnitudes = np.maximum(flow_magnitudes, 1e-10)  # Prevent division by zero.
    stackFlowUnitVectors_GP1 = local_velocities / flow_magnitudes

    # Compute sin(alpha) = n_hat dot U_hat for each Panel.
    stackSinAlpha = np.einsum(
        "ij,ij->i",
        self.stackUnitNormals_GP1,
        stackFlowUnitVectors_GP1,
    )

    # Compute lift direction: P_U_hat * n_hat = n_hat - (n_hat dot U_hat) * U_hat.
    # This is the Panel normal with its flow parallel component removed.
    stackLiftDirections_GP1 = (
        self.stackUnitNormals_GP1
        - stackSinAlpha[:, np.newaxis] * stackFlowUnitVectors_GP1
    )

    return stackFlowUnitVectors_GP1, stackLiftDirections_GP1, stackSinAlpha
```

#### Step 12: Implement Lambert's Force Calculation

**12.1 Rename Current Method and Create New Implementation**

To preserve the current implementation for reference and comparison:

1. Rename `_calculate_loads_katz()` to `_calculate_loads_katz_old()`
2. Create a new `_calculate_loads_katz()` that implements Lambert's decomposition
3. Update the dispatcher in `_calculate_loads()` if needed

**New `_calculate_loads_katz()` Implementation:**

Implements Lambert's decomposition:

```python
def _calculate_loads_katz(self) -> None:
    """Calculates forces using Katz pressure integration with induced drag correction.

    Implements Lambert (2015) Equations 2.13 to 2.16, which extend Katz and Plotkin's method to
    handle complex kinematics by defining lift and drag relative to local Panel
    velocities.

    :return: None
    """
    # ... [existing pressure calculation code] ...

    # === NEW: Induced Drag Correction ===

    # Get local flow reference frame.
    (
        stackFlowUnitVectors_GP1,
        stackLiftDirections_GP1,
        stackSinAlpha,
    ) = self._calculate_local_flow_directions()

    # cos(alpha) = |P_U_hat * n_hat| (magnitude of lift direction vector).
    stackCosAlpha = np.linalg.norm(stackLiftDirections_GP1, axis=1)
    stackCosAlpha = np.maximum(stackCosAlpha, 1e-10)  # Prevent division by zero.

    # Normalize lift directions.
    stackLiftDirectionsNormalized_GP1 = (
        stackLiftDirections_GP1 / stackCosAlpha[:, np.newaxis]
    )

    # --- Lift Calculation (Lambert Eq. 2.14) ---
    # delta_L = delta_p * S * cos(alpha)
    stackLiftMagnitudes = delta_p * self.stackAreas * stackCosAlpha

    # --- Induced Drag Calculation (Lambert Eq. 2.15) ---
    # First term: (U_bc + U_w) dot (P_U_hat * n_hat) * (Gamma - Gamma_front) * b
    w_ind = self._calculate_induced_downwash()
    w_wake = self._calculate_wake_downwash()
    total_downwash = w_ind + w_wake

    # Project downwash onto lift direction to get the component that matters.
    # Note: w_ind and w_wake are already normal components, but we need to
    # account for the projection operator.
    downwash_lift_component = total_downwash * stackCosAlpha

    # Chordwise circulation difference (Gamma - Gamma_front).
    chordwise_circulation_diff = self._calculate_chordwise_circulation_differences()

    # First term of induced drag.
    induced_drag_term1 = (
        self.current_operating_point.density
        * downwash_lift_component
        * chordwise_circulation_diff
        * self._panel_span_lengths
    )

    # Second term: (dGamma/dt) * S * sin(alpha)
    induced_drag_term2 = (
        self.current_operating_point.density
        * d_gamma_dt
        * self.stackAreas
        * stackSinAlpha
    )

    stackDragMagnitudes = induced_drag_term1 + induced_drag_term2

    # --- Total Force (Lambert Eq. 2.16) ---
    # F = D * U_hat + L * (P_U_hat * n_hat) / |P_U_hat * n_hat|
    forces_GP1 = (
        stackDragMagnitudes[:, np.newaxis] * stackFlowUnitVectors_GP1
        + stackLiftMagnitudes[:, np.newaxis] * stackLiftDirectionsNormalized_GP1
    )

    # ... [rest of existing code to apply forces to Panels] ...
```

**12.2 New Helper Method: `_calculate_chordwise_vorticity_differences()` (Implemented)**

```python
def _calculate_chordwise_vorticity_differences(self) -> np.ndarray:
    """Computes (Gamma_ij - Gamma_{i-1,j}) for each Panel.

    For leading edge Panels, Gamma_{i-1,j} = 0 (no Panel upstream).

    :return: A (num_panels,) ndarray of floats for the circulation differences. The
        units are meters squared per second.
    """
    # This is similar to chordwise vorticity gradients but without dividing by
    # distance.
    differences = np.zeros(self.num_panels, dtype=float)

    global_panel_position = 0
    for airplane in self.current_airplanes:
        for wing in airplane.wings:
            _panels = wing.panels
            assert _panels is not None

            num_chordwise = _panels.shape[0]
            num_spanwise = _panels.shape[1]

            for i in range(num_chordwise):
                for j in range(num_spanwise):
                    current_gamma = self._current_bound_vortex_strengths[
                        global_panel_position
                    ]

                    if i == 0:
                        # Leading edge: Gamma_front = 0.
                        differences[global_panel_position] = current_gamma
                    else:
                        # Get strength of Panel in front.
                        front_panel_position = global_panel_position - num_spanwise
                        front_gamma = self._current_bound_vortex_strengths[
                            front_panel_position
                        ]
                        differences[global_panel_position] = current_gamma - front_gamma

                    global_panel_position += 1

    return differences
```

#### Step 13: Unit Tests

**File**: `tests/unit/test_unsteady_ring_vortex_lattice_method.py`

New test cases:
- `test_katz_induced_drag_symmetric_geometry`: Verify symmetric geometry produces zero roll moment
- `test_katz_induced_drag_lower_than_pressure_drag`: Verify induced drag correction reduces drag estimate
- `test_katz_local_angle_of_attack_calculation`: Verify sin/cos alpha computation for known geometry

#### Step 14: Integration Tests

**File**: `tests/integration/test_unsteady_ring_vortex_lattice_method_force_methods.py`

New test cases:
- `test_katz_vs_joukowski_induced_drag_comparison`: Compare drag estimates between methods
- `test_katz_induced_drag_steady_state_convergence`: Verify drag converges to expected value for simple case
- `test_katz_induced_drag_variable_geometry`: Verify method handles flapping motion

### Edge Cases

| Case                                           | Handling                                                                      |
|------------------------------------------------|-------------------------------------------------------------------------------|
| Zero local velocity                            | Guard with minimum magnitude (1e-10) to prevent NaN                           |
| Panel normal parallel to flow (alpha = 90 deg) | cos(alpha) approaches 0, lift direction undefined; use pressure only fallback |
| First time step (no wake)                      | w_wake = 0, proceed normally                                                  |
| Leading edge Panel                             | Gamma_{i-1,j} = 0 as specified by Katz and Plotkin                            |

### Validation Strategy

1. **XLFR5 comparison**: Compare against coefficients from XFLR5 simulations for simple steady cases (see `/tests/integration/test_unsteady_ring_vortex_lattice_method_static_geometry.py`)

2. **Analytical comparison**: For a flat rectangular wing in steady forward flight, compare induced drag coefficient with lifting line theory: `C_Di = C_L^2 / (pi * AR * e)`

3. **Symmetry check**: Verify symmetric geometry produces symmetric forces (zero roll/yaw moments)

4. **Convergence study**: Verify drag converges as Panel count increases

### References

- Katz, J. and Plotkin, A. (2001). *Low-Speed Aerodynamics*, 2nd Edition. Section 13.12. (local copy: `docs\katz_plotkin_13_12\katz_plotkin_13_12.md`)
- Lambert, T. (2015). *Discrete-Time State-Space UVLM*. Section 2.3-2.4. (local copy: `docs\lambert_2015_2_3__2_4\lambert_2015_2_3__2_4.md`)
