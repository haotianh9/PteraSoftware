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

Located in `_calculate_loads()` (lines 1018-1312):
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

| File | Changes |
|------|---------|
| `pterasoftware/unsteady_ring_vortex_lattice_method.py` | Added parameter, helper methods, Katz implementation |
| `tests/unit/test_unsteady_ring_vortex_lattice_method.py` | Unit tests for parameter validation |
| `tests/integration/test_unsteady_ring_vortex_lattice_method_force_methods.py` | Comparison tests between methods |

## Implementation Details

### Step 1: Add `force_method` Parameter to `run()` (lines 178-208)

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

Add parameter validation (after existing validations around line 208):
```python
force_method = _parameter_validation.str_return_str(force_method, "force_method")
if force_method not in ("joukowski", "katz"):
    raise ValueError(
        f"force_method must be 'joukowski' or 'katz', got '{force_method}'."
    )
self._force_method = force_method
```

### Step 2: Add Instance Attribute in `__init__`

**Location**: Around line 176

```python
self._force_method: str = "joukowski"
```

### Step 3: Add New Data Structure Attributes in `__init__`

**Location**: After existing attribute declarations (around line 139)

```python
# Katz method specific arrays.
self._panel_chord_lengths: np.ndarray = np.empty(0, dtype=float)
self._panel_span_lengths: np.ndarray = np.empty(0, dtype=float)
self._stackChordwiseTangent_GP1: np.ndarray = np.empty(0, dtype=float)
self._stackSpanwiseTangent_GP1: np.ndarray = np.empty(0, dtype=float)
self._stackCentroid_GP1_CgP1: np.ndarray = np.empty(0, dtype=float)
```

### Step 4: Rename `_calculate_loads()` to `_calculate_loads_joukowski()`

**Location**: Lines 1018-1312

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
    # ... existing implementation (lines 1018-1312) ...
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

#### 5.2 `_calculate_chordwise_vorticity_gradients()` (line 1552)

Calculates true vorticity gradients (dGamma/dx) using backward differencing for non-leading-edge panels and one-sided differencing for leading edge panels.

**Key implementation details:**
- Returns gradients in units of meters per second (circulation / distance)
- Leading edge panels: `Gamma / (chord/2)` assuming zero vorticity upstream
- Non-leading-edge panels: `(Gamma_this - Gamma_front) / distance_between_centers`

#### 5.3 `_calculate_spanwise_vorticity_gradients()` (line 1621)

Calculates true vorticity gradients (dGamma/dy) using symmetric differencing to prevent spurious roll moments.

**Key implementation details:**
- Returns gradients in units of meters per second (circulation / distance)
- Left edge panels: forward difference `(Gamma_right - Gamma_this) / distance`
- Right edge panels: backward difference `(Gamma_this - Gamma_left) / distance`
- Interior panels: central difference `(Gamma_right - Gamma_left) / distance`
- Single-panel spanwise: gradient is zero

This symmetric treatment was chosen to prevent asymmetric gradient calculations from introducing non-physical roll moments.

#### 5.4 `_calculate_current_movement_velocities_at_centroids()` (line 1730)

Returns apparent velocities at Panel centroids due to prescribed motion.

**Key implementation details:**
- Uses `_stackLastCentroid_GP1_CgP1` attribute populated by `_populate_last_centroid_positions()`
- Returns negative of displacement velocity (apparent velocity is opposite to motion)
- Returns zeros for the first time step

#### 5.5 `_populate_last_centroid_positions()` (line 924)

Populates the `_stackLastCentroid_GP1_CgP1` attribute with centroid positions from the previous time step. Called from `_collapse_geometry_katz_data()` when not at the first time step.

### Step 6: Implement `_calculate_loads_katz()` (line 1476)

The actual implementation differs from the original plan in several ways:

1. **Vorticity gradients are true gradients**: The gradient methods now return `dGamma/dx` (units: m/s), so `_calculate_loads_katz()` simply multiplies by velocity components without dividing by panel lengths again.

2. **Sign convention for unsteady term**: The implementation uses `- d_gamma_dt` instead of `+ d_gamma_dt` to account for Ptera Software's CCW vertex ordering convention (vs. Katz & Plotkin's CW ordering). See detailed comment in the code at lines 1519-1529.

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

**Location**: After `_collapse_geometry()` call (around line 436)

```python
if self._force_method == "katz":
    self._collapse_geometry_katz_data()
```

Also need to initialize arrays at beginning of time step loop (around line 412):

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

| Case | Chordwise Gradient | Spanwise Gradient |
|------|-------------------|-------------------|
| Leading edge | `Gamma / (chord/2)` | Forward/backward/central based on position |
| Left edge | Backward difference | Forward difference |
| Right edge | Backward difference | Backward difference |
| Interior | Backward difference | Central difference |
| Single panel spanwise | Backward difference | Zero gradient |
| Trailing edge | Backward difference | Forward/backward/central based on position |
| Zero length panel | Returns zero (guarded) | Returns zero (guarded) |
| First time step | `dGamma/dt` uses zeros for last strengths | Same |

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
1. Line 1551: Question about why leading and trailing edges are treated differently in chordwise gradient
2. Lines 1618-1620: Question about whether to assume zero vorticity off the wing for spanwise edge treatment (similar to chordwise leading edge)
3. Lines 1715-1718: Question about whether central distance formula is valid for non-uniform spacings
