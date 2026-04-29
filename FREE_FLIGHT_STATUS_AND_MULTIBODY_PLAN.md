# Free-Flight Status And Multibody Plan

## Scope

This file summarizes:

- the earlier exploratory free-flight work done in `/home/hht/PteraSoftware_free_flight`
- the branch-local work now implemented in `/home/hht/birdflock/PteraSoftware`
- the current recommended plan for multibody coupled free flight

The active development branch in this repo is `local/free_flight_cases`, based on `origin/feature/free_flight`, and all runs here use `.venv`.

## Repos And Locations

### Current working repo

- Repo: `/home/hht/birdflock/PteraSoftware`
- Branch: `local/free_flight_cases`
- Environment: `/home/hht/birdflock/PteraSoftware/.venv`

### Legacy exploratory worktree

- Worktree: `/home/hht/PteraSoftware_free_flight`
- Purpose: early flapping free-flight example development, tuning, long-run diagnostics, and movie generation
- Status: its custom example and all generated outputs have now been copied into this repo under `archive/previous_free_flight_worktree/`

## What Was Done In The Previous Folder

The previous free-flight worktree produced the first working flapping free-flight example, then iterated through diagnostics and tuning.

### Legacy example

- Original script:
  - `archive/previous_free_flight_worktree/examples/free_flight_flapping_wing_feature_branch.py`
- This script started as a no-ground-effect free-flight case derived from the feature branch's coupled UVLM + MuJoCo path.

### Legacy diagnostic and tuning milestones

1. A baseline flapping free-flight example was run and animated.
2. Wake-visible movies were generated so the shed vortex rings were shown explicitly.
3. Longer runs were executed, including:
   - 20-period diagnostics
   - 100-period attempts at smaller `dt`
4. Velocity, alpha, and Euler-angle histories were added for diagnosis.
5. A fixed-pitch mode was added to separate translational trim issues from pitch-dive feedback.
6. A weight-plus-body-drag search was run to find a forward-flight case with near-zero period-averaged `V_z`.

### Important legacy findings

- The early untuned flapping case did not reach 100 periods cleanly.
- The `96`-steps-per-flap long attempt failed at about `30.57` periods.
- The fixed-pitch tuned case that worked best used:
  - `weight = 0.0235 N`
  - `linear_drag = 0.00105 N/(m/s)`
  - `CdA = 0.0007 m^2`
  - `fixed_pitch = 5 deg`
- The long stable legacy fixed-pitch result reached `52` total flapping periods, with late-time mean vertical velocity very close to zero.

### Key legacy outputs now preserved here

- Early movies:
  - `archive/previous_free_flight_worktree/output/free_flight_movie/`
- Long-run movie and plot:
  - `archive/previous_free_flight_worktree/output/free_flight_long_run/`
- 20-period diagnostics:
  - `archive/previous_free_flight_worktree/output/free_flight_20_periods/`
- Failed 100-period attempts:
  - `archive/previous_free_flight_worktree/output/free_flight_100_periods_dt48/`
  - `archive/previous_free_flight_worktree/output/free_flight_100_periods_dt96/`
- Fixed-pitch drag/weight search:
  - `archive/previous_free_flight_worktree/output/fixed_pitch_weight_drag_search/`

## What Is Now Implemented In This Repo

### Dependency and environment updates

`imageio` is now recorded in:

- `requirements.txt`
- `requirements_min.txt`
- `setup.cfg`

This repo's `.venv` also has `mujoco` and `imageio` installed.

### New branch-local example scripts

- `examples/free_flight_case_utils.py`
  - shared history plotting, fixed-pitch clamping, JSON output, and WebP/MP4 animation helpers
- `examples/free_flight_gliding_wing.py`
  - case `01_gliding_wing`
  - same geometry family as the flapping case, but as a single continuous non-flapping wing
  - supports `fixed`, `free`, or `both` pitch modes
  - includes a glide-start search
- `examples/free_flight_flapping_forward.py`
  - case `03_flapping_forward`
  - tuned forward flapping case
  - fixed pitch only

## Current Organized Results

### Case 01: Gliding wing

Output root:

- `output/free_flight_cases/01_gliding_wing/`

Files:

- `output/free_flight_cases/01_gliding_wing/glide_start_search.json`
- `output/free_flight_cases/01_gliding_wing/fixed_pitch/summary.json`
- `output/free_flight_cases/01_gliding_wing/fixed_pitch/velocity_history.png`
- `output/free_flight_cases/01_gliding_wing/fixed_pitch/AnimateFreeFlight_gliding_wing_fixed_with_wake.webp`
- `output/free_flight_cases/01_gliding_wing/fixed_pitch/AnimateFreeFlight_gliding_wing_fixed_with_wake.mp4`
- `output/free_flight_cases/01_gliding_wing/free_pitch/summary.json`
- `output/free_flight_cases/01_gliding_wing/free_pitch/velocity_history.png`
- `output/free_flight_cases/01_gliding_wing/free_pitch/AnimateFreeFlight_gliding_wing_free_with_wake.webp`
- `output/free_flight_cases/01_gliding_wing/free_pitch/AnimateFreeFlight_gliding_wing_free_with_wake.mp4`

Result highlights:

- Glide-start search selected:
  - `initial_speed = 3.5 m/s`
  - `initial_alpha = 5.0 deg`
- Fixed-pitch case:
  - late-time behavior is a near-steady forward glide
  - last 2-reference-period mean velocity:
    - `[~0.0, 3.2154, 0.0487] m/s`
- Free-pitch case:
  - not trimmed
  - strong pitch excursion and large positive downward `V_z`
  - last 2-reference-period mean velocity:
    - `[~0.0, 1.5223, 20.5642] m/s`
  - pitch span:
    - about `91 deg`

### Case 03: Flapping forward

Output root:

- `output/free_flight_cases/03_flapping_forward/fixed_pitch/`

Files:

- `output/free_flight_cases/03_flapping_forward/fixed_pitch/summary.json`
- `output/free_flight_cases/03_flapping_forward/fixed_pitch/velocity_history.png`
- `output/free_flight_cases/03_flapping_forward/fixed_pitch/AnimateFreeFlight_flapping_forward_fixed_pitch_with_wake.webp`
- `output/free_flight_cases/03_flapping_forward/fixed_pitch/AnimateFreeFlight_flapping_forward_fixed_pitch_with_wake.mp4`

Result highlights:

- Tuned fixed-pitch case uses:
  - `weight = 0.0235 N`
  - `linear_drag = 0.00105 N/(m/s)`
  - `CdA = 0.0007 m^2`
  - `fixed_pitch = 5 deg`
- Late-time means from the current run:
  - last 2-period mean velocity:
    - `[-1.1826, 3.7010, -0.00223] m/s`
  - last 5-period mean velocity:
    - `[-0.5828, 3.7485, 0.000431] m/s`
- This is the current best branch-local forward-flapping benchmark case.

## What Was Moved Here

The following legacy assets were copied from `/home/hht/PteraSoftware_free_flight` into this repo:

- custom exploratory example script
- all generated diagnostic plots
- all movies and WebP animations
- all search CSV/JSON outputs
- all long-run summaries

They now live under:

- `archive/previous_free_flight_worktree/examples/`
- `archive/previous_free_flight_worktree/output/`

That means the old worktree is no longer needed as a record of the free-flight work. It can be removed later once we are comfortable with the archive copy.

## Current Gaps

- Case `02` and case `04` were intentionally not implemented yet.
- The gliding wing free-pitch case is not trimmed.
- The current coupled free-flight path still assumes one rigid body in the MuJoCo coupling layer.

## Multibody Progress So Far

Phase 0 is now started in code.

### Implemented validation slice

- Example:
  - `examples/multibody_two_gliding_wings_validation.py`
- Automated test:
  - `tests/integration/test_steady_ring_vortex_lattice_method_two_gliding_wings_far_apart.py`
- Output:
  - `output/free_flight_cases/phase0_multibody_validation/two_gliding_wings_far_apart/summary.json`

### What this validation does

- Builds the same geometry-matched gliding wing used in case `01`
- Solves a single-glider steady baseline
- Solves a two-glider steady problem with large lateral separation
- Compares each glider in the pair against the single-glider baseline

### Current result

The far-separated validation passes:

- max relative force-coefficient error vs single case:
  - about `5.87e-5`
- max relative moment-coefficient error vs single case:
  - about `5.85e-5`

This is the first implemented multibody regression check. It validates the existing
multi-airplane aerodynamic path for the simple no-interference limit before we begin
changing the true coupled free-flight dynamics layer.

### Newly implemented multibody MuJoCo wrapper slice

The first true multibody MuJoCo wrapper is now implemented in:

- `pterasoftware/_mujoco_model.py`
  - `MultiBodyMuJoCoModel`

What it now does:

- builds one MuJoCo rigid body per airplane
- gives each body its own `freejoint`
- packs per-body `qpos` and `qvel` into one MuJoCo model/keyframe
- stores per-body IDs and `qpos`/`qvel` address arrays
- applies one force vector and one moment vector per body
- returns full per-body state arrays through `get_states()`
- returns a one-body slice through `get_state(body_index)`
- supports clean reset of all bodies together

Wrapper verification now in repo:

- unit test:
  - `tests/unit/test_mujoco_model.py`

Current verification status:

- `.venv/bin/python -m unittest tests.unit.test_mujoco_model`
  - `Ran 10 tests ... OK`
- `.venv/bin/python -m unittest tests.integration.test_steady_ring_vortex_lattice_method_two_gliding_wings_far_apart`
  - `Ran 3 tests ... OK`

Important scope note:

- this is the MuJoCo multibody wrapper only
- the coupled UVLM free-flight solver path is still single-body above this layer
- the next implementation step is to thread per-body load/state arrays through the coupled problem and solver classes

## Multibody: Current Reality

The main aerodynamic codebase already has multi-airplane patterns in the general movement/problem infrastructure, but the coupled free-flight path is still single-body.

The strongest single-body assumptions currently sit in:

- `pterasoftware/movements/movement.py`
  - `CoupledMovement` takes exactly one `AirplaneMovement`
- `pterasoftware/problems.py`
  - `CoupledUnsteadyProblem` takes exactly one inertia matrix `I_BP1_CgP1`
- `pterasoftware/_mujoco_model.py`
  - builds one MuJoCo body with one `freejoint`
  - stores one `body_id`
  - applies loads to one rigid body
- `pterasoftware/coupled_unsteady_ring_vortex_lattice_method.py`
  - computes, applies, and reads back state for one airplane body
  - stores single-body histories like `stackPosition_E_E`
- `pterasoftware/output.py`
  - free-flight plotting and animation assume one body trajectory

## Recommended Multibody Plan

### Phase 0: Fix the initial scope

Start with the simplest useful multibody target:

- exactly `2` airplanes
- both are free rigid bodies
- no body-to-body contact model yet
- no articulated joints between aircraft
- same `delta_time` for both bodies
- same ambient air and global gravity
- first validation cases:
  - two gliding wings inline
  - then two flapping wings inline

This is enough to unlock case `02` and case `04` without over-designing the API.

### Phase 1: Generalize the coupled movement/problem data model

Add a multibody coupled movement class or generalize the existing one so it can carry:

- `airplane_movements: list[AirplaneMovement]`
- `initial_coupled_operating_points: list[CoupledOperatingPoint]`
- a shared `delta_time`
- shared prescribed/free step counts

Likewise, generalize `CoupledUnsteadyProblem` so it can accept:

- one inertia matrix per body
- optional per-body extra forces and moments

Practical recommendation:

- keep the current single-body API working
- add a new multibody path first, then merge APIs later if it stays clean

### Phase 2: Build a multibody MuJoCo wrapper

Refactor `_mujoco_model.py` so it can:

- create one MuJoCo body per airplane
- give each body its own `freejoint`
- store `body_ids: list[int]`
- accept per-body inertia and initial state
- apply loads body-by-body
- return state body-by-body

The right output shapes are probably:

- positions: `(num_bodies, 3)`
- rotation matrices: `(num_bodies, 3, 3)`
- velocities: `(num_bodies, 3)`
- angular velocities: `(num_bodies, 3)`

### Phase 3: Generalize the coupled solver loop

The coupled solver should then, at each time step:

1. solve the aerodynamic state for all airplanes together
2. extract per-airplane aerodynamic forces and moments
3. add per-body external/body-drag loads
4. add each airplane's weight
5. apply those loads to the matching MuJoCo body
6. step MuJoCo once
7. read back each body's new state
8. build the next `CoupledOperatingPoint` for each body

This is the critical step where single-body history arrays need to become multibody histories.

### Phase 4: Preserve aerodynamic interference cleanly

The good news is that the aerodynamic side should not be treated as independent solves if the aircraft are in formation. They must be solved together so each airplane sees the others' induced velocities and wakes.

That means the multibody implementation should:

- keep one combined aerodynamic solve per time step
- preserve airplane ownership of wings, panels, and wake structures
- retain per-airplane force and moment bookkeeping after the combined solve

### Phase 5: Extend outputs and examples

After the solver works, extend:

- velocity and Euler-angle plots to one subplot set per body or per selected body
- animation so multiple airplanes and wakes are shown together
- case scripts for:
  - inline gliding pair
  - inline flapping pair

## Main Multibody Challenges And How To Handle Them

### 1. Single-body assumptions are everywhere in the coupled layer

Challenge:

- many arrays, helper functions, and output paths assume only one body

Mitigation:

- make body index explicit early
- prefer shapes like `(num_steps, num_bodies, 3)` over ad hoc parallel lists
- add compatibility wrappers for one-body plotting and summaries

### 2. API design can get messy fast

Challenge:

- if we immediately over-generalize everything, the code will get hard to debug

Mitigation:

- start with `2` bodies only in the new path
- keep the current single-body API intact
- only generalize to arbitrary `N` after the two-body path is working cleanly

### 3. Formation aerodynamics and wake ownership

Challenge:

- each body needs its own loads, but the wake and induction are coupled across bodies

Mitigation:

- solve all bodies in one UVLM system each step
- keep explicit mappings from panels and wakes back to parent airplane/body

### 4. Numerical stability will get harder

Challenge:

- free-flight already gets sensitive in one body
- two interacting bodies can amplify that sensitivity

Mitigation:

- begin with prescribed wake
- begin with fixed pitch if needed
- start with the gliding pair before the flapping pair
- only increase DOF freedom after the baseline two-body path is stable

### 5. Initial condition selection is harder for two bodies

Challenge:

- it is not enough to trim one airplane; spacing, phase, and relative wake placement matter

Mitigation:

- first reuse single-body tuned conditions
- then add a simple spacing sweep
- for the flapping pair, keep the same flapping phase on both bodies initially

### 6. Output and debugging complexity rise quickly

Challenge:

- it becomes much harder to understand which body went unstable and why

Mitigation:

- store per-body summaries
- label all plots and histories by body index and name
- add a debug mode that plots only one selected body while keeping the full coupled solve

## Best First Multibody Implementation Target

The most practical first milestone is:

1. multibody MuJoCo wrapper with `2` freejoint bodies
2. coupled solver that advances two gliding wings inline
3. plots and animation for that two-glider case
4. then reuse the same infrastructure for two inline flapping wings

That sequence gives the lowest debugging risk while still moving directly toward case `02` and case `04`.
