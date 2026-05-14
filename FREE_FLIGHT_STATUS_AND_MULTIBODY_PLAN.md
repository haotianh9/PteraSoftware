# Free-Flight And Streamwise Formation Status

## Current Repo

- Repo: `/home/hht/birdflock/PteraSoftware`
- Branch: `local/free_flight_cases`
- Python environment: `.venv`

This branch now keeps the active free-flight and formation-study code, but no longer
tracks exploratory result artifacts or the copied legacy worktree outputs.

## Active Scripts

Single-body free-flight examples:

- `examples/free_flight_gliding_wing.py`
- `examples/free_flight_flapping_forward.py`
- `examples/free_flight_case_utils.py`

Current fixed-wing streamwise formation workflow:

- `examples/single_wing_streamwise_steady_check.py`
- `examples/multibody_streamwise_stability_energy_sweep.py`
- `examples/fixed_formation_parallel_sweep.py`
- `examples/organize_streamwise_fixed_wing_dataset.py`
- `examples/plot_single_wing_wake_four_panel.py`
- `examples/plot_streamwise_sim_by_aoa_single_analytic.py`

Legacy exploratory result archives and old generated phase outputs have been removed
from version control.

## Active Dataset

The only retained formation-study output tree is:

- `output/free_flight_cases/streamwise_stability_energy/`

Its clean layout is:

- `curated_fixed_wing_dataset/`: canonical CSV/JSON summaries and regenerated figures.
- `raw_sources/`: raw simulation summaries/history snapshots needed to rebuild the curated dataset.

The curated dataset currently excludes unconverged `X/B = 9` cases and contains:

- `280` selected rows
- no duplicate `(AOA, X/B, Y/B, Z/B)` keys
- `X/B = 0.25, 0.5, 1.5, 2, 3, 4, 5, 7`

Rebuild command:

```bash
.venv/bin/python examples/organize_streamwise_fixed_wing_dataset.py
```

## Figure Policy

The script `plot_streamwise_analytic_vs_sim_8panel.py` was removed because it repeated
an AOA-independent analytical model across AOA panels, which was misleading.

The replacement is:

- `examples/plot_streamwise_sim_by_aoa_single_analytic.py`

It keeps simulation panels by AOA and shows only one analytical reference panel.

## Scope Notes

- The fixed-wing formation runs use real/free wake, not prescribed wake.
- In the streamwise formation study, the bodies are held in a fixed formation and the
  streamwise clamp/thrust diagnostics are recorded.
- The current analytical panel is a lightweight tip-vortex reference. It should not be
  treated as an AOA-dependent theory until an explicit AOA/circulation model is added.

## Cleanup Notes

Removed from version control:

- copied legacy worktree archive under `archive/previous_free_flight_worktree/`
- old single-body generated result files under `output/free_flight_cases/01_gliding_wing/`
- old flapping generated result files under `output/free_flight_cases/03_flapping_forward/`
- old phase-0 and phase-1 generated multibody result files

The remaining tracked files are code, tests, documentation, and maintained workflow
scripts. Generated datasets and figures should stay untracked unless explicitly agreed
otherwise.
