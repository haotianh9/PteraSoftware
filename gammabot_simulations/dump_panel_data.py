"""Dump panel geometry data for specific convergence parameters.

Compares panel-level geometry across NC=12, 13, 14 at AR=1 to find anomalies
that might explain the NC=13 divergence.

Usage:
    python dump_panel_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import dxf_to_csv
import numpy as np

import pterasoftware as ps
from pterasoftware.convergence._functions import _verify_panel_aspect_ratio
from pterasoftware.convergence.unsteady_non_trapezoidal import (
    _get_num_cross_sections_for_panel_ar,
)

import configs

# ── Parameters ────────────────────────────────────────────────────────────────
TARGET_AR = 1
CHORDWISE_VALUES = [12, 13, 14]
CONFIG_NAME = "L170V_R180V_170Hz"
# ──────────────────────────────────────────────────────────────────────────────


def build_airplane(nc: int, wing_config: dict, shared: dict) -> ps.geometry.airplane.Airplane:
    """Build an Airplane at the given chordwise panel count."""
    wing_spacing = float(shared["wing_spacing"])
    chordwise_spacing = str(shared["chordwise_spacing"])
    dxf_filepath = Path(__file__).parent / "gammabot_approximate_wing.dxf"

    ref_data = dxf_to_csv.process_dxf_to_wing_section_data(str(dxf_filepath), 8)
    span = float(np.sum(np.sqrt(np.sum(ref_data[1:, :3] ** 2, axis=1))))
    avg_chord = float(np.mean(ref_data[:, 3]))

    num_sections = _get_num_cross_sections_for_panel_ar(span, avg_chord, TARGET_AR, nc)
    wing_section_data = dxf_to_csv.process_dxf_to_wing_section_data(
        str(dxf_filepath), num_sections
    )
    num_wcs = num_sections + 1

    def make_cross_sections():
        cross_sections = []
        for i in range(num_wcs):
            num_spanwise_panels = 1 if i < num_sections else None
            wcs = ps.geometry.wing_cross_section.WingCrossSection(
                Lp_Wcsp_Lpp=tuple(wing_section_data[i, :3]),
                angles_Wcsp_to_Wcs_ixyz=(0.0, 0.0, 0.0),
                chord=float(wing_section_data[i, 3]),
                airfoil=ps.geometry.airfoil.Airfoil(name="naca0012"),
                num_spanwise_panels=num_spanwise_panels,
            )
            cross_sections.append(wcs)
        return cross_sections

    left_params = wing_config["left"]
    right_params = wing_config["right"]

    return ps.geometry.airplane.Airplane(
        wings=[
            ps.geometry.wing.Wing(
                wing_cross_sections=make_cross_sections(),
                Ler_Gs_Cgs=(0.0, wing_spacing / 2, 0.0),
                angles_Gs_to_Wn_ixyz=(
                    left_params["phi_v_shift"],
                    left_params["psi_v_shift"],
                    0.0,
                ),
                symmetric=False,
                mirror_only=False,
                num_chordwise_panels=nc,
                chordwise_spacing=chordwise_spacing,
            ),
            ps.geometry.wing.Wing(
                wing_cross_sections=make_cross_sections(),
                Ler_Gs_Cgs=(0.0, wing_spacing / 2, 0.0),
                angles_Gs_to_Wn_ixyz=(
                    right_params["phi_v_shift"],
                    right_params["psi_v_shift"],
                    0.0,
                ),
                symmetric=False,
                mirror_only=True,
                symmetryNormal_G=(0, 1, 0),
                symmetryPoint_G_Cg=(0, 0, 0),
                num_chordwise_panels=nc,
                chordwise_spacing=chordwise_spacing,
            ),
        ],
        name="GammaBot",
    )


def analyze_panels(airplane: ps.geometry.airplane.Airplane, nc: int) -> dict:
    """Extract panel statistics from an airplane."""
    all_areas = []
    all_ars = []
    all_cpp = []  # collocation points
    all_frbvp = []  # front-right bound vortex points
    all_flbvp = []  # front-left bound vortex points

    for wing_id, wing in enumerate(airplane.wings):
        if wing.panels is None:
            continue
        panels = np.ravel(wing.panels)
        for panel in panels:
            all_areas.append(panel.area)
            all_ars.append(panel.aspect_ratio)
            all_cpp.append(panel.Cpp_G_Cg)
            all_frbvp.append(panel.Frbvp_G_Cg)
            all_flbvp.append(panel.Flbvp_G_Cg)

    areas = np.array(all_areas)
    ars = np.array(all_ars)
    cpp = np.array(all_cpp)
    frbvp = np.array(all_frbvp)
    flbvp = np.array(all_flbvp)

    # Compute minimum distance between each collocation point and all
    # bound vortex segment endpoints from OTHER panels.
    n = len(cpp)
    bvp = np.vstack([frbvp, flbvp])  # (2N, 3)

    # Distance from each Cpp to every bound vortex point.
    # cpp: (N, 3), bvp: (2N, 3)
    diffs = cpp[:, np.newaxis, :] - bvp[np.newaxis, :, :]  # (N, 2N, 3)
    dists = np.linalg.norm(diffs, axis=2)  # (N, 2N)

    # For each Cpp, exclude distances to its own panel's bvp (indices i and i+N).
    for i in range(n):
        dists[i, i] = np.inf      # own frbvp
        dists[i, i + n] = np.inf  # own flbvp

    min_cpp_to_bvp = np.min(dists, axis=1)  # (N,)

    # Also compute pairwise distances between collocation points.
    cpp_diffs = cpp[:, np.newaxis, :] - cpp[np.newaxis, :, :]
    cpp_dists = np.linalg.norm(cpp_diffs, axis=2)
    np.fill_diagonal(cpp_dists, np.inf)
    min_cpp_to_cpp = np.min(cpp_dists, axis=1)

    return {
        "areas": areas,
        "ars": ars,
        "cpp": cpp,
        "frbvp": frbvp,
        "flbvp": flbvp,
        "min_cpp_to_bvp": min_cpp_to_bvp,
        "min_cpp_to_cpp": min_cpp_to_cpp,
    }


def main() -> None:
    wing_config = configs.get_config(CONFIG_NAME)
    shared = configs.SHARED_PARAMS

    for nc in CHORDWISE_VALUES:
        print(f"{'=' * 70}")
        print(f"NC = {nc}")
        print(f"{'=' * 70}")

        airplane = build_airplane(nc, wing_config, shared)
        ar_ok, actual_ar = _verify_panel_aspect_ratio(airplane, TARGET_AR)

        total_panels = sum(
            np.ravel(w.panels).size for w in airplane.wings if w.panels is not None
        )
        print(f"  Total panels: {total_panels}, Actual AR: {actual_ar:.4f}")

        stats = analyze_panels(airplane, nc)

        print(f"\n  Panel areas (m^2):")
        print(f"    min:    {np.min(stats['areas']):.4e}")
        print(f"    max:    {np.max(stats['areas']):.4e}")
        print(f"    mean:   {np.mean(stats['areas']):.4e}")
        print(f"    std:    {np.std(stats['areas']):.4e}")
        print(f"    ratio:  {np.max(stats['areas']) / np.min(stats['areas']):.2f}")

        print(f"\n  Panel aspect ratios:")
        print(f"    min:    {np.min(stats['ars']):.4f}")
        print(f"    max:    {np.max(stats['ars']):.4f}")
        print(f"    mean:   {np.mean(stats['ars']):.4f}")
        print(f"    std:    {np.std(stats['ars']):.4f}")

        print(f"\n  Min distance: collocation point -> nearest OTHER bound vortex point:")
        print(f"    min:    {np.min(stats['min_cpp_to_bvp']):.6e}")
        print(f"    max:    {np.max(stats['min_cpp_to_bvp']):.6e}")
        print(f"    mean:   {np.mean(stats['min_cpp_to_bvp']):.6e}")

        # Find the panel(s) with smallest distance.
        worst_idx = np.argmin(stats['min_cpp_to_bvp'])
        print(f"    worst panel idx: {worst_idx}")
        print(f"    worst Cpp:   {stats['cpp'][worst_idx]}")
        print(f"    worst dist:  {stats['min_cpp_to_bvp'][worst_idx]:.6e}")

        print(f"\n  Min distance: collocation point -> nearest OTHER collocation point:")
        print(f"    min:    {np.min(stats['min_cpp_to_cpp']):.6e}")
        print(f"    max:    {np.max(stats['min_cpp_to_cpp']):.6e}")
        print(f"    mean:   {np.mean(stats['min_cpp_to_cpp']):.6e}")

        # Check for any near-zero area panels.
        tiny_area_threshold = 1e-14
        tiny_panels = np.sum(stats['areas'] < tiny_area_threshold)
        print(f"\n  Panels with area < {tiny_area_threshold}: {tiny_panels}")

        # Check for any near-zero distances.
        tiny_dist_threshold = 1e-10
        tiny_dists = np.sum(stats['min_cpp_to_bvp'] < tiny_dist_threshold)
        print(f"  Cpp-to-BVP distances < {tiny_dist_threshold}: {tiny_dists}")

        # Check for duplicate or nearly-coincident collocation points
        # (could indicate panels from left/right wing overlapping).
        tiny_cpp_threshold = 1e-10
        tiny_cpp = np.sum(stats['min_cpp_to_cpp'] < tiny_cpp_threshold)
        print(f"  Cpp-to-Cpp distances < {tiny_cpp_threshold}: {tiny_cpp}")

        print()


if __name__ == "__main__":
    main()
