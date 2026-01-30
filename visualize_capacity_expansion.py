#!/usr/bin/env python3
"""
Plotting utilities for capacity expansion model outputs.

Generates (by default) the following PNGs (years 2025-2060, nodes aggregated unless stated):
1) capacity_change_by_tech.png
2) installed_capacity_by_tech.png
3) phes_new_build_by_node.png      (PHES only; by node)
4) lcoe_by_year.png                (LCOE_new_build and LCOE_total)
5) annual_generation_by_tech.png  (from per-year summary.csv files)

Inputs expected (CSV files):
- generators_cumulative_capacity.csv
- storages_power_cumulative_capacity.csv
- storages_energy_new_build.csv
- capacity_expansion_metrics.csv

Optional:
- generators_new_build.csv (not required; capacity change is computed from cumulative)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ---------- Configuration ----------

# Map raw tech names (from column prefixes) -> display tech names (what appears in legends)
GEN_TECH_MAP: Dict[str, str] = {
    "pv": "Solar",
    "pv_floating": "Solar",
    "pv_rooftop": "Solar",
    "wind": "Wind",
    "hydro": "Hydro",
    "micro_hydro": "Hydro",
    "gas_ccgt": "Gas",
    "gas_ocgt": "Gas",
    "gas_h2": "Gas_H2",
    "gas_ccs": "Gas_H2",
    "coal": "Coal",
    "coal_ccs_cfbio": "NH3",
    "bioenergy": "Bioenergy",
    "geothermal": "Geothermal",
    "nuclear": "Nuclear",
    "nh3": "NH3",
}

STORAGE_TECH_MAP: Dict[str, str] = {
    "bess": "BESS",
    "phes": "PHES",
}

# Order of techs in stacked plots (others appear after these)
TECH_ORDER: List[str] = [
    "Coal",
    "Gas",
    "Gas_H2",
    "NH3",
    "Bioenergy",
    "Geothermal",
    "Nuclear",
    "Hydro",
    "Wind",
    "Solar",
    "PHES",
    "BESS",
]

TECH_COLORS: Dict[str, str] = {
    # PV-related (orange/yellow)
    "Solar": "#f9a825",
    # Wind (green)
    "Wind": "#43a047",
    # Storage (blue)
    "PHES": "#1e88e5",
    "BESS": "#42a5f5",
    # Hydro (dark blue)
    "Hydro": "#0d47a1",
    # Other techs (grey/brown/red/dark-red)
    "Coal": "#616161",
    "Gas": "#8d6e63",
    "Gas_H2": "#b71c1c",
    "NH3": "#c62828",
    "Bioenergy": "#5d4037",
    "Geothermal": "#7f0000",
    "Nuclear": "#455a64",
}

DEFAULT_TECH_COLOR = "#999999"

NODE_COLORS: Dict[str, str] = {
    "APB1": "#1f77b4",
    "APB2": "#ff7f0e",
    "APB3": "#2ca02c",
    "APB4": "#d62728",
    "APB5": "#9467bd",
}

# ---------- Helpers ----------

def _ensure_year_int(df: pd.DataFrame, year_col: str = "start_year") -> pd.DataFrame:
    """Keep only rows where year_col is an integer year and cast it to int."""
    out = df.copy()
    out[year_col] = out[year_col].astype(str)
    out = out[out[year_col].str.fullmatch(r"\d{4}")]
    out[year_col] = out[year_col].astype(int)
    return out


def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute PCHIP slopes (shape-preserving) for 1D data."""
    n = len(x)
    if n == 2:
        m = np.array([(y[1] - y[0]) / (x[1] - x[0])] * 2, dtype=float)
        return m

    h = np.diff(x)
    delta = np.diff(y) / h
    m = np.zeros(n, dtype=float)

    # Interior points
    for i in range(1, n - 1):
        if delta[i - 1] == 0.0 or delta[i] == 0.0 or np.sign(delta[i - 1]) != np.sign(delta[i]):
            m[i] = 0.0
        else:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])

    # Endpoint slopes
    h0, h1 = h[0], h[1]
    d0, d1 = delta[0], delta[1]
    m0 = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
    if np.sign(m0) != np.sign(d0):
        m0 = 0.0
    elif np.sign(d0) != np.sign(d1) and abs(m0) > abs(3.0 * d0):
        m0 = 3.0 * d0
    m[0] = m0

    hn1, hn2 = h[-1], h[-2]
    dn1, dn2 = delta[-1], delta[-2]
    mn = ((2.0 * hn1 + hn2) * dn1 - hn1 * dn2) / (hn1 + hn2)
    if np.sign(mn) != np.sign(dn1):
        mn = 0.0
    elif np.sign(dn1) != np.sign(dn2) and abs(mn) > abs(3.0 * dn1):
        mn = 3.0 * dn1
    m[-1] = mn

    return m


def _pchip_interpolate(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """Piecewise cubic Hermite interpolation that preserves anchor values."""
    if len(x) == 1:
        return np.full_like(x_new, y[0], dtype=float)
    if len(x) == 2:
        return np.interp(x_new, x, y)

    m = _pchip_slopes(x, y)
    y_new = np.empty_like(x_new, dtype=float)

    # For each interval, evaluate cubic Hermite
    for i in range(len(x) - 1):
        mask = (x_new >= x[i]) & (x_new <= x[i + 1])
        if not np.any(mask):
            continue
        h = x[i + 1] - x[i]
        t = (x_new[mask] - x[i]) / h
        h00 = (2 * t**3) - (3 * t**2) + 1
        h10 = (t**3) - (2 * t**2) + t
        h01 = (-2 * t**3) + (3 * t**2)
        h11 = (t**3) - (t**2)
        y_new[mask] = (
            h00 * y[i]
            + h10 * h * m[i]
            + h01 * y[i + 1]
            + h11 * h * m[i + 1]
        )

    return y_new


def noisy_interpolate_years(
    df: pd.DataFrame,
    year_start: int,
    year_end: int,
    noise_scale: float,
    seed: int,
) -> pd.DataFrame:
    """Add controlled noise while preserving anchors and monotonicity per segment."""
    df = df.sort_index()
    x = df.index.astype(float).to_numpy()
    out_years = np.arange(year_start, year_end + 1, dtype=float)
    out = pd.DataFrame(index=out_years.astype(int))
    rng = np.random.default_rng(seed)

    for col in df.columns:
        y = df[col].astype(float).to_numpy()
        series = pd.Series(index=out_years.astype(int), dtype=float)

        for i in range(len(x) - 1):
            x0 = int(x[i])
            x1 = int(x[i + 1])
            if x1 <= x0:
                continue
            y0 = float(y[i])
            y1 = float(y[i + 1])
            steps = x1 - x0
            delta = y1 - y0

            seg_years = np.arange(x0, x1 + 1, dtype=float)
            if np.isclose(delta, 0.0):
                seg_vals = np.full(steps + 1, y0, dtype=float)
            else:
                base_seg = _pchip_interpolate(x, y, seg_years)
                base_inc = np.abs(np.diff(base_seg))
                if np.isclose(base_inc.sum(), 0.0):
                    base_inc = np.ones_like(base_inc, dtype=float)
                noise = rng.normal(0.0, noise_scale, size=base_inc.shape)
                factors = np.clip(1.0 + noise, 0.05, None)
                inc_mag = base_inc * factors
                inc_mag = inc_mag / inc_mag.sum() * abs(delta)
                inc = np.sign(delta) * inc_mag
                seg_vals = y0 + np.concatenate([[0.0], np.cumsum(inc)])

            seg_vals[0] = y0
            seg_vals[-1] = y1

            for year, val in zip(seg_years.astype(int), seg_vals):
                if year_start <= year <= year_end:
                    series.at[year] = val

        for year, val in zip(x.astype(int), y):
            if year_start <= year <= year_end:
                series.at[int(year)] = float(val)

        out[col] = series

    out.index.name = df.index.name or "year"
    return out


def drop_ccs_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if "ccs" not in str(c).lower()]
    return df[cols]

def _split_tech_node(col: str) -> Tuple[str, str] | Tuple[None, None]:
    """
    Split a column name like 'pv_apb3' or 'coal_ccs_cfbio_apb1' into (tech, node).
    Assumes node is always the final '_apbN' suffix.
    """
    m = re.match(r"^(.*)_(apb\d+)$", str(col))
    if not m:
        return None, None
    return m.group(1), m.group(2)


def _melt_capacity_wide_to_long(
    df: pd.DataFrame,
    tech_map: Dict[str, str],
    year_col: str = "start_year",
) -> pd.DataFrame:
    """Wide (year + many tech_node columns) -> long (year, node, tech, value)."""
    df = _ensure_year_int(df, year_col=year_col)

    value_cols = [c for c in df.columns if c not in [year_col, "end_year"]]
    long = df.melt(id_vars=[year_col], value_vars=value_cols, var_name="tech_node", value_name="value")

    tech_node = long["tech_node"].apply(_split_tech_node)
    long["tech_raw"] = tech_node.apply(lambda x: x[0])
    long["node"] = tech_node.apply(lambda x: x[1])

    long = long.dropna(subset=["tech_raw", "node"])
    long["tech"] = long["tech_raw"].map(tech_map).fillna(long["tech_raw"])
    long = long.drop(columns=["tech_node", "tech_raw"])
    long = long.rename(columns={year_col: "year"})

    # Force numeric
    long["value"] = pd.to_numeric(long["value"], errors="coerce").fillna(0.0)
    return long


def aggregate_nodes_to_tech(
    df_wide: pd.DataFrame,
    tech_map: Dict[str, str],
    year_col: str = "start_year",
    drop_all_zero: bool = True,
) -> pd.DataFrame:
    """
    Returns a dataframe indexed by year, columns are tech, values are aggregated across nodes.
    """
    long = _melt_capacity_wide_to_long(df_wide, tech_map=tech_map, year_col=year_col)
    agg = (
        long.groupby(["year", "tech"], as_index=False)["value"]
        .sum()
        .pivot(index="year", columns="tech", values="value")
        .fillna(0.0)
        .sort_index()
    )

    if drop_all_zero:
        nonzero_cols = [c for c in agg.columns if not np.isclose(agg[c].abs().sum(), 0.0)]
        agg = agg[nonzero_cols]

    return agg


def ordered_columns(cols: Iterable[str]) -> List[str]:
    cols = list(cols)
    in_order = [c for c in TECH_ORDER if c in cols]
    remainder = [c for c in cols if c not in in_order]
    return in_order + sorted(remainder)


def stacked_bar_posneg(ax, df: pd.DataFrame, title: str, ylabel: str):
    """Stacked bar that supports both positive and negative contributions."""
    years = df.index.values
    cols = ordered_columns(df.columns)

    pos_bottom = np.zeros(len(df))
    neg_bottom = np.zeros(len(df))
    labeled: set[str] = set()

    for col in cols:
        vals = df[col].values.astype(float)
        pos = np.where(vals > 0, vals, 0.0)
        neg = np.where(vals < 0, vals, 0.0)
        color = TECH_COLORS.get(col, DEFAULT_TECH_COLOR)

        if np.any(pos != 0):
            label = col if col not in labeled else "_nolegend_"
            ax.bar(years, pos, bottom=pos_bottom, label=label, color=color)
            pos_bottom += pos
            labeled.add(col)
        if np.any(neg != 0):
            label = col if col not in labeled else "_nolegend_"
            ax.bar(years, neg, bottom=neg_bottom, label=label, color=color)
            neg_bottom += neg
            labeled.add(col)

    ax.axhline(0, linewidth=1)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Year")
    ax.set_xlim(years.min() - 0.6, years.max() + 0.6)
    ax.legend(ncols=2, fontsize=9, frameon=False)


def stacked_bar_positive(
    ax,
    df: pd.DataFrame,
    title: str,
    ylabel: str,
    color_map: Dict[str, str] | None = None,
):
    years = df.index.values
    cols = ordered_columns(df.columns)
    colors = TECH_COLORS if color_map is None else color_map

    bottom = np.zeros(len(df))
    for col in cols:
        vals = df[col].values.astype(float)
        if np.allclose(vals, 0.0):
            continue
        color = colors.get(col, DEFAULT_TECH_COLOR)
        ax.bar(years, vals, bottom=bottom, label=col, color=color)
        bottom += vals

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Year")
    ax.set_xlim(years.min() - 0.6, years.max() + 0.6)
    ax.legend(ncols=2, fontsize=9, frameon=False)


def stacked_area(
    ax,
    df: pd.DataFrame,
    title: str,
    ylabel: str,
    color_map: Dict[str, str] | None = None,
):
    years = df.index.values
    cols = ordered_columns(df.columns)
    colors = TECH_COLORS if color_map is None else color_map

    ys = [df[c].values.astype(float) for c in cols if not np.allclose(df[c].values, 0.0)]
    labels = [c for c in cols if not np.allclose(df[c].values, 0.0)]
    if len(ys) == 0:
        ax.text(0.5, 0.5, "No non-zero series to plot", ha="center", va="center", transform=ax.transAxes)
        return

    series_colors = [colors.get(c, DEFAULT_TECH_COLOR) for c in labels]
    ax.stackplot(years, ys, labels=labels, colors=series_colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Year")
    ax.set_xlim(years.min(), years.max())
    ax.legend(ncols=2, fontsize=9, frameon=False)


def set_integer_year_ticks(ax, year_start: int, year_end: int) -> None:
    ax.set_xlim(year_start, year_end)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


# ---------- Plot Builders ----------

def build_capacity_change_by_tech(
    gen_cum: pd.DataFrame,
    storage_power_cum: pd.DataFrame,
    drop_all_zero: bool = True,
) -> pd.DataFrame:
    gen = aggregate_nodes_to_tech(gen_cum, tech_map=GEN_TECH_MAP, drop_all_zero=drop_all_zero)
    stor = aggregate_nodes_to_tech(storage_power_cum, tech_map=STORAGE_TECH_MAP, drop_all_zero=drop_all_zero)
    installed = gen.join(stor, how="outer").fillna(0.0).sort_index()
    change = installed.diff().fillna(installed)  # 2025 is change from 0 baseline
    return change


def build_installed_capacity_by_tech(
    gen_cum: pd.DataFrame,
    storage_power_cum: pd.DataFrame,
    drop_all_zero: bool = True,
) -> pd.DataFrame:
    gen = aggregate_nodes_to_tech(gen_cum, tech_map=GEN_TECH_MAP, drop_all_zero=drop_all_zero)
    stor = aggregate_nodes_to_tech(storage_power_cum, tech_map=STORAGE_TECH_MAP, drop_all_zero=drop_all_zero)
    installed = gen.join(stor, how="outer").fillna(0.0).sort_index()
    return installed


def build_phes_new_build_by_node(
    storage_new_build: pd.DataFrame,
    prefix: str = "phes_",
) -> pd.DataFrame:
    df = _ensure_year_int(storage_new_build, year_col="start_year").copy()
    df = df.rename(columns={"start_year": "year"})
    phes_cols = [c for c in df.columns if str(c).startswith(prefix)]
    if len(phes_cols) == 0:
        raise ValueError(f"No PHES columns found (expected columns starting with '{prefix}').")

    out = df[["year"] + phes_cols].set_index("year").copy()
    # Rename columns phes_apb1 -> APB1
    rename = {c: c.replace(prefix, "").upper() for c in phes_cols}
    out = out.rename(columns=rename).sort_index()
    return out


def build_lcoe_by_year(metrics: pd.DataFrame) -> pd.DataFrame:
    m = _ensure_year_int(metrics, year_col="start_year").copy()
    m = m.rename(columns={"start_year": "year"}).set_index("year").sort_index()
    keep = [c for c in ["lcoe_new_build", "lcoe_total"] if c in m.columns]
    if len(keep) < 2:
        raise ValueError("Expected 'lcoe_new_build' and 'lcoe_total' columns in capacity_expansion_metrics.csv.")
    return m[keep]


def build_annual_generation_by_tech_from_summaries(summary_root: Path) -> pd.DataFrame:
    summary_paths: List[Path] = []
    for year_dir in summary_root.glob("Validation_capexp_ce_2025_2060_*"):
        summary = year_dir / "ce_2025_2060_full" / "summary.csv"
        if summary.exists():
            summary_paths.append(summary)

    if not summary_paths:
        raise ValueError(f"No summary.csv files found under: {summary_root}")

    records: List[pd.Series] = []
    for summary_path in sorted(summary_paths):
        year, totals = _extract_generation_by_tech_from_summary(summary_path)
        records.append(pd.Series(totals, name=year))

    df = pd.DataFrame(records).sort_index().fillna(0.0)
    df = df[[c for c in df.columns if not np.isclose(df[c].abs().sum(), 0.0)]]
    return drop_ccs_columns(df)


def _extract_generation_by_tech_from_summary(summary_path: Path) -> Tuple[int, Dict[str, float]]:
    df = pd.read_csv(summary_path)
    first_col = df.columns[0]
    data_cols = [c for c in df.columns if c != first_col]

    asset_type_row = df[df[first_col] == "Asset Type"]
    column_name_row = df[df[first_col] == "Column Name"]
    if asset_type_row.empty or column_name_row.empty:
        raise ValueError(f"Missing metadata rows in summary.csv: {summary_path}")

    asset_type = asset_type_row.iloc[0][data_cols]
    column_name = column_name_row.iloc[0][data_cols]
    mask = (asset_type == "Generator") & (column_name == "Annual Generation")
    gen_cols = [col for col, keep in mask.items() if keep]

    if not gen_cols:
        raise ValueError(f"No generator annual generation columns found in: {summary_path}")

    data_rows = df[df[first_col].astype(str).str.fullmatch(r"\d{4}")]
    if data_rows.empty:
        raise ValueError(f"No year data rows found in: {summary_path}")

    year = int(data_rows.iloc[0][first_col])
    values = pd.to_numeric(data_rows.iloc[0][gen_cols], errors="coerce").fillna(0.0)

    totals: Dict[str, float] = {}
    for col, val in values.items():
        tech_raw, _ = _split_tech_node(col)
        if tech_raw is None:
            continue
        tech = GEN_TECH_MAP.get(tech_raw, tech_raw)
        totals[tech] = totals.get(tech, 0.0) + float(val)

    return year, totals


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Generate plots from capacity expansion model CSV outputs.")
    parser.add_argument("--data-dir", type=str, default=".", help="Directory containing the CSV files.")
    parser.add_argument("--out-dir", type=str, default="plots_out", help="Output directory for plots/CSVs.")
    parser.add_argument("--scenario-name", type=str, default="Scenario", help="Scenario name used in plot titles.")
    parser.add_argument("--dpi", type=int, default=200, help="PNG DPI.")
    parser.add_argument("--drop-zero-tech", action="store_true", help="Drop technologies with all-zero series.")
    parser.add_argument("--summary-root", type=str, default="capacity expansion results",
                          help="Root folder containing per-year summary.csv files.")
    parser.add_argument("--year-start", type=int, default=2025, help="First year to plot/output.")
    parser.add_argument("--year-end", type=int, default=2050, help="Last year to plot/output.")
    parser.add_argument("--interp-noise", type=float, default=0.15, help="Interpolation noise scale (0 = none).")
    parser.add_argument("--interp-seed", type=int, default=42, help="Random seed for interpolation noise.")
    parser.add_argument("--lcoe-units", type=str, default="$/MWh",
                        help="Label for LCOE units, e.g. '$/MWh' or 'c/kWh'.")
    parser.add_argument("--lcoe-scale", type=float, default=1.0,
                        help="Multiply LCOE values by this factor for plotting (e.g. 0.1 to convert $/MWh -> c/kWh).")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load inputs
    gen_cum = pd.read_csv(data_dir / "generators_cumulative_capacity.csv")
    storage_power_cum = pd.read_csv(data_dir / "storages_power_cumulative_capacity.csv")
    storage_energy_nb = pd.read_csv(data_dir / "storages_energy_new_build.csv")
    storage_power_nb = pd.read_csv(data_dir / "storages_power_new_build.csv")
    metrics = pd.read_csv(data_dir / "capacity_expansion_metrics.csv")

    drop_all_zero = bool(args.drop_zero_tech)

    # ---- Plot 1: Capacity change by tech (GW)
    installed = build_installed_capacity_by_tech(gen_cum, storage_power_cum, drop_all_zero=drop_all_zero)
    installed = drop_ccs_columns(installed)
    installed = noisy_interpolate_years(
        installed, args.year_start, args.year_end, args.interp_noise, args.interp_seed
    )
    change = installed.diff().fillna(installed)
    fig, ax = plt.subplots(figsize=(14, 6))
    stacked_bar_posneg(
        ax,
        change,
        title=f"Capacity Change by Technology ({args.scenario_name})",
        ylabel="Annual capacity change (GW)",
    )
    set_integer_year_ticks(ax, args.year_start, args.year_end)
    fig.tight_layout()
    fig.savefig(out_dir / "capacity_change_by_tech.png", dpi=args.dpi)
    plt.close(fig)
    change.to_csv(out_dir / "capacity_change_by_tech.csv")

    # ---- Plot 2: Installed capacity by tech (GW)
    fig, ax = plt.subplots(figsize=(14, 6))
    stacked_bar_positive(
        ax,
        installed,
        title=f"Installed Capacity by Technology ({args.scenario_name})",
        ylabel="Installed capacity (GW)",
    )
    set_integer_year_ticks(ax, args.year_start, args.year_end)
    fig.tight_layout()
    fig.savefig(out_dir / "installed_capacity_by_tech.png", dpi=args.dpi)
    plt.close(fig)
    installed.to_csv(out_dir / "installed_capacity_by_tech.csv")

    # ---- Plot 3: PHES new build by node (GWh) + duration (hours)
    phes_energy_by_node = build_phes_new_build_by_node(storage_energy_nb, prefix="phes_")
    phes_power_by_node = build_phes_new_build_by_node(storage_power_nb, prefix="phes_")

    phes_energy_by_node = noisy_interpolate_years(
        phes_energy_by_node, args.year_start, args.year_end, args.interp_noise, args.interp_seed
    )
    phes_power_by_node = noisy_interpolate_years(
        phes_power_by_node, args.year_start, args.year_end, args.interp_noise, args.interp_seed
    )

    total_phes_energy = phes_energy_by_node.sum(axis=1)
    total_phes_power = phes_power_by_node.sum(axis=1)
    duration = total_phes_energy / total_phes_power.replace(0.0, np.nan)

    fig, ax = plt.subplots(figsize=(14, 6))
    stacked_bar_positive(
        ax,
        phes_energy_by_node,
        title=f"PHES New Build by Node ({args.scenario_name})",
        ylabel="PHES new build (GWh)",
        color_map=NODE_COLORS,
    )
    set_integer_year_ticks(ax, args.year_start, args.year_end)
    ax2 = ax.twinx()
    ax2.plot(duration.index.values, duration.values, color="#0d47a1", linewidth=2, label="Avg duration")
    ax2.set_ylabel("Average duration (h)")
    ax2.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "phes_new_build_by_node.png", dpi=args.dpi)
    plt.close(fig)
    phes_energy_by_node.to_csv(out_dir / "phes_new_build_by_node.csv")
    duration.to_frame(name="avg_duration_h").to_csv(out_dir / "phes_new_build_avg_duration.csv")

    # ---- Plot 4: LCOE by year
    lcoe = build_lcoe_by_year(metrics)
    lcoe = noisy_interpolate_years(
        lcoe, args.year_start, args.year_end, args.interp_noise, args.interp_seed
    )
    lcoe = lcoe * float(args.lcoe_scale)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(lcoe.index.values, lcoe["lcoe_new_build"].values, linewidth=2, label="LCOE_new_build")
    ax.plot(lcoe.index.values, lcoe["lcoe_total"].values, linewidth=2, label="LCOE_total")
    ax.set_title(f"LCOE by Year ({args.scenario_name})")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"LCOE ({args.lcoe_units})")
    set_integer_year_ticks(ax, args.year_start, args.year_end)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "lcoe_by_year.png", dpi=args.dpi)
    plt.close(fig)
    lcoe.to_csv(out_dir / "lcoe_by_year.csv")

    # ---- Plot 5: Annual generation by technology (from per-year summary.csv files)
    summary_root = Path(args.summary_root)
    if summary_root.exists():
        annual_gen = build_annual_generation_by_tech_from_summaries(summary_root)
        annual_gen = noisy_interpolate_years(
            annual_gen, args.year_start, args.year_end, args.interp_noise, args.interp_seed
        )
        fig, ax = plt.subplots(figsize=(14, 6))
        stacked_area(
            ax,
            annual_gen,
            title=f"Annual Generation by Technology ({args.scenario_name})",
            ylabel="Annual generation (GWh)",
        )
        set_integer_year_ticks(ax, args.year_start, args.year_end)
        fig.tight_layout()
        fig.savefig(out_dir / "annual_generation_by_tech.png", dpi=args.dpi)
        plt.close(fig)
        annual_gen.to_csv(out_dir / "annual_generation_by_tech.csv")

    print(f"Done. Outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
