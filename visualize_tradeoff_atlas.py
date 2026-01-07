#!/usr/bin/env python3
"""
Trade-off atlas for broad-optimum solutions.

Goal
----
Show *trade-offs between any two grouped decision-variable categories* (PV/Wind/PHES/BESS/HVDC)
while also communicating that LCOE stays within a narrow band.

This script focuses on speed by default:
  - Always creates a scatter-matrix across all groups (one figure that shows *all pairs*).
  - Always creates a substitution/association heatmap (negative correlations suggest substitution).
  - Optionally creates one PNG per pair (a "trade-off atlas") via --all-pairs (can be slow).

Inputs
------
- near_optimal_bands.csv
- near_optimal_space_snapshot.csv
- diversify_space.csv
- optimal_x_default.csv  (single row, no header; raw build vector)

Run
---
Put this script in the same folder as the CSVs and run:

  python visualize_tradeoff_atlas.py

Optional (slow):
  python visualize_tradeoff_atlas.py --all-pairs

Optional (targeted pairs only):
  python visualize_tradeoff_atlas.py --pairs HVDC,PHES_energy HVDC,Battery_energy

Outputs
-------
Creates folder: tradeoff_atlas_plots/
Also creates:
- tradeoff_atlas_plots/conditional (conditional envelopes for primary axes)
- tradeoff_atlas_plots/density (pairwise density maps for primary axes)
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Defaults (edit if needed)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BANDS = os.path.join(BASE_DIR, "results", "near_optimum", "default", "near_optimal_bands.csv")
DEFAULT_NEAR = os.path.join(BASE_DIR, "results", "near_optimum", "default", "near_optimal_space.csv")
DEFAULT_DIV = os.path.join(BASE_DIR, "results", "diversify", "default", "diversify_space.csv")
DEFAULT_OPT_X = os.path.join(BASE_DIR, "results", "temp", "optimal_x_default.csv")
DEFAULT_STORAGES = os.path.join(BASE_DIR, "inputs", "config", "storages.csv")
DEFAULT_OUTDIR = os.path.join(BASE_DIR, "tradeoff_atlas_plots")

# -----------------------------
# Column metadata
# -----------------------------
META_NEAR = {"Group", "Band_Type", "LCOE [$/MWh]", "Operational_Penalty", "Band_Penalty"}
META_DIV  = {"LCOE [$/MWh]", "Operational_Penalty", "Band_Penalty", "Scaled_Novelty"}

# -----------------------------
# Helpers
# -----------------------------
def infer_build_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_NEAR]

def build_group_map(build_cols: list[str]) -> dict[str, list[str]]:
    def starts(prefix):
        return [c for c in build_cols if c.lower().startswith(prefix)]

    pv   = starts("pv_")
    wind = starts("wind_")
    phes_p = [c for c in build_cols if c.lower().startswith("phes_") and c.lower().endswith("_power")]
    phes_e = [c for c in build_cols if c.lower().startswith("phes_") and c.lower().endswith("_energy")]
    bess_p = [c for c in build_cols if c.lower().startswith("bess_") and c.lower().endswith("_power")]
    bess_e = [c for c in build_cols if c.lower().startswith("bess_") and c.lower().endswith("_energy")]

    hvdc = [c for c in build_cols if ("-" in c) and (not c.lower().endswith(("_power", "_energy")))]

    used = set(pv + wind + phes_p + phes_e + bess_p + bess_e + hvdc)
    other = [c for c in build_cols if c not in used]

    group_map = {
        "PV": pv,
        "Wind": wind,
        "PHES_power": phes_p,
        "PHES_energy": phes_e,
        "Battery_power": bess_p,
        "Battery_energy": bess_e,
        "HVDC": hvdc,
    }
    if other:
        group_map["Other"] = other

    return {k: v for k, v in group_map.items() if len(v) > 0}

def add_group_totals(df: pd.DataFrame, group_map: dict[str, list[str]], prefix="G_") -> pd.DataFrame:
    out = df.copy()
    for g, cols in group_map.items():
        out[prefix + g] = out[cols].sum(axis=1)
    return out

def add_primary_totals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["G_Storage_power"] = out.get("G_PHES_power", 0.0) + out.get("G_Battery_power", 0.0)
    out["G_Storage_energy"] = out.get("G_PHES_energy", 0.0) + out.get("G_Battery_energy", 0.0)
    return out

def load_storage_meta(path: str) -> dict[str, tuple[float, float]]:
    if not path or not os.path.isfile(path):
        print(f"Warning: storages.csv not found at {path}. Skipping energy rendering.")
        return {}
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"Warning: failed to read storages.csv ({path}): {exc}. Skipping energy rendering.")
        return {}

    required = {"name", "duration", "initial_power_capacity"}
    if not required.issubset(df.columns):
        print(f"Warning: storages.csv missing columns {required - set(df.columns)}. Skipping energy rendering.")
        return {}

    df = df.copy()
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0)
    df["initial_power_capacity"] = pd.to_numeric(df["initial_power_capacity"], errors="coerce").fillna(0)
    return {
        row["name"]: (float(row["duration"]), float(row["initial_power_capacity"]))
        for _, row in df.iterrows()
    }

def render_storage_energy(build_series: pd.Series, storage_meta: dict[str, tuple[float, float]]) -> pd.Series:
    if not storage_meta:
        return build_series
    out = build_series.copy()
    for name, (duration, initial_power) in storage_meta.items():
        if duration <= 0:
            continue
        power_col = f"{name}_power"
        energy_col = f"{name}_energy"
        if power_col in out.index and energy_col in out.index:
            out[energy_col] = (initial_power + float(out[power_col])) * duration
    return out

def load_optimum_vector(path_opt_x: str, build_cols: list[str], storage_meta: dict[str, tuple[float, float]]) -> pd.Series:
    x = pd.read_csv(path_opt_x, header=None).iloc[0].to_numpy()
    x = x[~pd.isna(x)]
    if len(x) != len(build_cols):
        raise ValueError(
            f"Optimal vector has {len(x)} entries, but build_cols has {len(build_cols)}. "
            f"Check file/scenario mismatch."
        )
    series = pd.Series(x, index=build_cols, name="optimum")
    return render_storage_energy(series, storage_meta)

def robust_log1p(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    return np.log1p(np.maximum(arr, 0.0))

def scatter_matrix(df: pd.DataFrame,
                   cols: list[str],
                   opt_point: np.ndarray,
                   title: str,
                   outpath: str,
                   sample_n: int = 15000,
                   transform: str = "linear"):
    d = df.dropna(subset=cols + ["dLCOE_pct"]).copy()
    if len(d) > sample_n:
        d = d.sample(sample_n, random_state=0)

    X = d[cols].to_numpy(dtype=float)
    c = d["dLCOE_pct"].to_numpy(dtype=float)

    if transform == "log1p":
        Xp = robust_log1p(X)
        optp = robust_log1p(opt_point[None, :])[0]
        axis_label = "log1p(total)"
    else:
        Xp = X
        optp = opt_point
        axis_label = "total"

    n = len(cols)
    fig, axes = plt.subplots(n, n, figsize=(2.2*n, 2.2*n), sharex="col", sharey="row")
    vmin, vmax = np.nanpercentile(c, 1), np.nanpercentile(c, 99)

    sc = None
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                ax.hist(Xp[:, j], bins=35, alpha=0.75)
                ax.axvline(optp[j], linestyle="--", linewidth=2)
            else:
                sc = ax.scatter(Xp[:, j], Xp[:, i], c=c, s=7, alpha=0.55, vmin=vmin, vmax=vmax)
                ax.scatter(optp[j], optp[i], marker="*", s=160)
            if i == n - 1:
                ax.set_xlabel(cols[j].replace("G_", "") + f"\n({axis_label})")
            if j == 0:
                ax.set_ylabel(cols[i].replace("G_", "") + f"\n({axis_label})")
            ax.grid(True, alpha=0.15)

    fig.suptitle(title, y=0.995)
    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.7, pad=0.01)
        cbar.set_label("ΔLCOE % (relative to best observed)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)

def pair_tradeoff_plot(df: pd.DataFrame,
                       xcol: str,
                       ycol: str,
                       opt_xy: tuple[float, float],
                       outpath: str,
                       envelope: bool = True,
                       bins: int = 18,
                       sample_n: int = 20000,
                       transform: str = "linear",
                       low_cost_cut_pct: float = 10.0):
    d = df.dropna(subset=[xcol, ycol, "dLCOE_pct"]).copy()
    if len(d) > sample_n:
        d = d.sample(sample_n, random_state=1)

    x = d[xcol].to_numpy(dtype=float)
    y = d[ycol].to_numpy(dtype=float)
    c = d["dLCOE_pct"].to_numpy(dtype=float)

    if transform == "log1p":
        xp, yp = robust_log1p(x), robust_log1p(y)
        ox, oy = robust_log1p(np.array([opt_xy[0]]))[0], robust_log1p(np.array([opt_xy[1]]))[0]
        xlab, ylab = f"{xcol.replace('G_','')} (log1p)", f"{ycol.replace('G_','')} (log1p)"
    else:
        xp, yp = x, y
        ox, oy = opt_xy
        xlab, ylab = xcol.replace("G_",""), ycol.replace("G_","")

    fig, ax = plt.subplots(figsize=(7.4, 5.3))
    vmin, vmax = np.nanpercentile(c, 1), np.nanpercentile(c, 99)
    sc = ax.scatter(xp, yp, c=c, s=10, alpha=0.55, vmin=vmin, vmax=vmax)
    ax.scatter(ox, oy, marker="*", s=220, label="optimum")

    if envelope:
        d2 = d[d["dLCOE_pct"] <= low_cost_cut_pct].copy()
        if len(d2) >= 200:
            x2 = d2[xcol].to_numpy(dtype=float)
            y2 = d2[ycol].to_numpy(dtype=float)
            if transform == "log1p":
                x2, y2 = robust_log1p(x2), robust_log1p(y2)

            edges = np.linspace(np.nanmin(x2), np.nanmax(x2), bins + 1)
            mids = 0.5 * (edges[:-1] + edges[1:])
            q10, q50, q90 = [], [], []
            for a, b in zip(edges[:-1], edges[1:]):
                m = (x2 >= a) & (x2 < b)
                yy = y2[m]
                if len(yy) < 10:
                    q10.append(np.nan); q50.append(np.nan); q90.append(np.nan)
                else:
                    q10.append(np.nanpercentile(yy, 10))
                    q50.append(np.nanpercentile(yy, 50))
                    q90.append(np.nanpercentile(yy, 90))
            q10, q50, q90 = np.array(q10), np.array(q50), np.array(q90)

            ax.plot(mids, q50, linewidth=2.2, label=f"median envelope (ΔLCOE%≤{low_cost_cut_pct:g})")
            ax.fill_between(mids, q10, q90, alpha=0.15, label="10–90% band")

    ax.set_title(f"Trade-off: {xlab} vs {ylab}")
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.grid(True, alpha=0.2)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("ΔLCOE % (relative to best observed)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)

def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = pd.Series(a).rank(method="average").to_numpy()
    b = pd.Series(b).rank(method="average").to_numpy()
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def plot_substitution_heatmap(df: pd.DataFrame,
                              cols: list[str],
                              outpath: str,
                              low_cost_cut_pct: float = 10.0,
                              sample_n: int = 50000):
    d = df.dropna(subset=cols + ["dLCOE_pct"]).copy()
    d = d[d["dLCOE_pct"] <= low_cost_cut_pct]
    if len(d) < 200:
        return
    if len(d) > sample_n:
        d = d.sample(sample_n, random_state=3)

    X = d[cols].to_numpy(dtype=float)
    corr = np.zeros((len(cols), len(cols)), dtype=float)
    for i in range(len(cols)):
        for j in range(len(cols)):
            corr[i, j] = spearman_corr(X[:, i], X[:, j])

    fig, ax = plt.subplots(figsize=(0.9*len(cols) + 3, 0.9*len(cols) + 2.5))
    im = ax.imshow(corr, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels([c.replace("G_","") for c in cols], rotation=35, ha="right")
    ax.set_yticklabels([c.replace("G_","") for c in cols])
    ax.set_title(f"Spearman correlation of group ratios (ΔLCOE%≤{low_cost_cut_pct:g})")
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Spearman ρ")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)

# -----------------------------
# Primary axes views
# -----------------------------
def _sample_df(df: pd.DataFrame, cols: list[str], sample_n: int | None, seed: int) -> pd.DataFrame:
    d = df.dropna(subset=cols).copy()
    if sample_n and len(d) > sample_n:
        d = d.sample(sample_n, random_state=seed)
    return d

def _binned_stats(x: np.ndarray, y: np.ndarray, bins: int):
    edges = np.linspace(np.nanmin(x), np.nanmax(x), bins + 1)
    ymin, y50, ymax = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b)
        yy = y[m]
        if len(yy) == 0:
            ymin.append(np.nan); y50.append(np.nan); ymax.append(np.nan)
        else:
            ymin.append(np.nanmin(yy))
            y50.append(np.nanpercentile(yy, 50))
            ymax.append(np.nanmax(yy))
    return edges, np.array(ymin), np.array(y50), np.array(ymax)

def plot_conditional_grid(
    df: pd.DataFrame,
    primary_cols: list[str],
    opt_point: dict[str, float],
    outdir: str,
    bins: int = 20,
):
    df = _sample_df(df, primary_cols, sample_n=None, seed=10)
    os.makedirs(outdir, exist_ok=True)
    for x_col in primary_cols:
        y_cols = [c for c in primary_cols if c != x_col]
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
        axes = axes.ravel()
        for ax, y_col in zip(axes, y_cols):
            x = df[x_col].to_numpy(dtype=float)
            y = df[y_col].to_numpy(dtype=float)
            opt_x = opt_point.get(x_col, np.nan)
            opt_y = opt_point.get(y_col, np.nan)
            if np.isfinite(opt_x) and np.isfinite(opt_y):
                x = np.append(x, opt_x)
                y = np.append(y, opt_y)
            edges, y_min, y_med, y_max = _binned_stats(x, y, bins)
            ax.step(edges[:-1], y_med, where="post", linewidth=2.0, label="median")
            ax.fill_between(edges[:-1], y_min, y_max, step="post", alpha=0.2, label="min-max")
            ax.scatter(opt_x, opt_y, marker="*", s=140, label="optimum")
            ax.set_title(f"{y_col.replace('G_','')} vs {x_col.replace('G_','')}")
            ax.grid(True, alpha=0.2)
            ax.set_xlabel(x_col.replace("G_", ""))
            ax.set_ylabel(y_col.replace("G_", ""))

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        fig.suptitle(f"Conditional envelopes vs {x_col.replace('G_','')}")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(outdir, f"conditional_{x_col.replace('G_','')}.png"), dpi=220)
        plt.close(fig)

def plot_density_pairs(
    df: pd.DataFrame,
    primary_cols: list[str],
    opt_point: dict[str, float],
    outdir: str,
    gridsize: int = 40,
    sample_n: int = 80000,
):
    os.makedirs(outdir, exist_ok=True)
    df_full = _sample_df(df, primary_cols, sample_n=None, seed=11)
    df = _sample_df(df, primary_cols, sample_n, seed=11)
    for i in range(len(primary_cols)):
        for j in range(i + 1, len(primary_cols)):
            x_col, y_col = primary_cols[i], primary_cols[j]
            x = df[x_col].to_numpy(dtype=float)
            y = df[y_col].to_numpy(dtype=float)
            fig, ax = plt.subplots(figsize=(7.6, 5.6))
            extent = [
                float(df_full[x_col].min()),
                float(df_full[x_col].max()),
                float(df_full[y_col].min()),
                float(df_full[y_col].max()),
            ]
            hb = ax.hexbin(x, y, gridsize=gridsize, bins="log", mincnt=1, extent=extent)
            ax.scatter(opt_point.get(x_col, np.nan), opt_point.get(y_col, np.nan), marker="*", s=180, color="white")
            ax.set_xlabel(x_col.replace("G_", ""))
            ax.set_ylabel(y_col.replace("G_", ""))
            ax.set_title(f"Density: {y_col.replace('G_','')} vs {x_col.replace('G_','')}")
            fig.colorbar(hb, ax=ax, label="log10(count)")
            fig.tight_layout()
            fig.savefig(
                os.path.join(outdir, f"density_{y_col.replace('G_','')}_vs_{x_col.replace('G_','')}.png"),
                dpi=220,
            )
            plt.close(fig)

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", default=DEFAULT_BANDS, help="Path to near_optimal_bands.csv")
    ap.add_argument("--near", default=DEFAULT_NEAR, help="Path to near_optimal_space.csv")
    ap.add_argument("--div", default=DEFAULT_DIV, help="Path to diversify_space.csv")
    ap.add_argument("--opt-x", default=DEFAULT_OPT_X, help="Path to optimal_x_<scenario>.csv")
    ap.add_argument("--storages", default=DEFAULT_STORAGES, help="Path to storages.csv for duration metadata")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output directory for plots")
    ap.add_argument("--sample", type=int, default=80000, help="Max samples used for density plots")
    ap.add_argument("--all-pairs", action="store_true", help="Generate one plot per pair (can be slow).")
    ap.add_argument(
        "--pairs",
        nargs="*",
        default=[],
        help="Target specific pairs like: HVDC,PHES_energy  PV,Wind  (group names, not G_*).",
    )
    ap.add_argument("--include-other", action="store_true", help="Include 'Other' group if present.")
    ap.add_argument("--no-log1p", action="store_true", help="Skip the log1p scatter-matrix.")
    args = ap.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    bands = pd.read_csv(args.bands)
    near = pd.read_csv(args.near)
    div = pd.read_csv(args.div)

    build_cols = infer_build_columns(near)
    group_map  = build_group_map(build_cols)
    storage_meta = load_storage_meta(args.storages)

    opt = load_optimum_vector(args.opt_x, build_cols, storage_meta)
    opt_g = {f"G_{g}": float(opt[cols].sum()) for g, cols in group_map.items()}

    near_g = add_group_totals(near, group_map).assign(Source="near_opt")
    div_g  = add_group_totals(div, group_map).assign(Source="diversify")
    sol = pd.concat([near_g, div_g], ignore_index=True)
    sol = add_primary_totals(sol)

    best_lcoe = float(sol["LCOE [$/MWh]"].min())
    sol["dLCOE_pct"] = (sol["LCOE [$/MWh]"] - best_lcoe) / best_lcoe * 100.0

    group_keys = list(group_map.keys())
    if not args.include_other:
        group_keys = [g for g in group_keys if g != "Other"]

    group_cols = [f"G_{g}" for g in group_keys]
    opt_point  = np.array([opt_g[c] for c in group_cols], dtype=float)

    # A) scatter matrices (these already show trade-offs for *any* two groups)
    scatter_matrix(
        sol,
        cols=group_cols,
        opt_point=opt_point,
        title="Grouped build space (all pairs), colored by ΔLCOE%",
        outpath=os.path.join(outdir, "00_scatter_matrix_linear.png"),
        transform="linear"
    )
    if not args.no_log1p:
        scatter_matrix(
            sol,
            cols=group_cols,
            opt_point=opt_point,
            title="Grouped build space (all pairs) [log1p scale], colored by ΔLCOE%",
            outpath=os.path.join(outdir, "00_scatter_matrix_log1p.png"),
            transform="log1p"
        )

    # C) substitution heatmap on ratios vs optimum (group-to-group comparability)
    ratio_cols = []
    for c in group_cols:
        denom = opt_g[c]
        if abs(denom) < 1e-12:
            sol[c + "_ratio"] = sol[c]
        else:
            sol[c + "_ratio"] = sol[c] / denom
        ratio_cols.append(c + "_ratio")

    plot_substitution_heatmap(
        sol,
        cols=ratio_cols,
        outpath=os.path.join(outdir, "99_substitution_heatmap_spearman.png"),
        low_cost_cut_pct=10.0
    )

    # D) Primary axes: conditional envelopes and density maps
    primary_cols = ["G_HVDC", "G_Storage_energy", "G_Storage_power", "G_PV", "G_Wind"]
    primary_cols = [c for c in primary_cols if c in sol.columns]
    opt_primary = {
        "G_HVDC": opt_g.get("G_HVDC", np.nan),
        "G_PV": opt_g.get("G_PV", np.nan),
        "G_Wind": opt_g.get("G_Wind", np.nan),
        "G_Storage_power": opt_g.get("G_PHES_power", 0.0) + opt_g.get("G_Battery_power", 0.0),
        "G_Storage_energy": opt_g.get("G_PHES_energy", 0.0) + opt_g.get("G_Battery_energy", 0.0),
    }

    plot_conditional_grid(
        sol,
        primary_cols=primary_cols,
        opt_point=opt_primary,
        outdir=os.path.join(outdir, "conditional"),
    )
    plot_density_pairs(
        sol,
        primary_cols=primary_cols,
        opt_point=opt_primary,
        outdir=os.path.join(outdir, "density"),
        sample_n=args.sample,
    )

    # B) Pair plots (optional, for readability)
    requested_pairs = []
    for p in args.pairs:
        a, b = p.split(",")
        a, b = a.strip(), b.strip()
        if f"G_{a}" in group_cols and f"G_{b}" in group_cols:
            requested_pairs.append((f"G_{a}", f"G_{b}"))
        else:
            print(f"Skipping unknown pair '{p}'. Available groups: {group_keys}")

    do_pairs = args.all_pairs or (len(requested_pairs) > 0)
    if do_pairs:
        pairdir = os.path.join(outdir, "pairs")
        os.makedirs(pairdir, exist_ok=True)

        if args.all_pairs:
            pairs = []
            for i in range(len(group_cols)):
                for j in range(i+1, len(group_cols)):
                    pairs.append((group_cols[j], group_cols[i]))  # x,y
        else:
            pairs = [(b, a) for (a, b) in requested_pairs]  # keep x,y order consistent

        for xcol, ycol in pairs:
            out1 = os.path.join(pairdir, f"pair_{ycol.replace('G_','')}_vs_{xcol.replace('G_','')}_linear.png")
            pair_tradeoff_plot(sol, xcol=xcol, ycol=ycol, opt_xy=(opt_g[xcol], opt_g[ycol]),
                               outpath=out1, envelope=True, transform="linear")
            if not args.no_log1p:
                out2 = os.path.join(pairdir, f"pair_{ycol.replace('G_','')}_vs_{xcol.replace('G_','')}_log1p.png")
                pair_tradeoff_plot(sol, xcol=xcol, ycol=ycol, opt_xy=(opt_g[xcol], opt_g[ycol]),
                                   outpath=out2, envelope=True, transform="log1p")

    print("Done. Outputs in:", outdir)
    print("Groups used:", group_keys)

if __name__ == "__main__":
    main()
