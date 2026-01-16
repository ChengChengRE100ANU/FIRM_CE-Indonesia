
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Broad optimum visualisation
#   - near_optimal_bands.csv (extreme within-band portfolios)
#   - near_optimal_space_snapshot.csv (all feasible within-band samples seen during min/max searches)
#   - diversify_space.csv (additional feasible, structurally-far samples)
#   - optimal_x_default.csv (single raw build-vector, no header)
#
# Output: PNGs under OUTDIR
# ============================================================

# -----------------------------
# Config: CLI args
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BANDS = os.path.join(BASE_DIR, "results", "near_optimum", "default", "near_optimal_bands.csv")
DEFAULT_NEAR = os.path.join(BASE_DIR, "results", "near_optimum", "default", "near_optimal_space.csv")
DEFAULT_DIV = os.path.join(BASE_DIR, "results", "diversify", "default", "diversify_space.csv")
DEFAULT_OPT_X = os.path.join(BASE_DIR, "results", "temp", "optimal_x_default.csv")
DEFAULT_STORAGES = os.path.join(BASE_DIR, "inputs", "config", "storages.csv")
DEFAULT_OUTDIR = os.path.join(BASE_DIR, "broad_optimum_plots")

parser = argparse.ArgumentParser(description="Broad optimum visualisation")
parser.add_argument("--bands", default=DEFAULT_BANDS, help="Path to near_optimal_bands.csv")
parser.add_argument("--near", default=DEFAULT_NEAR, help="Path to near_optimal_space.csv")
parser.add_argument("--div", default=DEFAULT_DIV, help="Path to diversify_space.csv")
parser.add_argument("--opt-x", default=DEFAULT_OPT_X, help="Path to optimal_x_<scenario>.csv")
parser.add_argument("--storages", default=DEFAULT_STORAGES, help="Path to storages.csv for duration metadata")
parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output directory for plots")
args = parser.parse_args()

PATH_BANDS = args.bands
PATH_NEAR = args.near
PATH_DIV = args.div
PATH_OPT_X = args.opt_x
PATH_STORAGES = args.storages

OUTDIR = args.outdir
os.makedirs(OUTDIR, exist_ok=True)

# -----------------------------
# Helpers
# -----------------------------
META_NEAR = {"Group", "Band_Type", "LCOE [$/MWh]", "Operational_Penalty", "Band_Penalty"}
META_DIV  = {"LCOE [$/MWh]", "Operational_Penalty", "Band_Penalty", "Scaled_Novelty"}

def infer_build_columns(df: pd.DataFrame) -> list[str]:
    """Build columns are everything except known metadata columns."""
    cols = [c for c in df.columns if c not in META_NEAR]
    return cols

def build_group_map(build_cols: list[str]) -> dict[str, list[str]]:
    """
    Infer column groups using naming patterns used by FIRM-CE renders.
    Adjust patterns here if your column naming changes.
    """
    def starts(prefix):
        return [c for c in build_cols if c.lower().startswith(prefix)]

    pv   = starts("pv_")
    wind = starts("wind_")
    phes_p = [c for c in build_cols if c.lower().startswith("phes_") and c.lower().endswith("_power")]
    phes_e = [c for c in build_cols if c.lower().startswith("phes_") and c.lower().endswith("_energy")]
    bess_p = [c for c in build_cols if c.lower().startswith("bess_") and c.lower().endswith("_power")]
    bess_e = [c for c in build_cols if c.lower().startswith("bess_") and c.lower().endswith("_energy")]

    # HVDC lines are typically named like "APB1-APB2"
    hvdc = [c for c in build_cols if "-" in c and not c.lower().endswith(("_power", "_energy"))]

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

    return {k:v for k,v in group_map.items() if len(v) > 0}

def add_group_totals(df: pd.DataFrame, group_map: dict[str, list[str]], prefix="G_") -> pd.DataFrame:
    out = df.copy()
    for g, cols in group_map.items():
        out[prefix + g] = out[cols].sum(axis=1)
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
    """
    optimal_x_default.csv is a one-line CSV with the raw decision vector (no header).
    We align it to build_cols inferred from near_optimal_space and render storage energy if needed.
    """
    x = pd.read_csv(path_opt_x, header=None).iloc[0].to_numpy()
    # Some writers leave a trailing comma -> extra NaN column; drop NaNs at the end.
    x = x[~pd.isna(x)]
    if len(x) != len(build_cols):
        raise ValueError(
            f"Optimal vector has {len(x)} entries, but build_cols has {len(build_cols)}. "
            f"Check file/scenario mismatch."
        )
    series = pd.Series(x, index=build_cols, name="optimum")
    return render_storage_energy(series, storage_meta)

def pca_fit_transform(X: np.ndarray, n_components=2):
    """
    Lightweight PCA (SVD) with standardisation.
    Returns (Z, mean, std, Vt) where:
      Z   : projected coordinates (N x n_components)
      mean/std : feature standardisation
      Vt  : right singular vectors (components), shape (D x D)
    """
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    std = Xc.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    Xn = Xc / std
    U, S, Vt = np.linalg.svd(Xn, full_matrices=False)
    Z = Xn @ Vt.T[:, :n_components]
    return Z, mean, std, Vt

# -----------------------------
# Load data
# -----------------------------
bands = pd.read_csv(PATH_BANDS)
near  = pd.read_csv(PATH_NEAR)
div   = pd.read_csv(PATH_DIV)

build_cols = infer_build_columns(near)
group_map  = build_group_map(build_cols)
storage_meta = load_storage_meta(PATH_STORAGES)

opt = load_optimum_vector(PATH_OPT_X, build_cols, storage_meta)

# Add grouped totals
near_g = add_group_totals(near, group_map)
div_g  = add_group_totals(div, group_map)
opt_g  = pd.Series({f"G_{g}": float(opt[cols].sum()) for g, cols in group_map.items()}, name="optimum_groups")

# Combine near + diversify spaces
near_g = near_g.assign(Source="near_opt", Novelty=np.nan)
div_g  = div_g.assign(Source="diversify").rename(columns={"Scaled_Novelty":"Novelty"})
sol = pd.concat([near_g, div_g], ignore_index=True)

# Optional: keep only feasible & within-band (if your logs include near-feasible points)
# tol = 1e-6
# sol = sol[(sol["Operational_Penalty"] <= tol) & (sol["Band_Penalty"] <= tol)].copy()

GROUP_KEYS = [g for g in group_map.keys() if g != "Other"]
GROUP_COLS = [f"G_{g}" for g in GROUP_KEYS]

# -----------------------------
# Visual 1: Broad-optimum band ranges vs optimum
# -----------------------------
def plot_band_ranges(bands_df: pd.DataFrame):
    rows = []
    for g in bands_df["Group"].unique():
        if g not in group_map:
            continue
        cols = group_map[g]
        sub = bands_df[bands_df["Group"] == g]
        for bt in ["min", "max"]:
            r = sub[sub["Band_Type"] == bt]
            if r.empty:
                continue
            rows.append({
                "Group": g,
                "Band_Type": bt,
                "Group_Total": float(r[cols].sum(axis=1).iloc[0]),
            })
    band_tot = pd.DataFrame(rows)
    pivot = band_tot.pivot_table(index="Group", columns="Band_Type", values="Group_Total", aggfunc="first")
    pivot = pivot.reset_index()
    pivot["optimum"] = [opt_g.get(f"G_{g}", np.nan) for g in pivot["Group"]]
    pivot = pivot.sort_values("Group")

    y = np.arange(len(pivot))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.hlines(y=y, xmin=pivot["min"], xmax=pivot["max"], linewidth=7, alpha=0.65)
    ax.plot(pivot["optimum"], y, marker="*", linestyle="None", markersize=11, label="optimum")
    ax.set_yticks(y)
    ax.set_yticklabels(pivot["Group"])
    ax.set_xlabel("Total capacity (grouped)")
    ax.set_title("Near-optimal band extremes (min to max) per group")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "01_band_ranges_vs_optimum.png"), dpi=220)
    plt.close(fig)

plot_band_ranges(bands)

# -----------------------------
# Visual 2: Group distributions across aggregated solution space
# -----------------------------
def plot_group_distributions(sol_df: pd.DataFrame, groups: list[str], max_points=25000):
    # Sample for faster histogram rendering
    if len(sol_df) > max_points:
        sol_s = sol_df.sample(max_points, random_state=0)
    else:
        sol_s = sol_df

    for g in groups:
        col = f"G_{g}"
        if col not in sol_s.columns:
            continue
        fig, ax = plt.subplots(figsize=(8.6, 4.8))
        for src in ["near_opt", "diversify"]:
            d = sol_s.loc[sol_s["Source"] == src, col].dropna()
            if len(d) > 0:
                ax.hist(d, bins=60, alpha=0.5, density=True, label=src)
        ax.axvline(opt_g.get(col, np.nan), linewidth=2, linestyle="--", label="optimum")
        # band min/max from near_optimal_bands (if available)
        if "Group" in bands.columns:
            bsub = bands[bands["Group"] == g]
            if not bsub.empty:
                cols = group_map[g]
                bmin = float(bsub[bsub["Band_Type"] == "min"][cols].sum(axis=1).iloc[0])
                bmax = float(bsub[bsub["Band_Type"] == "max"][cols].sum(axis=1).iloc[0])
                ax.axvline(bmin, linewidth=1.5, linestyle=":", label="band min")
                ax.axvline(bmax, linewidth=1.5, linestyle=":", label="band max")

        ax.set_title(f"Distribution of grouped total: {g}")
        ax.set_xlabel("Total capacity (grouped)")
        ax.set_ylabel("Density (sampled)")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, f"02_dist_{g}.png"), dpi=220)
        plt.close(fig)

plot_group_distributions(sol, groups=GROUP_KEYS)

# -----------------------------
# Visual 3: PCA projection of solution space (near vs diversify), colored by LCOE
# -----------------------------
def plot_pca(sol_df: pd.DataFrame, groups: list[str], n=20000):
    use_cols = [f"G_{g}" for g in groups if f"G_{g}" in sol_df.columns]
    df = sol_df.dropna(subset=use_cols + ["LCOE [$/MWh]"]).copy()
    if len(df) > n:
        df = df.sample(n, random_state=1)

    X = df[use_cols].to_numpy()
    Z, mean, std, Vt = pca_fit_transform(X, n_components=2)

    # Project optimum into the same PCA space
    x_opt = np.array([opt_g[c] for c in use_cols], dtype=float)[None, :]
    x_opt_n = (x_opt - mean) / std
    z_opt = x_opt_n @ Vt.T[:, :2]

    fig, ax = plt.subplots(figsize=(8.8, 5.4))

    # plot near and diversify separately so marker encodes source
    mask_near = (df["Source"] == "near_opt").to_numpy()
    mask_div  = (df["Source"] == "diversify").to_numpy()

    sc1 = ax.scatter(Z[mask_near, 0], Z[mask_near, 1],
                     c=df.loc[mask_near, "LCOE [$/MWh]"].to_numpy(),
                     s=10, alpha=0.6, marker="o", label="near_opt")
    ax.scatter(Z[mask_div, 0], Z[mask_div, 1],
               c=df.loc[mask_div, "LCOE [$/MWh]"].to_numpy(),
               s=14, alpha=0.6, marker="^", label="diversify")

    ax.scatter(z_opt[0,0], z_opt[0,1], marker="*", s=220, label="optimum")

    ax.set_title("PCA of grouped build totals (colored by LCOE)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.2)
    fig.colorbar(sc1, ax=ax, label="LCOE [$/MWh]")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "03_pca_grouped_space.png"), dpi=220)
    plt.close(fig)

plot_pca(sol, groups=GROUP_KEYS)

# -----------------------------
# Visual 4: Parallel coordinates (normalized to optimum)
# -----------------------------
def plot_parallel(sol_df: pd.DataFrame, groups: list[str], n=2500):
    use_cols = [f"G_{g}" for g in groups if f"G_{g}" in sol_df.columns]
    df = sol_df.dropna(subset=use_cols).copy()

    if len(df) > n:
        df = df.sample(n, random_state=2)

    opt_vals = np.array([opt_g[c] for c in use_cols], dtype=float)
    denom = np.where(np.abs(opt_vals) < 1e-12, 1.0, opt_vals)
    R = df[use_cols].to_numpy(dtype=float) / denom

    fig, ax = plt.subplots(figsize=(9.2, 5.3))
    xs = np.arange(len(use_cols))

    # lighter lines for near-opt; heavier for diversify
    for src, alpha in [("near_opt", 0.06), ("diversify", 0.10)]:
        rsrc = R[df["Source"].to_numpy() == src, :]
        for i in range(rsrc.shape[0]):
            ax.plot(xs, rsrc[i, :], alpha=alpha)

    ax.plot(xs, np.ones_like(xs), linewidth=2.3, linestyle="--", label="optimum (1.0)")
    ax.set_xticks(xs)
    ax.set_xticklabels([c.replace("G_","") for c in use_cols], rotation=30, ha="right")
    ax.set_ylabel("Capacity ratio vs optimum (1.0 = optimum)")
    ax.set_title("Parallel coordinates: sampled solutions normalized to optimum")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "04_parallel_normalized.png"), dpi=220)
    plt.close(fig)

plot_parallel(sol, groups=GROUP_KEYS)

# -----------------------------
# Visual 5: Novelty vs LCOE for diversify solutions
# -----------------------------
def plot_novelty(sol_df: pd.DataFrame, n=20000):
    df = sol_df[sol_df["Source"] == "diversify"].dropna(subset=["Novelty", "LCOE [$/MWh]"]).copy()
    if df.empty:
        return
    if len(df) > n:
        df = df.sample(n, random_state=3)

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.scatter(df["Novelty"].to_numpy(), df["LCOE [$/MWh]"].to_numpy(), s=12, alpha=0.55)
    ax.set_xlabel("Scaled novelty score")
    ax.set_ylabel("LCOE [$/MWh]")
    ax.set_title("Diversify space: novelty vs LCOE (sampled)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "05_diversify_novelty_vs_lcoe.png"), dpi=220)
    plt.close(fig)

plot_novelty(sol)

print(f"Saved plots to: {OUTDIR}")
