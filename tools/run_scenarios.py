"""
Utility script to run one or more scenarios without manually editing scenarios.csv.

Usage (from repo root):

    .\.venv\Scripts\python.exe tools/run_scenarios.py --scenarios ce_more_baseload

Optional arguments:
    --config   Path to a config directory (default: inputs/config)
    --data     Path to a data directory   (default: inputs/data)
    --workdir  Directory to place the filtered config copy (default: auto-created temp dir)
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile

import pandas as pd

from firm_ce.model import Model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run selected FIRM_CE scenarios.")
    parser.add_argument(
        "--scenarios",
        required=True,
        help="Comma-separated scenario names to solve (e.g. ce_more_baseload,ce_2025_2060).",
    )
    parser.add_argument("--config", default="inputs/config", help="Path to base config directory.")
    parser.add_argument("--data", default="inputs/data", help="Path to data directory.")
    parser.add_argument(
        "--workdir",
        default=None,
        help="Directory to copy filtered config into (default: auto-created temp dir).",
    )
    args = parser.parse_args()

    scenario_names = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    if not scenario_names:
        raise SystemExit("No scenarios provided.")

    base_config_dir = os.path.abspath(args.config)
    data_dir = os.path.abspath(args.data)

    if not os.path.isdir(base_config_dir):
        raise SystemExit(f"Config directory not found: {base_config_dir}")
    if not os.path.isdir(data_dir):
        raise SystemExit(f"Data directory not found: {data_dir}")

    if args.workdir:
        work_config_dir = os.path.abspath(args.workdir)
        os.makedirs(work_config_dir, exist_ok=True)
    else:
        work_root = tempfile.mkdtemp(prefix="firm_ce_run_")
        work_config_dir = work_root

    shutil.copytree(base_config_dir, work_config_dir, dirs_exist_ok=True)

    scenarios_path = os.path.join(work_config_dir, "scenarios.csv")
    scenarios_df = pd.read_csv(scenarios_path)
    filtered_scenarios = scenarios_df[scenarios_df["scenario_name"].isin(scenario_names)]
    if filtered_scenarios.empty:
        raise SystemExit("None of the requested scenarios are present in scenarios.csv.")
    filtered_scenarios.to_csv(scenarios_path, index=False)

    initial_guess_path = os.path.join(work_config_dir, "initial_guess.csv")
    if os.path.isfile(initial_guess_path):
        ig_df = pd.read_csv(initial_guess_path)
        ig_df = ig_df[ig_df["scenario"].isin(scenario_names)]
        ig_df.to_csv(initial_guess_path, index=False)

    print(f"Running scenarios: {', '.join(filtered_scenarios['scenario_name'])}")
    print(f"Using config copy at: {work_config_dir}")
    model = Model(config_directory=work_config_dir, data_directory=data_dir)
    model.solve()


if __name__ == "__main__":
    main()
