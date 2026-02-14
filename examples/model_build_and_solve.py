"""
An example is provided for building a FIRM Model instance and then solving it. The Model object is built using the default `inputs/config` and
`inputs/data` files. Each scenario in `inputs/config/scenarios.csv` is optimised sequentially using the SciPy differential evolution algorithm.
Results are saved in the `results` folder.

Alternative filepaths for the config and data folders can be provided as arguments to the Model instantiation.
"""

import argparse
import time
from pathlib import Path

from firm_ce.model import Model


def _default_input_dirs() -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "inputs" / "config", repo_root / "inputs" / "data"


def _parse_args() -> argparse.Namespace:
    default_config_dir, default_data_dir = _default_input_dirs()
    parser = argparse.ArgumentParser(description="Build and solve a FIRM model from input folders.")
    parser.add_argument("--config-dir", type=Path, default=default_config_dir, help="Path to inputs/config.")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir, help="Path to inputs/data.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a very small optimisation (overrides config.type, iterations, and population).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    start_time = time.time()
    model = Model(
        config_directory=str(args.config_dir),
        data_directory=str(args.data_dir),
        logging_flag=False,
    )
    model_build_time = time.time()

    print(model.scenarios)
    print(f"Model build time: {model_build_time - start_time:.4f} seconds")

    if args.smoke:
        model.config.type = "single_time"
        model.config.iterations = 1
        model.config.population = 1

    model.solve()
    end_time = time.time()
    print(f"Model solve time: {end_time - model_build_time:.4f} seconds")


if __name__ == "__main__":
    main()
