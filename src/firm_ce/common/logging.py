import logging
import os
from datetime import datetime
from typing import Tuple

logging.getLogger("numba").setLevel(logging.WARNING)


def init_model_logger(model_name: str, logging_flag: bool) -> Tuple[logging.Logger, str]:
    """
    Initialize a logger for the model run, configured to log both to console and a log file.
    Logger does not work within JIT compiled code.

    The logger writes:
    - All messages (DEBUG and above) to a log file in a new `results/Model_<timestamp>` directory.
    - INFO and higher messages to the console.

    Returns:
    -------
    Tuple[logging.Logger, str]: A tuple containing the configured `Logger` instance and the path to the results
        directory.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if logging_flag:
        results_dir = os.path.join("results", f"{model_name}_{timestamp}")
    else:
        results_dir = os.path.join("results", "temp")
    os.makedirs(results_dir, exist_ok=True)

    log_path = os.path.join(results_dir, "log.txt")

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_format)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False

    logger.info(f"Logger initialized. Writing to {log_path}")
    return logger, results_dir
