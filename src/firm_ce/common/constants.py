import os

import numpy as np

JIT_ENABLED = True
SAVE_POPULATION = True
# Enable verbose debug output by setting FIRM_CE_DEBUG=1 in the environment.
DEBUG = os.getenv("FIRM_CE_DEBUG", "0") == "1"
# Optional debug filters for dispatch instrumentation.
DEBUG_INTERVAL = int(os.getenv("FIRM_CE_DEBUG_INTERVAL", "-1"))
DEBUG_NODE_ORDER = int(os.getenv("FIRM_CE_DEBUG_NODE", "-1"))
DEBUG_GENERATOR_ORDER = int(os.getenv("FIRM_CE_DEBUG_GENERATOR", "-1"))
EPSILON_FLOAT64 = np.finfo(np.float64).eps
NP_FLOAT_MAX = np.finfo(np.float64).max
NP_FLOAT_MIN = np.finfo(np.float64).min
NP_INT64_MAX = np.iinfo(np.int64).max
PENALTY_MULTIPLIER = 1e6
TOLERANCE = 1e-6
NUM_THREADS = int(os.getenv("NUM_THREADS", os.cpu_count()))
FASTMATH = True

# Defaults for PV growth constraints (overridden by config if present)
DEFAULT_PV_CAGR_CAP = 0.69
DEFAULT_PV_FIRST_YEAR_CAP_GW = 10.0
