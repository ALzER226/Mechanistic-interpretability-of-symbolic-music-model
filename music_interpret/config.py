from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Base paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.debug(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
MODEL_DIR = PROJ_ROOT / "models"

# Checkpoints access paths
BASE_MODEL_DIR = MODEL_DIR

# Data access paths
BASE_DATA_DIR = DATA_DIR / "base"
ACTIVATION_DIR = DATA_DIR / "activations"

REPORTS_DIR = PROJ_ROOT / "reports"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
