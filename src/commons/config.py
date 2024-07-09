from dotenv import find_dotenv
import os
from pathlib import Path

# TODO : split config paths int models and data

PROJECT_PATH = os.environ["PROJECT_PATH"]
DATA_PATH = os.environ["DATA_PATH"]  # simlink to /var/data
MODEL_PATH = os.environ["CHECKPOINTS_PATH"]  # simlink to /var/data
CONFIG_PATH = Path(PROJECT_PATH, "config")

SAM_DATA_DEMO_PATH = os.environ["SAM_DATA_DEMO_PATH"]

# ds paths
LEVIRCD_PATH = Path(DATA_PATH, "levir-cd")
SECOND_PATH = Path(DATA_PATH, "SECOND")

# sam chkps paths
SAM_MODEL_PATH = Path(MODEL_PATH, "sam")
SAM_MODEL_LARGE_PATH = Path(SAM_MODEL_PATH, "sam_vit_h_4b8939.pth")
SAM_MODEL_SMALL_PATH = Path(SAM_MODEL_PATH, "sam_vit_b_01ec64.pth")
SAM_DICT_CHECKPOINT = {"vit_h": SAM_MODEL_LARGE_PATH, "vit_b": SAM_MODEL_SMALL_PATH}

LOGS_DIR = Path(PROJECT_PATH, "lightning_logs")  # simlink to /var/data

### constants

SEED = 12
DEVICE = "cpu"
IMG_SIZE = (1024, 1024)
