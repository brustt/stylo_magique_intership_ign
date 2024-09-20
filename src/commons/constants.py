from enum import Enum
import os
from pathlib import Path

# TODO : split config paths int models and data

PROJECT_PATH = os.environ["PROJECT_PATH"]
DATA_PATH = os.environ["DATA_PATH"]  # simlink to /var/data
MODEL_PATH = os.environ["CHECKPOINTS_PATH"]  # simlink to /var/data
CONFIG_PATH = Path(PROJECT_PATH, "config")
LOGS_PATH = os.environ["LOGS_PATH"]

SAM_DATA_DEMO_PATH = os.environ["SAM_DATA_DEMO_PATH"]

# ds paths
LEVIRCD_PATH = Path(DATA_PATH, "levir-cd")
SECOND_PATH = Path(DATA_PATH, "SECOND")

# sam chkps paths
SAM_MODEL_PATH = Path(MODEL_PATH, "sam")
SAM_MODEL_LARGE_PATH = Path(SAM_MODEL_PATH, "sam_vit_h_4b8939.pth")
SAM_MODEL_SMALL_PATH = Path(SAM_MODEL_PATH, "sam_vit_b_01ec64.pth")
SAM_DICT_CHECKPOINT = {"vit_h": SAM_MODEL_LARGE_PATH, "vit_b": SAM_MODEL_SMALL_PATH}

### constants

SEED = 12
DEVICE = "cpu"
DEVICE_MAP = {"gpu": "cuda", "cpu": "cpu"}
IMG_SIZE = (1024, 1024)


class NamedDataset(Enum):
    LEVIR_CD = "levir-cd"
    SECOND = "second"


class NamedModels(Enum):
    DUMMY = "dummy"
    SEGANYMATCHING = "matching"
    SEGANYPROMPT = "seganyprompt"


SECOND_RGB_TO_CAT = {
    (0, 255, 0): 1,
    (128, 128, 128): 2,
    (255, 0, 0): 3,
    (0, 128, 0): 4,
    (0, 0, 255): 5,
    (128, 0, 0): 6,
    (255, 255, 255): 0,
}

SECOND_NO_CHANGE_RGB = [255, 255, 255]


# Tree = [0,255,0]
# NVG = [128,128,128]
# Playground = [255,0,0]
# Low Vegetation = [0,128,0]
# Water = [0,0,255]
# Building = [128,0,0]
# No change = [255,255,25
