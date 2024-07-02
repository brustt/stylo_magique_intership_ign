from dotenv import find_dotenv
import os
from pathlib import Path

# TODO : split config paths int models and data

# add sim link to lightnings logs and root_data_path
project_path = os.path.dirname(find_dotenv())

root_data_path = Path(os.path.expanduser("~"), "data/dl")
data_path = Path(project_path, "data")
sam_data_path = Path(data_path, "demo", "sam")

CONFIG_PATH = Path(project_path, "config")

# TODO : change constants name to uppercase
# ds paths
levirCD_path = Path(root_data_path, "levir-cd")
SECOND_PATH = Path(root_data_path, "SECOND")

model_path = Path(project_path, "checkpoints")
sam_model_path = Path(model_path, "sam")
sam_model_large = Path(sam_model_path, "sam_vit_h_4b8939.pth")
sam_model_small = Path(sam_model_path, "sam_vit_b_01ec64.pth")
sam_dict_checkpoint = {"vit_h": sam_model_large, "vit_b": sam_model_small}

logs_dir = Path(project_path, "lightning_logs")  # need simlink

### constants

SEED = 12
DEVICE = "cpu"
IMG_SIZE = (1024, 1024)
