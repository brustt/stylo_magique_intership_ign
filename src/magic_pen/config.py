from dotenv import find_dotenv
import os
from pathlib import Path

project_path = os.path.dirname(find_dotenv())

root_data_path = Path(os.path.expanduser("~"), "data")
data_path = Path(project_path, "data")
sam_data_path = Path(data_path, "demo", "sam")
levirCD_path = Path(root_data_path, "levir-cd")

model_path = Path(project_path, "models")
sam_model_path = Path(model_path, "sam")
sam_model_large = Path(sam_model_path, "sam_vit_h_4b8939.pth")
sam_model_small = Path(sam_model_path, "sam_vit_b_01ec64.pth")
sam_dict_checkpoint = {"vit_h": sam_model_large, "vit_b": sam_model_small}


### constants

SEED = 12
DEVICE = "cuda"
