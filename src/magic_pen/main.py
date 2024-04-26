import torch
from segment_anything import SamPredictor, sam_model_registry  # type: ignore

if __name__ == "__main__":
    print(torch.cuda.is_available())
