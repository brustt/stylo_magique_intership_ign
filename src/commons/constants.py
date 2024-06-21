from enum import Enum


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
