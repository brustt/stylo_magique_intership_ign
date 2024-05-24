from typing import Any, List, Optional, Tuple, Union
import numpy as np
from magic_pen.config import DEVICE
from magic_pen.data.loader import BiTemporalDataset
from magic_pen.data.process import DefaultTransform
from segment_any_change.embedding import (
    compute_mask_embedding,
    get_img_embedding_normed,
)
from segment_any_change.mask_items import (
    FilteringType,
    ListProposal,
)
from segment_any_change.matching import (
    BitemporalMatching,
    neg_cosine_sim,
)
from segment_any_change.model import BiSam
from segment_any_change.sa_dev_v0.predictor import SamPredictor
from torch.utils.data import DataLoader
from segment_any_change.utils import flush_memory, load_sam, to_degre, timeit
import logging

# TO DO : define globally
logging.basicConfig(format="%(asctime)s - %(levelname)s ::  %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == "__main__":

    flush_memory()

    sam_params = {
        "points_per_side": 10, #lower for speed
        "points_per_batch": 64, # not used
        "pred_iou_thresh": 0.88, # configure lower for exhaustivity
        "stability_score_thresh": 0.95, # configure lower for exhaustivity
        "stability_score_offset": 1.0,
        "box_nms_thresh": 0.7,
        "min_mask_region_area": 0,
    }
    logger.info("==== start ====")
    # experiment parameters
    filter_change_proposals = "otsu"
    filter_query_sim = 70
    batch_size=1
    model_type="vit_b"

    ds = BiTemporalDataset(
        name="levir-cd", 
        dtype="train", 
        transform=DefaultTransform()
    )
    dataloader = DataLoader(ds, batch_size=batch_size)

    model = load_sam(
        model_type=model_type, 
        model_cls=BiSam,
        version="dev", 
        device=DEVICE
        )

    matcher = BitemporalMatching(model, **sam_params)
    logger.info("--- Bitemporal matching ---")
    for i_batch, batch in enumerate(dataloader):
        logger.info(f"Run batch {i_batch}")

        items_change, th = matcher.run(
            batch=batch,
            filter_method=filter_change_proposals,
        )
        print(f"Done : {len(items_change)}")
        break

