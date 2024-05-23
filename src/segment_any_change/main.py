from typing import Any, List, Optional, Tuple, Union
import numpy as np
from segment_any_change.embedding import (
    compute_mask_embedding,
    get_img_embedding_normed,
)
from segment_any_change.mask_items import (
    FilteringType,
    ListProposal,
)
from segment_any_change.matching import (
    neg_cosine_sim,
)
from segment_any_change.sa_dev_v0.predictor import SamPredictor

from segment_any_change.utils import (
    to_degre,
    timeit
)
import logging

# TO DO : define globally
logging.basicConfig(format="%(asctime)s - %(levelname)s ::  %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# To update with new batch version
class PointQueryMecanism:

    S_MASK_EMB = (1024, 1024)
    S_CHGT_EMB = (256,)

    def __init__(self, predictor: SamPredictor, items_change: ListProposal) -> None:
        self.predictor = predictor
        self.items_change = items_change
        self.th_filtering = None

    @timeit
    def run(
        self,
        points: np.ndarray,
        method_filtering: Union[str, float],
        image: np.ndarray = None,
        labels: Optional[np.ndarray] = None,
        multimask_output: bool = True,
    ) -> ListProposal:

        masks = self.extract_mask_from_multiple_prompt(
            points, image, labels, multimask_output
        )
        embedding = self.extract_proposal_embedding(masks)
        self.match_changes(embedding, method_filtering)

        return self.items_change

    def match_changes(
        self, emb_proposal: np.ndarray, method_filtering: Union[str, float, int]
    ) -> ListProposal:

        scores_chg = [
            neg_cosine_sim(emb_proposal, item.embedding) for item in self.items_change
        ]
        self.items_change.update_field("confidence_score", scores_chg)
        self.items_change.update_field("chgt_angle", [to_degre(c) for c in scores_chg])
        self.th_filtering = self.items_change.apply_change_filtering(
            method_filtering, FilteringType.Inf
        )

    def extract_mask_from_multiple_prompt(
        self,
        points: np.ndarray,
        image: np.ndarray = None,
        labels: Optional[np.ndarray] = None,
        multimask_output: bool = True,
    ) -> np.ndarray:
        """Predict mask for each input points

        To DO : batch inference with .predict_torch()

        Args:
            points (np.ndarray): input points (x, y) - shape : (N, 1, 2)
            image (np.ndarray, optional): image np.ndarray. Defaults to None. 
            labels (Optional[Tuple[int]], optional): label of prompt (foreground / background). Defaults to None. shape : (N, 1, 2)
            multimask_output (bool, optional): output multimask for a prompt. Defaults to True (recommanded).

        Returns:
            np.ndarray: masks for objects NxHxW
        """

        selected_masks = np.zeros((points.shape[0], *self.S_MASK_EMB), dtype=np.uint8)

        if labels is None:
            # init point label as foreground by default
            labels = np.ones(len(points))

        # set image if not
        if not self.predictor.is_image_set:
            if image is None:
                raise ValueError("Please provide image to set")
            self.predictor.set_image(image)

        # predict mask for each prompt
        # TO DO :  change for batch inference : use predict_torch instead
        for i, input in enumerate(zip(points, labels)):
            # from demo notebook : best to keep multimask_output to True
            point, label = input
            masks, scores, logits = self.predictor.predict(
                point_coords=point,
                point_labels=label,
                multimask_output=multimask_output,
            )
            # Choose the model's best mask
            selected_masks[i, :, :] = masks[np.argmax(scores), :, :]

        return selected_masks

    def extract_proposal_embedding(self, masks: np.ndarray):
        embedding = np.zeros((len(masks), *self.S_CHGT_EMB))

        for i in range(masks.shape[0]):
            embedding[i, :] = compute_mask_embedding(
                masks[i, :, :], get_img_embedding_normed(self.predictor)
            )
        # get avg of embedding mask
        embedding = np.mean(embedding, axis=0)
        return embedding


if __name__ == "__main__":
    """

    flush_memory()

    pair_img = load_levircd_sample(1, seed=42)
    path_label, path_A, path_B = pair_img.iloc[0]
    # default parameters for auto-generation
    sam_params = {
        "points_per_side": 32,
        "points_per_batch": 64,
        "pred_iou_thresh": 0.88,
        "stability_score_thresh": 0.95,
        "stability_score_offset": 1.0,
        "box_nms_thresh": 0.7,
        "crop_n_layers": 0,
        "crop_nms_thresh": 0.7,
        "crop_overlap_ratio": 512 / 1500,
        "crop_n_points_downscale_factor": 1,
        "point_grids": None,
        "min_mask_region_area": 0,
        "output_mode": "binary_mask",
    }
    logger.info("==== start ====")
    filter_change_proposals = "otsu"
    filter_query_sim = 70

    logger.info("--- Bitemporal matching ---")

    matcher = BitemporalMatching(model_type="vit_b", **sam_params)
    items_change = matcher.run(
        img_A=load_img(path_A),
        img_B=load_img(path_B),
        filter_method=filter_change_proposals,
    )
    print(f"Done : {len(items_change)}")

    input_points = np.array([[[272, 272]]]) # weird dim caused by sequential inference
    input_labels = np.array([[1]])

    logger.info("---Point Query Mecanism ---")

    sim_obj_change = PointQueryMecanism(
        predictor=matcher.mask_generator.predictor, items_change=items_change
    ).run(
        points=input_points,
        method_filtering=filter_query_sim,
        image=None,
        labels=input_labels,
    )

    print(f"Done : {len(sim_obj_change)}")

    """