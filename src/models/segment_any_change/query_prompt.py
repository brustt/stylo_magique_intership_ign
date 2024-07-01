from copy import deepcopy
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, Union
from src.data.process import generate_prompt
from src.models.commons.mask_process import binarize_mask
from .matching import BitemporalMatching
from src.commons.utils import resize
import numpy as np
import torch
from commons.config import IMG_SIZE
from models.segment_any_change.embedding import (
    compute_mask_embedding,
    get_img_embedding_normed,
)
from src.models.commons.mask_items import (
    FilteringType,
    thresholding,
    MaskData
)

from models.commons.model import SamModeInference

from src.commons.utils import to_degre, timeit, to_degre_torch
import logging
from torch import nn
from torch.nn.utils.rnn import pad_sequence


class SegAnyPrompt:
    def __init__(
            self,
            matching_engine: BitemporalMatching,
            **params):
        self.matching_engine = matching_engine
        self.params = params

    def __call__(self, batch: Dict[str, torch.Tensor]):

        
        grid_params = deepcopy(self.params)
        grid_batch = batch.copy()

        # switch for matching via grid
        grid_params["prompt_type"] = "grid"
        grid_params["n_prompt"] = grid_params["n_points_grid"] 

        point_coords, point_labels = generate_prompt(grid_batch["label"], 
                                                    grid_params["prompt_type"], 
                                                    grid_params["n_prompt"], 
                                                    **grid_params)
        
        grid_batch["point_coords"] = point_coords.repeat(self.params["batch_size"], 1, 1)
        grid_batch["point_labels"] = point_labels.repeat(self.params["batch_size"], 1)

        items_change = self.matching_engine(grid_batch,  **grid_params)

        # query prompt
        query_res = QueryPointMecanism(items_change, 
                                          model=self.matching_engine.mask_generator.model, # sam 
                                          th_sim=self.params["th_sim"]).run(batch)
        
        return query_res

class QueryPointMecanism:
    def __init__(self, items_change: Dict, model: Any, th_sim: Any):
        self.items_change = items_change
        self.model = model
        self.th_sim = th_sim

    def get_best_masks(self, masks: torch.Tensor, ious: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract mask associated to the highest IoU

        predictions from multimask_output == True

        Args:
            masks (torch.Tensor): sam output masks : B x N x 3 x He x We 
            ious (torch.Tensor): sam ious predicted : B x N x 3

        Returns:
            torch.Tensor: best masks : B x N x He x We
        """
        best_ious, max_indices = torch.max(ious, dim=2)
        
        # align dimensions to masks
        max_indices_expanded = (
            max_indices
            .unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, -1, *masks.shape[-2:])
        )
        best_masks = masks.gather(2, max_indices_expanded).squeeze(2)

        return best_masks, best_ious
    
    def compute_cross_sim(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        
        Check why some items change embedding are null => pad sequence ?
        Args:
            x1 (torch.Tensor): mask embedding from prompt:  1 x 256
            x2 (torch.Tensor): items change from bitemporal matching : M x 256

        Returns:
            torch.Tensor: sim cosine : 1 x M
        """
        norm_1 = torch.linalg.norm(x1)
        norm_2 = torch.linalg.norm(x2, dim=1)

        dot_prod = (x1 @ x2.permute(1, 0))
        dm = norm_1 * norm_2

        return dot_prod / dm
    

    def run(self, batched_input: Dict) -> torch.Tensor:

        batch_size = batched_input[next(iter(batched_input))].shape[0]

        # SamModeInference.INTERACTIVE == inference on img_B only
        outputs = self.model(
            batched_input=batched_input, 
            multimask_output=True, 
            return_logits=True,
            mode=SamModeInference.INTERACTIVE,
        )
        # TODO : prevent image encoder to recompute images embedding

        # new_masks : B x N x Hm x Wm - can be computed on batch
        # iou_predictions :  B x N
        new_masks, iou_predictions = outputs.values()
        new_masks = new_masks > 0.
        best_masks, best_ious = self.get_best_masks(new_masks, iou_predictions)

        # only one img type was provided to the model for query prompting
        imgs_embedding_B = get_img_embedding_normed(self.model, img_type=None)
        # B x N x 256 -- N  number of masks in the img i (== number of prompts)
        masks_embedding = compute_mask_embedding(best_masks, imgs_embedding_B)
        
        # need to have 1 emb => mean of mask_embedding : B x 256
        masks_embedding = masks_embedding.mean(dim=1)

        print(masks_embedding.shape)
        print(self.items_change["proposal_emb"].shape)
        print(batch_size)

        batch_masked = []
        # check if best_masks not null
        for i in range(batch_size): 

            # 1 x M --| M number of masks in items change
            sim_scores = self.compute_cross_sim(masks_embedding[i,...], self.items_change["proposal_emb"][i,...])

            data = MaskData(
                # M x H x W
                masks=self.items_change["masks"][i,...], # need to align dimensions
                # M
                sim=sim_scores,
                iou_preds=self.items_change["iou_preds"][i,...],
                ci=self.items_change["confidence_scores"][i,...],
            )

            # get similar changes
            data, th = thresholding(data, attr="sim", method=self.th_sim, filtering_type=FilteringType.Sup)
            batch_masked.append(data)

        if best_masks.shape[1]:
            best_masks = resize(best_masks, IMG_SIZE)

        sim_masks = pad_sequence([elem["masks"] for elem in batch_masked], batch_first=True)
        sim_iou_preds = pad_sequence([elem["iou_preds"] for elem in batch_masked], batch_first=True)
        sim_ci = pad_sequence([elem["iou_preds"] for elem in batch_masked], batch_first=True)
        sim_sim = pad_sequence([elem["sim"] for elem in batch_masked], batch_first=True)

        masks = torch.cat([sim_masks, best_masks], dim=1)
        iou_preds = torch.cat([sim_iou_preds, best_ious], dim=1)
        
        # simulate confidence score as changement, i.e neg cosine sim high
        ci = torch.cat([sim_ci, torch.ones((batch_size, best_masks.shape[1]))], dim=1)
        # simulate  i.e cosine sim max for prompts objects
        sim = torch.cat([sim_sim, torch.ones((batch_size, best_masks.shape[1]))], dim=1)

        return dict(masks=masks, 
                    all_changes=self.items_change["masks"], # tmp
                    sim=sim, # tmp
                    prompt_masks=best_masks,
                    iou_preds=iou_preds, # B x max(NA, NB) 
                    confidence_scores=ci) # B x max(NA, NB) )