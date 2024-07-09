from typing import Any, Dict, List, Tuple, Union
from commons.config import IMG_SIZE
from models.segment_any_change.embedding import (
    compute_mask_embedding,
    get_img_embedding_normed,
)
from models.segment_any_change.mask_generator import SegAnyMaskGenerator
from models.segment_anything.build_sam import load_ckpt_sam
from src.models.commons.mask_items import (
    FilteringType,
    ImgType,
    MaskData,
    thresholding,
)
from torch.nn.utils.rnn import pad_sequence
from src.models.commons.mask_process import binarize_mask
from src.commons.utils import (
    SegAnyChangeVersion,
    resize,
    timeit,
    to_degre_torch,
)
import torch
from torchvision.ops.boxes import batched_nms
import logging


# TO DO : define globally
logging.basicConfig(format="%(asctime)s - %(levelname)s ::  %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



class BitemporalMatching:
    def __init__(
        self,
        network,
        th_change_proposals: Union[float, str],
        col_nms_threshold: str,
        **sam_kwargs,
    ) -> None:
                
        if not sam_kwargs.get("sam_ckpt_path", None):
            raise ValueError("please provide sam checkpoint")
        print(sam_kwargs)
        self.mask_generator = SegAnyMaskGenerator(model=load_ckpt_sam(network, sam_kwargs.get("sam_ckpt_path")), **sam_kwargs)
        self.col_nms_threshold = col_nms_threshold
        self.filter_method = th_change_proposals

    def __call__(self, batch: Dict[str, torch.Tensor], **params) -> Any:
        items_batch = self.run(batch, **params)
        return items_batch

    @timeit
    def run(self, batch: Dict[str, torch.Tensor], **params) -> Any:
        """
        Run Bitemporal matching - vectorized manner

        # TODO : check new return on sample

        """
        batch_filtered = []

        
        img_anns = self.mask_generator.generate(batch)
        batch_size = self.mask_generator.batch_size


        imgs_embedding_A = get_img_embedding_normed(self.mask_generator.model, ImgType.A)
        imgs_embedding_B = get_img_embedding_normed(self.mask_generator.model, ImgType.B)

        for i in range(batch_size):

            img_anns_A = img_anns[i]
            img_anns_B = img_anns[i + batch_size]
            img_anns_curr = [img_anns_A, img_anns_B]

            img_emb_A = imgs_embedding_A[i]
            img_emb_B = imgs_embedding_B[i]

            # masks = stack_tensor_from_sam_output(img_anns, key="masks")
            # masks_A, masks_B = masks[:batch_size], masks[batch_size:]
            
            # NA x H x W           
            masks_A =  img_anns_A["masks"]
            # NB x H x W
            masks_B =  img_anns_B["masks"]

            # TODO : clean outputs temporal_matching if not needed
            # t -> t+1
            # ci : N
            x_t_mA, _, ci = temporal_matching_torch(
                img_emb_A, img_emb_B, masks_A
            )
            # t+1 -> t
            # ci1 : N
            _, x_t1_mB, ci1 = temporal_matching_torch(
                img_emb_A, img_emb_B, masks_B
            )
            # 2, max(NA,NB) x H x W
            masks = pad_sequence([masks_A, masks_B], batch_first=True)
            # 2, max(NA,NB)
            confidence_scores = pad_sequence([ci, ci1], batch_first=True)

            # 2 x max(NA, NB) x 4
            bboxes = stack_tensor_from_sam_output(img_anns_curr, key="bbox")
            # 2 x max(NA, NB)
            iou_preds = stack_tensor_from_sam_output(img_anns_curr, key="predicted_iou")
            # 2 x max(NA, NB) x H x W
            masks_logits = stack_tensor_from_sam_output(img_anns_curr, key="masks_logits")

            # 2, max(NA,NB) x 256
            proposal_emb = pad_sequence([x_t_mA, x_t1_mB], batch_first=True)

            print("NMS masks fusion")
            print("masks", masks.shape)
            print("masks i A", masks_A.shape)
            print("masks i B", masks_B.shape)
            print("ci", confidence_scores.shape)
            print("bboxes", bboxes.shape)
            print("ious", iou_preds.shape)
            print("masks_logits", masks_logits.shape)

            # use data structure
            data = MaskData(
                masks=masks.flatten(0, 1),
                masks_logits=masks_logits.flatten(0, 1),
                bboxes=bboxes.flatten(0, 1),
                ci=confidence_scores.flatten(0, 1),
                iou_preds=iou_preds.flatten(0, 1),
                proposal_emb=proposal_emb.flatten(0, 1),

            )
            # simple fusion based on NMS
            data = proposal_matching_nms(
                data=data,
                nms_threshold=self.mask_generator.box_nms_thresh,
                col_threshold=self.col_nms_threshold,
            )

            # apply change threshold
            data["chgt_angle"] = to_degre_torch(data["ci"])
            print("chgt_angle")
            print(data["chgt_angle"].shape)
            data, th = thresholding(data, attr="chgt_angle", method=self.filter_method, filtering_type=FilteringType.Sup)
            print(data["chgt_angle"].shape)

            # we need to get back batch information for each prediction
            # data = reconstruct_batch(data, masks.shape[0])

            batch_filtered.append(data)
        
        # B x N x H x W - N : filtered masks from A & B
        masks = pad_sequence([elem["masks"] for elem in batch_filtered], batch_first=True)
        masks_logits = pad_sequence([elem["masks_logits"] for elem in batch_filtered], batch_first=True)
        iou_preds = pad_sequence([elem["iou_preds"] for elem in batch_filtered], batch_first=True)
        ci = pad_sequence([elem["ci"] for elem in batch_filtered], batch_first=True)
        proposal_emb = pad_sequence([elem["proposal_emb"] for elem in batch_filtered], batch_first=True)

        # check if we catch some masks
        if masks_logits.shape[1]:
            masks_logits = resize(masks_logits, IMG_SIZE)
        # else:
        #     masks_logits =  torch.zeros((masks_logits.shape[0], 0, IMG_SIZE, IMG_SIZE))
        masks_bin = binarize_mask(masks_logits, self.mask_generator.mask_threshold)

        return dict(
            masks=masks_bin, # B x max(NA, NB) x H x W
            proposal_emb=proposal_emb, # B x max(NA, NB) x 256
            iou_preds=iou_preds, # B x max(NA, NB) 
            confidence_scores=ci, # B x max(NA, NB) 
        )

def neg_cosine_sim_torch(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Compute negative cosime similarities on mask embedding from a image paire

    Args:
        x1 (torch.Tensor): embedding 1 : N x C
        x2 (torch.Tensor): embedding 2 : N x C

    Returns:
        torch.Tensor: negative similarities N
    """
    # only interested by element wise dot product
    dot_prod = torch.diagonal((x1 @ x2.permute(1, 0)), dim1=0, dim2=1)
    # vectors norms
    dm = torch.linalg.norm(x1, dim=1) * torch.linalg.norm(x2, dim=1)
    return -dot_prod / dm

def stack_tensor_from_sam_output(img_anns: List[Dict], key: str="masks") -> torch.Tensor:
    # (Bx2) x max(NA,NB) x (H x W)
    if key:
        return pad_sequence([anns[key] for anns in img_anns], batch_first=True)
    else:
        return pad_sequence(img_anns, batch_first=True)
    
@timeit
def proposal_matching_nms(
    data: MaskData, nms_threshold: float, col_threshold: str = "ci"
) -> MaskData:

    keep_by_nms = batched_nms(
        data["bboxes"].float(),
        data[col_threshold],
        torch.zeros_like(data["bboxes"][:, 0]),  # categories
        iou_threshold=nms_threshold,  # default SAM : 0.7 for iou - for ci need to search
    )
    data.filter(keep_by_nms)

    return data

@timeit
def temporal_matching_torch(
    img_embedding_A: torch.Tensor, img_embedding_B: torch.Tensor, masks: torch.Tensor
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """Compute mask embedding and confidence score for both images for some masks (mt or mt+1)

    Args:
        img_embedding_A torch.Tensor: C x He x We
        img_embedding_B torch.Tensor: C x He x We
        masks torch.Tensor: C x H x W

    Returns:
        Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]: bi-temporal mask embeddings and confidence score
    """
    x_t = compute_mask_embedding(masks, img_embedding_A)
    x_t1 = compute_mask_embedding(masks, img_embedding_B)
    chg_ci = neg_cosine_sim_torch(x_t, x_t1)

    return x_t, x_t1, chg_ci

