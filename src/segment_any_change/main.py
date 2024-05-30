from typing import Any, List, Optional, Tuple, Union
import torch
from segment_any_change.eval import PX_METRICS, MetricEngine
from segment_any_change.matching import BitemporalMatching
from segment_any_change.tensorboard_callback import CustomWriter, TensorBoardCallbackLogger
from src.magic_pen.dummy import DummyModel
from magic_pen.config import DEVICE
import pytorch_lightning as pl
from pytorch_lightning.profilers import PyTorchProfiler
from magic_pen.data.datamodule import CDDataModule
from segment_any_change.model import BiSam
from segment_any_change.task import CDModule
from segment_any_change.utils import flush_memory, load_sam
import logging
from pytorch_lightning.loggers import TensorBoardLogger

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
    batch_size=2
    model_type="vit_b"
    output_dir = "lightning_logs/predictions"

    logger = TensorBoardLogger(save_dir="lightning_logs", name="pred_v0")

    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/profiler0"),
        schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=20) # parameters meaning ?
    )

    model = DummyModel(3, 1).to(DEVICE)

    # sam = load_sam(
    #     model_type=model_type, 
    #     model_cls=BiSam,
    #     version="dev", 
    #     device=DEVICE
    #     )
    # model = BitemporalMatching(sam, filter_method=filter_change_proposals, **sam_params)


    pl_module = CDModule(model=model, metrics=PX_METRICS)
    
    dm = CDDataModule(name="levir-cd", batch_size=batch_size)

    trainer = pl.Trainer(
        logger=logger,
        accelerator=DEVICE,
        profiler=profiler,
        callbacks=[TensorBoardCallbackLogger(), CustomWriter(output_dir, write_interval="epoch")]
    )

    trainer.predict(pl_module, dm)






