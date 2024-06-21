from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence, Union
import numpy as np
from commons.constants import NamedDataset
from segment_any_change.config_run import (
    ExperimentParams,
    choose_model,
    load_exp_params,
)
from segment_any_change.eval import MetricEngine
from torchmetrics import Metric
from tqdm import tqdm
from segment_any_change.utils import  load_img
import torch
from magic_pen.data.loader import BiTemporalDataset
from magic_pen.data.process import DefaultTransform, generate_prompt
from magic_pen.utils_io import load_levircd_sample
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinaryJaccardIndex
)

@dataclass
class DataSample:
    A_path: str
    B_path: str
    label_path: str


def load_partial_ds(ds_name: str, dtype,  params=None, indices: Sequence[int] = None):
    ds = BiTemporalDataset(name=ds_name, dtype=dtype, transform=DefaultTransform(), params=params)
    if indices is not None and any(indices):
        ds = torch.utils.data.Subset(ds, indices)
    print(f"DATASET SUBSET : {len(ds)}")

    return ds

class SampleDataset(Dataset):
    def __init__(self, data_dict: Dict, name: str, transform: Any, params: ExperimentParams):
        """Generate Torch Dataset for sample

        Args:
            data_dict (Dict): sample data : img_A, img_B, label
            name (str): _description_
            transform (Any): _description_

        Raises:
            RuntimeError: _description_
        """
        self.sample = data_dict
        self.name = name
        self.transform = transform
        self.params = params

        if name != NamedDataset.LEVIR_CD.value:
            # need to implement custom loading / processing
            raise RuntimeError("only levircd is implemented")

    def __len__(self):
        return 1

    def __getitem__(self, idx):    

    # add generation prompts based on each label zone
        if isinstance(self.sample["img_A"], str):
            self.sample["img_A"] = load_img(self.sample["img_A"])
            self.sample["img_B"] = load_img(self.sample["img_B"])
            self.sample["label"] = load_img(self.sample["label"])

        
        
        if self.transform:
            self.sample = self.transform(self.sample)

        prompt_coords, prompt_labels = generate_prompt(self.sample["label"], self.params.prompt_type, self.params.n_prompt, **asdict(self.params))
        self.sample  = self.sample  | dict(index=0, point_coords=prompt_coords, point_labels=prompt_labels)

        return self.sample 

def infer_on_sample(
    A_path: Any, B_path: Any, label_path: Any, model: Any, params
) -> Dict[str, Any]:

    ds = SampleDataset(data_dict=dict(
        img_A=A_path, 
        img_B=B_path, 
        label=label_path),
        name = "levir-cd",
        transform=DefaultTransform(),
        params=params
        )

    dloader = DataLoader(ds, batch_size=1, shuffle=False)

    outputs = []
    batch = next(iter(dloader))
    with torch.no_grad():
        outputs.append({"pred": model(batch), "batch": batch})
    return outputs[0]


def partial_inference(
    params: ExperimentParams,
    ds_dtype: str = "test",
    indices: Sequence[int] = None,
    dummy: bool = False,
    return_batch: bool = False,
    in_metrics: List[Metric] = None,
) -> List[Dict]:

    if params is None:
        params = load_exp_params(**asdict(params))

    model = choose_model(is_debug=dummy, params=params)

    ds = load_partial_ds(params.ds_name, ds_dtype, params, indices)
    if in_metrics is None:
        in_metrics = [  BinaryF1Score(),
                        BinaryPrecision(),
                        BinaryRecall(),
                        BinaryJaccardIndex()]

    engine_eval = MetricEngine(in_metrics=in_metrics, name="classif")

    dloader = torch.utils.data.DataLoader(
        ds, batch_size=params.batch_size, shuffle=False
    )
    outputs = []
    for i, batch in tqdm(enumerate(dloader), total=len(dloader), desc="Processing"):
        with torch.no_grad():
            preds = model(batch)
            metrics = engine_eval(preds, batch["label"])
            out = {
                "pred": preds,
                "metrics": metrics,
                "batch": batch if return_batch else None,
            }
            outputs.append(out)
    return outputs


if __name__ == "__main__":
    pair_img = load_levircd_sample(1, seed=8)
    path_label,path_A, path_B = pair_img.iloc[0]

    params = dict(
        batch_size=2,
        ds_name="levir-cd",
        dev=True,
        n_job_by_node=2,
        th_change_proposals=50,
        prompt_type="grid",
        n_prompt=1024,
        loc="random",
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95
    )
    params = load_exp_params(**params)
    print(params)
    model = choose_model(is_debug=False, params=params)

    output = infer_on_sample(A_path=path_A,
                    B_path=path_B, 
                    label_path=path_label,
                    params=params,
                    model=model)

    # outputs = partial_inference(
    #     ds_name="levir-cd",
    #     dtype="test",
    #     indices=np.arange(4),
    #     dev=True,
    #     dummy=True,
    #     return_batch=True,
    # )

    # output = infer_on_sample(
    #     A_path=path_A, B_path=path_B, label_path=path_label, model=None
    # )

    # from segment_any_change.masks.mask_generator import SegAnyMaskGenerator
    # params = load_default_sam_params()

    # sam = load_sam(
    #             model_type=params.model_type,
    #             model_cls=BiSam,
    #             version="dev",
    #             device=DEVICE
    #             )

    # item = DataSample(A_path=path_A,
    #                     B_path=path_B,
    #                     label_path=path_label)

    # ds = BiTemporalDataset(items=item, transform=DefaultTransform())
    # dloader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
    # batch = next(iter(dloader))

    # mask_generator = SegAnyMaskGenerator(model=sam,
    #                                 points_per_side=params.points_per_side,
    #                                 points_per_batch=params.points_per_batch,
    #                                 pred_iou_thresh=params.pred_iou_thresh,
    #                                 stability_score_thresh=params.stability_score_thresh,
    #                                 stability_score_offset=params.stability_score_offset,
    #                                 box_nms_thresh=params.box_nms_thresh,
    #                                 min_mask_region_area=params.min_mask_region_area)

    # img_anns = mask_generator.generate(batch)

    # import pickle

    # with open(f"tmp/generator_return_{params.model_type}.pkl", "wb") as fp:
    #     pickle.dump(img_anns, fp)
    #     print("masks dumped ! ")
