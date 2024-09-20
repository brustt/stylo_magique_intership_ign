from dataclasses import asdict

from commons.utils_io import load_sam

from src.commons.constants import DEVICE
from src.commons.constants import NamedModels
from src.models.segment_any_change.config_run import ExperimentParams
from src.models.segment_any_change.matching import BitemporalMatching
from src.models.segment_any_change.query_prompt import SegAnyPrompt

from ..segment_any_change.model import BiSam


def choose_model(params: ExperimentParams):

    if params.model_name == NamedModels.SEGANYMATCHING.value:

        sam = load_sam(
            model_type=params.model_type, model_cls=BiSam, version="dev", device=DEVICE
        )

        return BitemporalMatching(model=sam, **asdict(params))

    elif params.model_name == NamedModels.SEGANYPROMPT.value:

        sam = load_sam(
            model_type=params.model_type, model_cls=BiSam, version="dev", device=DEVICE
        )

        matching_engine = BitemporalMatching(model=sam, **asdict(params))

        return SegAnyPrompt(matching_engine, **asdict(params))
    else:
        raise RuntimeError(f"Model {params.model_name} not known")
