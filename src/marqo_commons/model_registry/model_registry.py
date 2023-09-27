from typing import Dict
import json

from marqo_commons.model_registry.sbert_properties import _get_sbert_properties
from marqo_commons.model_registry.sbert_onnx_properties import _get_sbert_onnx_properties
from marqo_commons.model_registry.clip_properties import _get_clip_properties
from marqo_commons.model_registry.open_clip_properties import _get_open_clip_properties
from marqo_commons.model_registry.multilingual_clip_properties import _get_multilingual_clip_properties
from marqo_commons.model_registry.test_properties import _get_sbert_test_properties
from marqo_commons.model_registry.hf_properties import _get_hf_properties
from marqo_commons.model_registry.fp16_clip_properties import _get_fp16_clip_properties
from marqo_commons.model_registry.onnx_clip_properties import _get_onnx_clip_properties
from marqo_commons.model_registry.random_properties import _get_random_properties


# we need to keep track of the embed dim and model load functions/classes
# we can use this as a registry

def get_model_properties_dict() -> Dict:
    # also truncate the name if not already
    sbert_model_properties = _get_sbert_properties()
    sbert_model_properties.update({k.split('/')[-1]:v for k,v in sbert_model_properties.items()})

    sbert_onnx_model_properties = _get_sbert_onnx_properties()

    clip_model_properties = _get_clip_properties()
    test_model_properties = _get_sbert_test_properties()
    random_model_properties = _get_random_properties()
    hf_model_properties = _get_hf_properties()
    open_clip_model_properties = _get_open_clip_properties()
    onnx_clip_model_properties = _get_onnx_clip_properties()
    multilingual_clip_model_properties = _get_multilingual_clip_properties()
    fp16_clip_model_properties = _get_fp16_clip_properties()

    # combine the above dicts
    model_properties = dict(clip_model_properties.items())
    model_properties.update(sbert_model_properties)
    model_properties.update(test_model_properties)
    model_properties.update(sbert_onnx_model_properties)
    model_properties.update(random_model_properties)
    model_properties.update(hf_model_properties)
    model_properties.update(open_clip_model_properties)
    model_properties.update(onnx_clip_model_properties)
    model_properties.update(multilingual_clip_model_properties)
    model_properties.update(fp16_clip_model_properties)

    return model_properties


def get_model_properties_json() -> str:
    return json.dumps(get_model_properties_dict())
