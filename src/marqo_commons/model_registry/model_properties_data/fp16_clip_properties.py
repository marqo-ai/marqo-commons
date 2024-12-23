"""
FP16 Clip Model Properties

This file defines properties for FP16 Clip models. It is intended to be used in conjunction with the model registry
and should not be used in isolation.
"""

from typing import Dict, List

from marqo_commons.model_registry.model_properties_object import (
    Modality,
    ModelProperties,
    ModelType,
    VectorNumericType,
)
from marqo_commons.model_registry.utils import convert_model_properties_to_dict


class FP16ClipModelProperties(ModelProperties):
    vector_numeric_type: VectorNumericType = VectorNumericType.float32
    modality: List[Modality] = [Modality.image, Modality.text]
    type: ModelType = ModelType.fp16_clip

    @classmethod
    def get_all_model_properties_objects(cls) -> Dict[str, "FP16ClipModelProperties"]:
        return {
            "fp16/ViT-L/14": FP16ClipModelProperties(
                name="fp16/ViT-L/14",
                dimensions=768,
                notes="The faster version (fp16, load from `cuda`) of openai clip model",
            ),
            "fp16/ViT-B/32": FP16ClipModelProperties(
                name="fp16/ViT-B/32",
                dimensions=512,
                notes="The faster version (fp16, load from `cuda`) of openai clip model",
            ),
            "fp16/ViT-B/16": FP16ClipModelProperties(
                name="fp16/ViT-B/16",
                dimensions=512,
                notes="The faster version (fp16, load from `cuda`) of openai clip model",
            ),
        }


@convert_model_properties_to_dict
def _get_fp16_clip_properties() -> Dict:
    return FP16ClipModelProperties.get_all_model_properties_objects()
