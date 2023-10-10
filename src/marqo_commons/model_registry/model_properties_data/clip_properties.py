"""
CLIP Model Properties

This file defines properties for CLIP models. It is intended to be used in conjunction with the model registry
and should not be used in isolation.
"""
from marqo_commons.model_registry.model_properties_object import ModelProperties, VectorNumericType, Modality, ModelType
from typing import Dict, List
from marqo_commons.model_registry.utils import convert_model_properties_to_dict


class ClipProperties(ModelProperties):
    vector_numeric_type: VectorNumericType = VectorNumericType.float32
    modality: List[Modality] = [Modality.image, Modality.text]
    default_memory_size: float = 1.0
    type: ModelType = ModelType.clip


@convert_model_properties_to_dict
def _get_clip_properties() -> Dict:
    CLIP_MODEL_PROPERTIES = {
        'RN50': ClipProperties(
            name="RN50",
            dimensions=1024,
            notes="CLIP resnet50",
        ),
        'RN101': ClipProperties(
            name="RN101",
            dimensions=512,
            notes="CLIP resnet101",
        ),
        'RN50x4': ClipProperties(
            name="RN50x4",
            dimensions=640,
            notes="CLIP resnet50x4",
        ),
        'RN50x16': ClipProperties(
            name="RN50x16",
            dimensions=768,
            notes="CLIP resnet50x16",
        ),
        'RN50x64': ClipProperties(
            name="RN50x64",
            dimensions=1024,
            notes="CLIP resnet50x64",
        ),
        'ViT-B/32': ClipProperties(
            name="ViT-B/32",
            dimensions=512,
            notes="CLIP ViT-B/32",
        ),
        'ViT-B/16': ClipProperties(
            name="ViT-B/16",
            dimensions=512,
            notes="CLIP ViT-B/16",
        ),
        'ViT-L/14': ClipProperties(
            name="ViT-L/14",
            dimensions=768,
            notes="CLIP ViT-L/14",
        ),
        'ViT-L/14@336px': ClipProperties(
            name="ViT-L/14@336px",
            dimensions=768,
            notes="CLIP ViT-L/14@336px",
        ),
    }
    return CLIP_MODEL_PROPERTIES
