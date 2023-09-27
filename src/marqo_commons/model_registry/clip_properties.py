"""
CLIP Model Properties

This file defines properties for CLIP models. It is intended to be used in conjunction with the model registry
and should not be used in isolation.
"""
from marqo_commons.model_registry.model_properties_object import ModelProperties, VectorNumericType, Modality
from typing import Dict, List


class ClipProperties(ModelProperties):
    vector_numeric_type: VectorNumericType = VectorNumericType.float32
    modality: List[Modality] = [Modality.image, Modality.text]
    type: str = "clip"


def _get_clip_properties() -> Dict:
    CLIP_MODEL_PROPERTIES = {
        'RN50': vars(ClipProperties(
            name="RN50",
            memory_size=1,
            dimensions=1024,
            notes="CLIP resnet50",
        )),
        'RN101': vars(ClipProperties(
            name="RN101",
            memory_size=1,
            dimensions=512,
            notes="CLIP resnet101",
        )),
        'RN50x4': vars(ClipProperties(
            name="RN50x4",
            memory_size=1,
            dimensions=640,
            notes="CLIP resnet50x4",
        )),
        'RN50x16': vars(ClipProperties(
            name="RN50x16",
            memory_size=1,
            dimensions=768,
            notes="CLIP resnet50x16",
        )),
        'RN50x64': vars(ClipProperties(
            name="RN50x64",
            memory_size=1,
            dimensions=1024,
            notes="CLIP resnet50x64",
        )),
        'ViT-B/32': vars(ClipProperties(
            name="ViT-B/32",
            memory_size=1,
            dimensions=512,
            notes="CLIP ViT-B/32",
        )),
        'ViT-B/16': vars(ClipProperties(
            name="ViT-B/16",
            memory_size=1,
            dimensions=512,
            notes="CLIP ViT-B/16",
        )),
        'ViT-L/14': vars(ClipProperties(
            name="ViT-L/14",
            memory_size=1.5,
            dimensions=768,
            notes="CLIP ViT-L/14",
        )),
        'ViT-L/14@336px': vars(ClipProperties(
            name="ViT-L/14@336px",
            memory_size=1.5,
            dimensions=768,
            notes="CLIP ViT-L/14@336px",
        )),
    }
    return CLIP_MODEL_PROPERTIES
