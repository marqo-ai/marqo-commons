"""
CLIP Model Properties

This file defines properties for CLIP models. It is intended to be used in conjunction with the model registry
and should not be used in isolation.
"""
from marqo_commons.model_registry.model_properties_object import ModelProperties, VectorNumericType, Modality
from typing import Dict


class ClipProperties(ModelProperties):
    vector_numeric_type: VectorNumericType = VectorNumericType.float32
    modality: list[Modality] = [Modality.image, Modality.text]
    type: str = "clip"
    memory_size: int = 0  # TODO: add memory size


def _get_clip_properties() -> Dict:
    CLIP_MODEL_PROPERTIES = {
        'RN50': ClipProperties(
            name="RN50",
            dimensions=1024,
            notes="CLIP resnet50",
        ).__dict__,
        'RN101': ClipProperties(
            name="RN101",
            dimensions=512,
            notes="CLIP resnet101",
        ).__dict__,
        'RN50x4': ClipProperties(
            name="RN50x4",
            dimensions=640,
            notes="CLIP resnet50x4",
        ).__dict__,
        'RN50x16': ClipProperties(
            name="RN50x16",
            dimensions=768,
            notes="CLIP resnet50x16",
        ).__dict__,
        'RN50x64': ClipProperties(
            name="RN50x64",
            dimensions=1024,
            notes="CLIP resnet50x64",
        ).__dict__,
        'ViT-B/32': ClipProperties(
            name="ViT-B/32",
            dimensions=512,
            notes="CLIP ViT-B/32",
        ).__dict__,
        'ViT-B/16': ClipProperties(
            name="ViT-B/16",
            dimensions=512,
            notes="CLIP ViT-B/16",
        ).__dict__,
        'ViT-L/14': ClipProperties(
            name="ViT-L/14",
            dimensions=768,
            notes="CLIP ViT-L/14",
        ).__dict__,
        'ViT-L/14@336px': ClipProperties(
            name="ViT-L/14@336px",
            dimensions=768,
            notes="CLIP ViT-L/14@336px",
        ).__dict__,
    }
    return CLIP_MODEL_PROPERTIES
