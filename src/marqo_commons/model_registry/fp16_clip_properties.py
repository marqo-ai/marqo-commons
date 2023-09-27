from typing import Dict, List

from marqo_commons.model_registry.model_properties_object import ModelProperties, VectorNumericType, Modality, ModelType


class FP16ClipModelProperties(ModelProperties):
    vector_numeric_type: VectorNumericType = VectorNumericType.float32
    modality: List[Modality] = [Modality.image, Modality.text]
    type: ModelType = ModelType.fp16_clip


def _get_fp16_clip_properties() -> Dict:
    FP16_CLIP_MODEL_PROPERTIES = {
        "fp16/ViT-L/14": FP16ClipModelProperties(
            name="fp16/ViT-L/14",
            dimensions=768,
            notes="The faster version (fp16, load from `cuda`) of openai clip model",
        ).to_dict(),
        'fp16/ViT-B/32': FP16ClipModelProperties(
            name="fp16/ViT-B/32",
            dimensions=512,
            notes="The faster version (fp16, load from `cuda`) of openai clip model",
        ).to_dict(),
        'fp16/ViT-B/16': FP16ClipModelProperties(
            name="fp16/ViT-B/16",
            dimensions=512,
            notes="The faster version (fp16, load from `cuda`) of openai clip model",
        ).to_dict(),
    }

    return FP16_CLIP_MODEL_PROPERTIES
