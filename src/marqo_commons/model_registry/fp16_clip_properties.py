from typing import Dict

from marqo_commons.model_registry.model_properties_object import ModelProperties, VectorNumericType, Modality, ModelType


class FP16ClipModelProperties(ModelProperties):
    vector_numeric_type: VectorNumericType = VectorNumericType.float32
    modality: list[Modality] = [Modality.image, Modality.text]
    type: ModelType = ModelType.fp16_clip


def _get_fp16_clip_properties() -> Dict:
    FP16_CLIP_MODEL_PROPERTIES = {
        "fp16/ViT-L/14": vars(FP16ClipModelProperties(
            name="fp16/ViT-L/14",
            memory_size=1.5,
            dimensions=768,
            notes="The faster version (fp16, load from `cuda`) of openai clip model",
        )),
        'fp16/ViT-B/32': vars(FP16ClipModelProperties(
            name="fp16/ViT-B/32",
            memory_size=0.66,
            dimensions=512,
            notes="The faster version (fp16, load from `cuda`) of openai clip model",
        )),
        'fp16/ViT-B/16': vars(FP16ClipModelProperties(
            name="fp16/ViT-B/16",
            memory_size=0.66,
            dimensions=512,
            notes="The faster version (fp16, load from `cuda`) of openai clip model",
        )),
    }

    return FP16_CLIP_MODEL_PROPERTIES
