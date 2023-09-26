from typing import Dict

from marqo_commons.model_registry.model_properties_object import ModelProperties, VectorNumericType, Modality


class FP16ClipModelProperties(ModelProperties):
    vector_numeric_type: VectorNumericType = VectorNumericType.float32
    modality: list[Modality] = [Modality.image, Modality.text]
    type: str = "fp16_clip"
    memory_size: float = 0  # TODO: add memory size


def _get_fp16_clip_properties() -> Dict:
    FP16_CLIP_MODEL_PROPERTIES = {
        "fp16/ViT-L/14": FP16ClipModelProperties(
            name="fp16/ViT-L/14",
            dimensions=768,
            notes="The faster version (fp16, load from `cuda`) of openai clip model",
        ).__dict__,
        'fp16/ViT-B/32': FP16ClipModelProperties(
            name="fp16/ViT-B/32",
            dimensions=512,
            notes="The faster version (fp16, load from `cuda`) of openai clip model",
        ).__dict__,
        'fp16/ViT-B/16': FP16ClipModelProperties(
            name="fp16/ViT-B/16",
            dimensions=512,
            notes="The faster version (fp16, load from `cuda`) of openai clip model",
        ).__dict__,
    }

    return FP16_CLIP_MODEL_PROPERTIES
