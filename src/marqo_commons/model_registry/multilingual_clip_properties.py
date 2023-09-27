from typing import Dict, List

from marqo_commons.model_registry.model_properties_object import ModelProperties, VectorNumericType, Modality, ModelType


class MultilingualClipModelProperties(ModelProperties):
    vector_numeric_type: VectorNumericType = VectorNumericType.float32
    modality: List[Modality] = [Modality.text, Modality.image]
    type: ModelType = ModelType.multilingual_clip
    visual_model: str
    textual_model: str
    notes: str = ""


def _get_multilingual_clip_properties() -> Dict:
    """This is moved here from the model registry to avoid a circular import"""
    # Models are from github repo
    # https://github.com/FreddeFrallan/Multilingual-CLIP
    MULTILINGUAL_CLIP_PROPERTIES = {
        "multilingual-clip/XLM-Roberta-Large-Vit-L-14": vars(MultilingualClipModelProperties(
            name="multilingual-clip/XLM-Roberta-Large-Vit-L-14",
            memory_size=1.5,
            visual_model="openai/ViT-L/14",
            textual_model="M-CLIP/XLM-Roberta-Large-Vit-L-14",
            dimensions=768,
        )),
        "multilingual-clip/XLM-R Large Vit-B/16+": vars(MultilingualClipModelProperties(
            name="multilingual-clip/XLM-R Large Vit-B/16+",
            memory_size=5,
            visual_model="open_clip/ViT-B-16-plus-240/laion400m_e32",
            textual_model="M-CLIP/XLM-Roberta-Large-Vit-B-16Plus",
            dimensions=640,
        )),
        "multilingual-clip/XLM-Roberta-Large-Vit-B-32": vars(MultilingualClipModelProperties(
            name="multilingual-clip/XLM-Roberta-Large-Vit-B-32",
            memory_size=5,
            visual_model="openai/ViT-B/32",
            textual_model="M-CLIP/XLM-Roberta-Large-Vit-B-32",
            dimensions=512,
        )),
        "multilingual-clip/LABSE-Vit-L-14": vars(MultilingualClipModelProperties(
            name="multilingual-clip/LABSE-Vit-L-14",
            memory_size=1.5,
            visual_model="openai/ViT-L/14",
            textual_model="M-CLIP/LABSE-Vit-L-14",
            dimensions=768,
        )),
    }

    return MULTILINGUAL_CLIP_PROPERTIES
