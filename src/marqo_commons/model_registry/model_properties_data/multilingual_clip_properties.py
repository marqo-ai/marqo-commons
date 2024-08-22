"""
Multilingual CLIP model properties

This file defines properties for Multilingual CLIP models.
It is intended to be used in conjunction with the model registry
and should not be used in isolation.
"""
from typing import Dict, List

from marqo_commons.model_registry.model_properties_object import ModelProperties, VectorNumericType, Modality, \
    ModelType, T
from marqo_commons.model_registry.utils import convert_model_properties_to_dict


class MultilingualClipModelProperties(ModelProperties):
    vector_numeric_type: VectorNumericType = VectorNumericType.float32
    default_memory_size: float = 5.0
    modality: List[Modality] = [Modality.text, Modality.image]
    type: ModelType = ModelType.multilingual_clip
    visual_model: str
    textual_model: str
    notes: str = ""

    @classmethod
    def get_all_model_properties_objects(cls) -> Dict[str, T]:
        """This is moved here from the model registry to avoid a circular import"""
        # Models are from github repo
        # https://github.com/FreddeFrallan/Multilingual-CLIP
        return {
            "multilingual-clip/XLM-Roberta-Large-Vit-L-14": MultilingualClipModelProperties(
                name="multilingual-clip/XLM-Roberta-Large-Vit-L-14",
                visual_model="openai/ViT-L/14",
                textual_model="M-CLIP/XLM-Roberta-Large-Vit-L-14",
                dimensions=768,
            ),
            "multilingual-clip/XLM-R Large Vit-B/16+": MultilingualClipModelProperties(
                name="multilingual-clip/XLM-R Large Vit-B/16+",
                visual_model="open_clip/ViT-B-16-plus-240/laion400m_e32",
                textual_model="M-CLIP/XLM-Roberta-Large-Vit-B-16Plus",
                dimensions=640,
            ),
            "multilingual-clip/XLM-Roberta-Large-Vit-B-32": MultilingualClipModelProperties(
                name="multilingual-clip/XLM-Roberta-Large-Vit-B-32",
                visual_model="openai/ViT-B/32",
                textual_model="M-CLIP/XLM-Roberta-Large-Vit-B-32",
                dimensions=512,
            ),
            "multilingual-clip/LABSE-Vit-L-14": MultilingualClipModelProperties(
                name="multilingual-clip/LABSE-Vit-L-14",
                visual_model="openai/ViT-L/14",
                textual_model="M-CLIP/LABSE-Vit-L-14",
                dimensions=768,
            ),
        }


@convert_model_properties_to_dict
def _get_multilingual_clip_properties() -> Dict:
    return MultilingualClipModelProperties.get_all_model_properties_objects()
