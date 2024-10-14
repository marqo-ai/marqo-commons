"""
OpenCLIP model properties

This file contains properties for OpenCLIP models. It is intended to be used in conjunction with the model registry
and should not be used in isolation.
"""
from typing import Dict, List, Optional

from marqo_commons.model_registry.model_properties_data.onnx_clip_properties import OnnxClipModelProperties
from marqo_commons.model_registry.model_properties_object import ModelProperties, VectorNumericType, Modality, \
    ModelType, T
from marqo_commons.model_registry.utils import convert_model_properties_to_dict


class LanguagebindModelProperties(ModelProperties):
    vector_numeric_type: VectorNumericType = VectorNumericType.float32
    default_memory_size: float = 8
    modality: List[Modality]
    type: ModelType = ModelType.languagebind
    pretrained: Optional[str]
    notes: str = ""

    @classmethod
    def get_all_model_properties_objects(cls) -> Dict[str, T]:
        # use this link to find all the model_configs
        # https://github.com/mlfoundations/open_clip/tree/main/src/open_clip/model_configs
        return {
            'LanguageBind/Video_V1.5_FT_Audio_FT_Image': LanguagebindModelProperties(
                name="LanguageBind/Video_V1.5_FT_Audio_FT_Image",
                dimensions=768,
                modalities=[Modality.video, Modality.audio, Modality.text, Modality.image],
            ),
            'LanguageBind/Video_V1.5_FT_Audio_FT': LanguagebindModelProperties(
                name="LanguageBind/Video_V1.5_FT_Audio_FT",
                dimensions=768,
                modalities=[Modality.video, Modality.audio, Modality.text],
            ),
            'LanguageBind/Video_V1.5_FT_Image': LanguagebindModelProperties(
                name="LanguageBind/Video_V1.5_FT_Image",
                dimensions=768,
                modalities=[Modality.video, Modality.text, Modality.image],
            ),
            'LanguageBind/Audio_FT_Image': LanguagebindModelProperties(
                name="LanguageBind/Audio_FT_Image",
                dimensions=768,
                modalities=[Modality.audio, Modality.text, Modality.image],
            ),
            'LanguageBind/Audio_FT': LanguagebindModelProperties(
                name="LanguageBind/Audio_FT",
                dimensions=768,
                modalities=[Modality.audio, Modality.text],
            ),
            'LanguageBind/Video_V1.5_FT': LanguagebindModelProperties(
                name="LanguageBind/Video_V1.5_FT",
                dimensions=768,
                modalities=[Modality.video, Modality.text],
            ),
        }


@convert_model_properties_to_dict
def _get_languagebind_properties() -> Dict:
    return LanguagebindModelProperties.get_all_model_properties_objects()
