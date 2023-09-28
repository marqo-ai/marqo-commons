from typing import Dict, List

from marqo_commons.model_registry.model_properties_object import ModelProperties, VectorNumericType, Modality, ModelType
from marqo_commons.model_registry.utils import convert_model_properties_to_dict


class SbertOnnxProperties(ModelProperties):
    vector_numeric_type: VectorNumericType = VectorNumericType.float32
    default_memory_size: float = 0.7
    modality: List[Modality] = [Modality.text]
    type: ModelType = ModelType.sbert_onnx
    tokens: int
    notes: str = ""


@convert_model_properties_to_dict
def _get_sbert_onnx_properties() -> Dict:
    SBERT_ONNX_MODEL_PROPERTIES = {
        "onnx/all-MiniLM-L6-v1": SbertOnnxProperties(
            name="sentence-transformers/all-MiniLM-L6-v1",
            dimensions=384,
            tokens=128,
        ),
        "onnx/all-MiniLM-L6-v2": SbertOnnxProperties(
            name="sentence-transformers/all-MiniLM-L6-v2",
            dimensions=384,
            tokens=256,
        ),
        "onnx/all-mpnet-base-v1": SbertOnnxProperties(
            name="sentence-transformers/all-mpnet-base-v1",
            dimensions=768,
            tokens=128,
        ),
        "onnx/all-mpnet-base-v2": SbertOnnxProperties(
            name="sentence-transformers/all-mpnet-base-v2",
            dimensions=768,
            tokens=128,
        ),
        "onnx/all_datasets_v3_MiniLM-L12": SbertOnnxProperties(
            name="flax-sentence-embeddings/all_datasets_v3_MiniLM-L12",
            dimensions=384,
            tokens=128,
        ),
        "onnx/all_datasets_v3_MiniLM-L6": SbertOnnxProperties(
            name="flax-sentence-embeddings/all_datasets_v3_MiniLM-L6",
            dimensions=384,
            tokens=128,
        ),
        "onnx/all_datasets_v4_MiniLM-L12": SbertOnnxProperties(
            name="flax-sentence-embeddings/all_datasets_v4_MiniLM-L12",
            dimensions=384,
            tokens=128,
        ),
        "onnx/all_datasets_v4_MiniLM-L6": SbertOnnxProperties(
            name="flax-sentence-embeddings/all_datasets_v4_MiniLM-L6",
            dimensions=384,
            tokens=128,
        ),
        "onnx/all_datasets_v3_mpnet-base": SbertOnnxProperties(
            name="flax-sentence-embeddings/all_datasets_v3_mpnet-base",
            dimensions=768,
            tokens=128,
        ),
        "onnx/all_datasets_v4_mpnet-base": SbertOnnxProperties(
            name="flax-sentence-embeddings/all_datasets_v4_mpnet-base",
            dimensions=768,
            tokens=128,
        ),
    }

    return SBERT_ONNX_MODEL_PROPERTIES
