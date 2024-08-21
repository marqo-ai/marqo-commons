"""
SBERT ONNX Model Properties

This file contains properties for SBERT ONNX models. It is intended to be used in conjunction with the model registry
and should not be used in isolation.
"""
from typing import Dict, List

from marqo_commons.model_registry.model_properties_object import ModelProperties, VectorNumericType, Modality, \
    ModelType, T
from marqo_commons.model_registry.utils import convert_model_properties_to_dict


class SbertOnnxProperties(ModelProperties):
    vector_numeric_type: VectorNumericType = VectorNumericType.float32
    default_memory_size: float = 0.7
    modality: List[Modality] = [Modality.text]
    type: ModelType = ModelType.sbert_onnx
    tokens: int
    notes: str = ""

    @classmethod
    def list_model_properties(cls) -> Dict[str, T]:
        return {
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


@convert_model_properties_to_dict
def _get_sbert_onnx_properties() -> Dict:
    return SbertOnnxProperties.list_model_properties()
