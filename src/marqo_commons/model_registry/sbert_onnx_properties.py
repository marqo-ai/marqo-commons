from typing import Dict

from marqo_commons.model_registry.model_properties_object import ModelProperties, VectorNumericType, Modality


class SbertOnnxProperties(ModelProperties):
    vector_numeric_type: VectorNumericType = VectorNumericType.float32
    modality: list[Modality] = [Modality.text]
    type: str = "sbert_onnx"
    tokens: int
    notes: str = ""

def _get_sbert_onnx_properties() -> Dict:
    SBERT_ONNX_MODEL_PROPERTIES = {
        "onnx/all-MiniLM-L6-v1": vars(SbertOnnxProperties(
            name="sentence-transformers/all-MiniLM-L6-v1",
            memory_size=0.7,
            dimensions=384,
            tokens=128,
        )),
        "onnx/all-MiniLM-L6-v2": vars(SbertOnnxProperties(
            name="sentence-transformers/all-MiniLM-L6-v2",
            memory_size=0.7,
            dimensions=384,
            tokens=256,
        )),
        "onnx/all-mpnet-base-v1": vars(SbertOnnxProperties(
            name="sentence-transformers/all-mpnet-base-v1",
            memory_size=0.7,
            dimensions=768,
            tokens=128,
        )),
        "onnx/all-mpnet-base-v2": vars(SbertOnnxProperties(
            name="sentence-transformers/all-mpnet-base-v2",
            memory_size=0.7,
            dimensions=768,
            tokens=128,
        )),
        "onnx/all_datasets_v3_MiniLM-L12": vars(SbertOnnxProperties(
            name="flax-sentence-embeddings/all_datasets_v3_MiniLM-L12",
            memory_size=0.7,
            dimensions=384,
            tokens=128,
        )),
        "onnx/all_datasets_v3_MiniLM-L6": vars(SbertOnnxProperties(
            name="flax-sentence-embeddings/all_datasets_v3_MiniLM-L6",
            memory_size=0.7,
            dimensions=384,
            tokens=128,
        )),
        "onnx/all_datasets_v4_MiniLM-L12": vars(SbertOnnxProperties(
            name="flax-sentence-embeddings/all_datasets_v4_MiniLM-L12",
            memory_size=0.7,
            dimensions=384,
            tokens=128,
        )),
        "onnx/all_datasets_v4_MiniLM-L6": vars(SbertOnnxProperties(
            name="flax-sentence-embeddings/all_datasets_v4_MiniLM-L6",
            memory_size=0.7,
            dimensions=384,
            tokens=128,
        )),
        "onnx/all_datasets_v3_mpnet-base": vars(SbertOnnxProperties(
            name="flax-sentence-embeddings/all_datasets_v3_mpnet-base",
            memory_size=0.7,
            dimensions=768,
            tokens=128,
        )),
        "onnx/all_datasets_v4_mpnet-base": vars(SbertOnnxProperties(
            name="flax-sentence-embeddings/all_datasets_v4_mpnet-base",
            memory_size=0.7,
            dimensions=768,
            tokens=128,
        )),
    }

    return SBERT_ONNX_MODEL_PROPERTIES
