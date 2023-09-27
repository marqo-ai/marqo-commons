from typing import Dict, List

from marqo_commons.model_registry.model_properties_object import ModelProperties, VectorNumericType, Modality, ModelType


class SbertProperties(ModelProperties):
    vector_numeric_type: VectorNumericType = VectorNumericType.float32
    default_memory_size: float = 0.7
    modality: List[Modality] = [Modality.text]
    type: ModelType = ModelType.sbert
    tokens: int
    notes: str = ""


def _get_sbert_properties() -> Dict:
    SBERT_MODEL_PROPERTIES = {
        "sentence-transformers/all-MiniLM-L6-v1": SbertProperties(
            name="sentence-transformers/all-MiniLM-L6-v1",
            dimensions=384,
            tokens=128,
        ).to_dict(),
        "sentence-transformers/all-MiniLM-L6-v2": SbertProperties(
            name="sentence-transformers/all-MiniLM-L6-v2",
            dimensions=384,
            tokens=256,
        ).to_dict(),
        "sentence-transformers/all-mpnet-base-v1": SbertProperties(
            name="sentence-transformers/all-mpnet-base-v1",
            dimensions=768,
            tokens=128,
        ).to_dict(),
        "sentence-transformers/all-mpnet-base-v2": SbertProperties(
            name="sentence-transformers/all-mpnet-base-v2",
            dimensions=768,
            tokens=128,
        ).to_dict(),
        "sentence-transformers/stsb-xlm-r-multilingual": SbertProperties(
            name="sentence-transformers/stsb-xlm-r-multilingual",
            dimensions=768,
            tokens=128,
        ).to_dict(),
        "flax-sentence-embeddings/all_datasets_v3_MiniLM-L12": SbertProperties(
            name="flax-sentence-embeddings/all_datasets_v3_MiniLM-L12",
            dimensions=384,
            tokens=128,
        ).to_dict(),
        "flax-sentence-embeddings/all_datasets_v3_MiniLM-L6": SbertProperties(
            name="flax-sentence-embeddings/all_datasets_v3_MiniLM-L6",
            dimensions=384,
            tokens=128,
        ).to_dict(),
        "flax-sentence-embeddings/all_datasets_v4_MiniLM-L12": SbertProperties(
            name="flax-sentence-embeddings/all_datasets_v4_MiniLM-L12",
            dimensions=384,
            tokens=128,
        ).to_dict(),
        "flax-sentence-embeddings/all_datasets_v4_MiniLM-L6": SbertProperties(
            name="flax-sentence-embeddings/all_datasets_v4_MiniLM-L6",
            dimensions=384,
            tokens=128,
        ).to_dict(),
        "flax-sentence-embeddings/all_datasets_v3_mpnet-base": SbertProperties(
            name="flax-sentence-embeddings/all_datasets_v3_mpnet-base",
            dimensions=768,
            tokens=128,
        ).to_dict(),
        "flax-sentence-embeddings/all_datasets_v4_mpnet-base": SbertProperties(
            name="flax-sentence-embeddings/all_datasets_v4_mpnet-base",
            dimensions=768,
            tokens=128,
        ).to_dict(),
    }

    return SBERT_MODEL_PROPERTIES
