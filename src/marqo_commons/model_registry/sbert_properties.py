from typing import Dict

from marqo_commons.model_registry.model_properties_object import ModelProperties, VectorNumericType, Modality


class SbertProperties(ModelProperties):
    vector_numeric_type: VectorNumericType = VectorNumericType.float32
    modality: list[Modality] = [Modality.text, Modality.image]
    type: str = "random"
    memory_size: int = 0  # TODO: add memory size
    tokens: int
    notes: str = ""


def _get_sbert_properties() -> Dict:
    SBERT_MODEL_PROPERTIES = {
        "sentence-transformers/all-MiniLM-L6-v1": SbertProperties(
            name="sentence-transformers/all-MiniLM-L6-v1",
            dimensions=384,
            tokens=128,
        ).__dict__,
        "sentence-transformers/all-MiniLM-L6-v2": SbertProperties(
            name="sentence-transformers/all-MiniLM-L6-v2",
            dimensions=384,
            tokens=256,
        ).__dict__,
        "sentence-transformers/all-mpnet-base-v1": SbertProperties(
            name="sentence-transformers/all-mpnet-base-v1",
            dimensions=768,
            tokens=128,
        ).__dict__,
        "sentence-transformers/all-mpnet-base-v2": SbertProperties(
            name="sentence-transformers/all-mpnet-base-v2",
            dimensions=768,
            tokens=128,
        ).__dict__,
        "sentence-transformers/stsb-xlm-r-multilingual": SbertProperties(
            name="sentence-transformers/stsb-xlm-r-multilingual",
            dimensions=768,
            tokens=128,
        ).__dict__,
        "flax-sentence-embeddings/all_datasets_v3_MiniLM-L12": SbertProperties(
            name="flax-sentence-embeddings/all_datasets_v3_MiniLM-L12",
            dimensions=384,
            tokens=128,
        ).__dict__,
        "flax-sentence-embeddings/all_datasets_v3_MiniLM-L6": SbertProperties(
            name="flax-sentence-embeddings/all_datasets_v3_MiniLM-L6",
            dimensions=384,
            tokens=128,
        ).__dict__,
        "flax-sentence-embeddings/all_datasets_v4_MiniLM-L12": SbertProperties(
            name="flax-sentence-embeddings/all_datasets_v4_MiniLM-L12",
            dimensions=384,
            tokens=128,
        ).__dict__,
        "flax-sentence-embeddings/all_datasets_v4_MiniLM-L6": SbertProperties(
            name="flax-sentence-embeddings/all_datasets_v4_MiniLM-L6",
            dimensions=384,
            tokens=128,
        ).__dict__,
        "flax-sentence-embeddings/all_datasets_v3_mpnet-base": SbertProperties(
            name="flax-sentence-embeddings/all_datasets_v3_mpnet-base",
            dimensions=768,
            tokens=128,
        ).__dict__,
        "flax-sentence-embeddings/all_datasets_v4_mpnet-base": SbertProperties(
            name="flax-sentence-embeddings/all_datasets_v4_mpnet-base",
            dimensions=768,
            tokens=128,
        ).__dict__,
    }

    return SBERT_MODEL_PROPERTIES
