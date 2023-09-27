from typing import Dict

from marqo_commons.model_registry.model_properties_object import ModelProperties, VectorNumericType, Modality, ModelType


class SbertProperties(ModelProperties):
    vector_numeric_type: VectorNumericType = VectorNumericType.float32
    modality: list[Modality] = [Modality.text]
    type: ModelType = ModelType.sbert
    tokens: int
    notes: str = ""


def _get_sbert_properties() -> Dict:
    SBERT_MODEL_PROPERTIES = {
        "sentence-transformers/all-MiniLM-L6-v1": vars(SbertProperties(
            name="sentence-transformers/all-MiniLM-L6-v1",
            memory_size=0.7,
            dimensions=384,
            tokens=128,
        )),
        "sentence-transformers/all-MiniLM-L6-v2": vars(SbertProperties(
            name="sentence-transformers/all-MiniLM-L6-v2",
            memory_size=0.7,
            dimensions=384,
            tokens=256,
        )),
        "sentence-transformers/all-mpnet-base-v1": vars(SbertProperties(
            name="sentence-transformers/all-mpnet-base-v1",
            memory_size=0.7,
            dimensions=768,
            tokens=128,
        )),
        "sentence-transformers/all-mpnet-base-v2": vars(SbertProperties(
            name="sentence-transformers/all-mpnet-base-v2",
            memory_size=0.7,
            dimensions=768,
            tokens=128,
        )),
        "sentence-transformers/stsb-xlm-r-multilingual": vars(SbertProperties(
            name="sentence-transformers/stsb-xlm-r-multilingual",
            memory_size=0.7,
            dimensions=768,
            tokens=128,
        )),
        "flax-sentence-embeddings/all_datasets_v3_MiniLM-L12": vars(SbertProperties(
            name="flax-sentence-embeddings/all_datasets_v3_MiniLM-L12",
            memory_size=0.7,
            dimensions=384,
            tokens=128,
        )),
        "flax-sentence-embeddings/all_datasets_v3_MiniLM-L6": vars(SbertProperties(
            name="flax-sentence-embeddings/all_datasets_v3_MiniLM-L6",
            memory_size=0.7,
            dimensions=384,
            tokens=128,
        )),
        "flax-sentence-embeddings/all_datasets_v4_MiniLM-L12": vars(SbertProperties(
            name="flax-sentence-embeddings/all_datasets_v4_MiniLM-L12",
            memory_size=0.7,
            dimensions=384,
            tokens=128,
        )),
        "flax-sentence-embeddings/all_datasets_v4_MiniLM-L6": vars(SbertProperties(
            name="flax-sentence-embeddings/all_datasets_v4_MiniLM-L6",
            memory_size=0.7,
            dimensions=384,
            tokens=128,
        )),
        "flax-sentence-embeddings/all_datasets_v3_mpnet-base": vars(SbertProperties(
            name="flax-sentence-embeddings/all_datasets_v3_mpnet-base",
            memory_size=0.7,
            dimensions=768,
            tokens=128,
        )),
        "flax-sentence-embeddings/all_datasets_v4_mpnet-base": vars(SbertProperties(
            name="flax-sentence-embeddings/all_datasets_v4_mpnet-base",
            memory_size=0.7,
            dimensions=768,
            tokens=128,
        )),
    }

    return SBERT_MODEL_PROPERTIES
