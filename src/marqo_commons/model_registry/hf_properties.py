from typing import Dict

from marqo_commons.model_registry.model_properties_object import ModelProperties, VectorNumericType, Modality


class HFModelProperties(ModelProperties):
    vector_numeric_type: VectorNumericType = VectorNumericType.float32
    modality: list[Modality] = [Modality.text]
    type: str = "hf"
    tokens: int
    notes: str = ""


def _get_hf_properties() -> Dict:
    HF_MODEL_PROPERTIES = {
        "hf/all-MiniLM-L6-v1": vars(HFModelProperties(
            name="sentence-transformers/all-MiniLM-L6-v1",
            memory_size=1,
            dimensions=384,
            tokens=128,
        )),
        "hf/all-MiniLM-L6-v2": vars(HFModelProperties(
            name="sentence-transformers/all-MiniLM-L6-v2",
            memory_size=1,
            dimensions=384,
            tokens=256,
        )),
        "hf/all-mpnet-base-v1": vars(HFModelProperties(
            name="sentence-transformers/all-mpnet-base-v1",
            memory_size=1,
            dimensions=768,
            tokens=128,
        )),
        "hf/all-mpnet-base-v2": vars(HFModelProperties(
            name="sentence-transformers/all-mpnet-base-v2",
            memory_size=1,
            dimensions=768,
            tokens=128,
        )),
        "hf/all_datasets_v3_MiniLM-L12": vars(HFModelProperties(
            name="flax-sentence-embeddings/all_datasets_v3_MiniLM-L12",
            memory_size=1,
            dimensions=384,
            tokens=128,
        )),
        "hf/all_datasets_v3_MiniLM-L6": vars(HFModelProperties(
            name="flax-sentence-embeddings/all_datasets_v3_MiniLM-L6",
            memory_size=1,
            dimensions=384,
            tokens=128,
        )),
        "hf/all_datasets_v4_MiniLM-L12": vars(HFModelProperties(
            name="flax-sentence-embeddings/all_datasets_v4_MiniLM-L12",
            memory_size=1,
            dimensions=384,
            tokens=128,
        )),
        "hf/all_datasets_v4_MiniLM-L6": vars(HFModelProperties(
            name="flax-sentence-embeddings/all_datasets_v4_MiniLM-L6",
            memory_size=1,
            dimensions=384,
            tokens=128,
        )),
        "hf/all_datasets_v3_mpnet-base": vars(HFModelProperties(
            name="flax-sentence-embeddings/all_datasets_v3_mpnet-base",
            memory_size=1,
            dimensions=768,
            tokens=128,
        )),
        "hf/all_datasets_v4_mpnet-base": vars(HFModelProperties(
            name="flax-sentence-embeddings/all_datasets_v4_mpnet-base",
            memory_size=1,
            dimensions=768,
            tokens=128,
        )),
        "hf/e5-small": vars(HFModelProperties(
            name="intfloat/e5-small",
            memory_size=1,
            dimensions=384,
            tokens=192,
        )),
        "hf/e5-base": vars(HFModelProperties(
            name="intfloat/e5-base",
            memory_size=1,
            dimensions=768,
            tokens=192,
        )),
        "hf/e5-large": vars(HFModelProperties(
            name="intfloat/e5-large",
            memory_size=1,
            dimensions=1024,
            tokens=192,
        )),
        "hf/e5-large-unsupervised": vars(HFModelProperties(
            name="intfloat/e5-large-unsupervised",
            memory_size=1,
            dimensions=1024,
            tokens=128,
        )),
        "hf/e5-base-unsupervised": vars(HFModelProperties(
            name="intfloat/e5-base-unsupervised",
            memory_size=1,
            dimensions=768,
            tokens=128,
        )),
        "hf/e5-small-unsupervised": vars(HFModelProperties(
            name="intfloat/e5-small-unsupervised",
            memory_size=1,
            dimensions=384,
            tokens=128,
        )),
        "hf/multilingual-e5-small": vars(HFModelProperties(
            name="intfloat/multilingual-e5-small",
            memory_size=1,
            dimensions=384,
            tokens=512,
        )),
        "hf/multilingual-e5-base": vars(HFModelProperties(
            name="intfloat/multilingual-e5-base",
            memory_size=1,
            dimensions=768,
            tokens=512,
        )),
        "hf/multilingual-e5-large": vars(HFModelProperties(
            name="intfloat/multilingual-e5-large",
            memory_size=1,
            dimensions=1024,
            tokens=512,
        )),
        "hf/e5-small-v2": vars(HFModelProperties(
            name="intfloat/e5-small-v2",
            memory_size=1,
            dimensions=384,
            tokens=512,
        )),
        "hf/e5-base-v2": vars(HFModelProperties(
            name="intfloat/e5-base-v2",
            memory_size=1,
            dimensions=768,
            tokens=512,
        )),
        "hf/e5-large-v2": vars(HFModelProperties(
            name="intfloat/e5-large-v2",
            memory_size=1,
            dimensions=1024,
            tokens=512,
        )),
    }
    return HF_MODEL_PROPERTIES
