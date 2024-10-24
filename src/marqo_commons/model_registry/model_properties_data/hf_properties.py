"""
HF Model Properties

This file defines properties for HF models. It is intended to be used in conjunction with the model registry
and should not be used in isolation.
"""

from typing import Dict, List

from marqo_commons.model_registry.model_properties_object import (
    Modality,
    ModelProperties,
    ModelType,
    VectorNumericType,
)
from marqo_commons.model_registry.utils import convert_model_properties_to_dict


class HFModelProperties(ModelProperties):
    vector_numeric_type: VectorNumericType = VectorNumericType.float32
    modality: List[Modality] = [Modality.text]
    type: ModelType = ModelType.hf
    default_memory_size: float = 1.0
    tokens: int
    notes: str = ""
    text_query_prefix: str = ""
    text_chunk_prefix: str = ""
    trustRemoteCode: bool = False

    @classmethod
    def get_all_model_properties_objects(cls) -> Dict[str, "HFModelProperties"]:
        return {
            "hf/all-MiniLM-L6-v1": HFModelProperties(
                name="sentence-transformers/all-MiniLM-L6-v1",
                dimensions=384,
                tokens=128,
            ),
            "hf/all-MiniLM-L6-v2": HFModelProperties(
                name="sentence-transformers/all-MiniLM-L6-v2",
                dimensions=384,
                tokens=256,
            ),
            "hf/all-mpnet-base-v1": HFModelProperties(
                name="sentence-transformers/all-mpnet-base-v1",
                dimensions=768,
                tokens=128,
            ),
            "hf/all-mpnet-base-v2": HFModelProperties(
                name="sentence-transformers/all-mpnet-base-v2",
                dimensions=768,
                tokens=128,
            ),
            "hf/all_datasets_v3_MiniLM-L12": HFModelProperties(
                name="flax-sentence-embeddings/all_datasets_v3_MiniLM-L12",
                dimensions=384,
                tokens=128,
            ),
            "hf/all_datasets_v3_MiniLM-L6": HFModelProperties(
                name="flax-sentence-embeddings/all_datasets_v3_MiniLM-L6",
                dimensions=384,
                tokens=128,
            ),
            "hf/all_datasets_v4_MiniLM-L12": HFModelProperties(
                name="flax-sentence-embeddings/all_datasets_v4_MiniLM-L12",
                dimensions=384,
                tokens=128,
            ),
            "hf/all_datasets_v4_MiniLM-L6": HFModelProperties(
                name="flax-sentence-embeddings/all_datasets_v4_MiniLM-L6",
                dimensions=384,
                tokens=128,
            ),
            "hf/all_datasets_v3_mpnet-base": HFModelProperties(
                name="flax-sentence-embeddings/all_datasets_v3_mpnet-base",
                dimensions=768,
                tokens=128,
            ),
            "hf/all_datasets_v4_mpnet-base": HFModelProperties(
                name="flax-sentence-embeddings/all_datasets_v4_mpnet-base",
                dimensions=768,
                tokens=128,
            ),
            "hf/e5-small": HFModelProperties(
                name="intfloat/e5-small",
                dimensions=384,
                tokens=192,
                memory_size=0.1342,
                text_query_prefix="query: ",
                text_chunk_prefix="passage: ",
            ),
            "hf/e5-base": HFModelProperties(
                name="intfloat/e5-base",
                dimensions=768,
                tokens=192,
                memory_size=0.438,
                text_query_prefix="query: ",
                text_chunk_prefix="passage: ",
            ),
            "hf/e5-large": HFModelProperties(
                name="intfloat/e5-large",
                dimensions=1024,
                tokens=192,
                memory_size=1.3,
                text_query_prefix="query: ",
                text_chunk_prefix="passage: ",
            ),
            "hf/e5-large-unsupervised": HFModelProperties(
                name="intfloat/e5-large-unsupervised",
                dimensions=1024,
                tokens=128,
                memory_size=1.3,
                text_query_prefix="query: ",
                text_chunk_prefix="passage: ",
            ),
            "hf/e5-base-unsupervised": HFModelProperties(
                name="intfloat/e5-base-unsupervised",
                dimensions=768,
                tokens=128,
                memory_size=0.438,
                text_query_prefix="query: ",
                text_chunk_prefix="passage: ",
            ),
            "hf/e5-small-unsupervised": HFModelProperties(
                name="intfloat/e5-small-unsupervised",
                dimensions=384,
                tokens=128,
                memory_size=0.134,
                text_query_prefix="query: ",
                text_chunk_prefix="passage: ",
            ),
            "hf/multilingual-e5-small": HFModelProperties(
                name="intfloat/multilingual-e5-small",
                dimensions=384,
                tokens=512,
                memory_size=0.471,
                text_query_prefix="query: ",
                text_chunk_prefix="passage: ",
            ),
            "hf/multilingual-e5-base": HFModelProperties(
                name="intfloat/multilingual-e5-base",
                dimensions=768,
                tokens=512,
                memory_size=1.11,
                text_query_prefix="query: ",
                text_chunk_prefix="passage: ",
            ),
            "hf/multilingual-e5-large": HFModelProperties(
                name="intfloat/multilingual-e5-large",
                dimensions=1024,
                tokens=512,
                memory_size=2.24,
                text_query_prefix="query: ",
                text_chunk_prefix="passage: ",
            ),
            "hf/e5-small-v2": HFModelProperties(
                name="intfloat/e5-small-v2",
                dimensions=384,
                tokens=512,
                memory_size=0.134,
                text_query_prefix="query: ",
                text_chunk_prefix="passage: ",
            ),
            "hf/e5-base-v2": HFModelProperties(
                name="intfloat/e5-base-v2",
                dimensions=768,
                tokens=512,
                memory_size=0.438,
                text_query_prefix="query: ",
                text_chunk_prefix="passage: ",
            ),
            "hf/e5-large-v2": HFModelProperties(
                name="intfloat/e5-large-v2",
                dimensions=1024,
                tokens=512,
                memory_size=1.34,
                text_query_prefix="query: ",
                text_chunk_prefix="passage: ",
            ),
            "hf/multilingual-e5-large-instruct": HFModelProperties(
                name="intfloat/multilingual-e5-large-instruct",
                dimensions=1024,
                tokens=512,
                notes="",
                text_query_prefix="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
            ),
            "hf/GIST-large-Embedding-v0": HFModelProperties(
                name="avsolatorio/GIST-large-Embedding-v0",
                dimensions=1024,
                tokens=512,
            ),
            "hf/bge-small-en-v1.5": HFModelProperties(
                name="BAAI/bge-small-en-v1.5",
                dimensions=384,
                tokens=512,
                text_query_prefix="Represent this sentence for searching relevant passages: ",
            ),
            "hf/bge-base-en-v1.5": HFModelProperties(
                name="BAAI/bge-base-en-v1.5",
                dimensions=768,
                tokens=512,
                text_query_prefix="Represent this sentence for searching relevant passages: ",
            ),
            "hf/bge-large-en-v1.5": HFModelProperties(
                name="BAAI/bge-large-en-v1.5",
                dimensions=1024,
                tokens=512,
                text_query_prefix="Represent this sentence for searching relevant passages: ",
            ),
            "hf/bge-small-zh-v1.5": HFModelProperties(
                name="BAAI/bge-small-zh-v1.5",
                dimensions=512,
                tokens=512,
                text_query_prefix="为这个句子生成表示以用于检索相关文章：",
            ),
            "hf/bge-base-zh-v1.5": HFModelProperties(
                name="BAAI/bge-base-zh-v1.5",
                dimensions=768,
                tokens=512,
                text_query_prefix="为这个句子生成表示以用于检索相关文章：",
            ),
            "hf/bge-large-zh-v1.5": HFModelProperties(
                name="BAAI/bge-large-zh-v1.5",
                dimensions=1024,
                tokens=512,
                text_query_prefix="为这个句子生成表示以用于检索相关文章：",
            ),
            "hf/snowflake-arctic-embed-l": HFModelProperties(
                name="Snowflake/snowflake-arctic-embed-l",
                dimensions=1024,
                tokens=512,
                text_query_prefix="Represent this sentence for searching relevant passages: ",
            ),
            "hf/ember-v1": HFModelProperties(
                name="llmrails/ember-v1",
                dimensions=1024,
                tokens=512,
            ),
            "hf/snowflake-arctic-embed-m-v1.5": HFModelProperties(
                name="Snowflake/snowflake-arctic-embed-m-v1.5",
                dimensions=768,
                tokens=512,
                text_query_prefix="Represent this sentence for searching relevant passages: ",
            ),
            "hf/snowflake-arctic-embed-m": HFModelProperties(
                name="Snowflake/snowflake-arctic-embed-m",
                dimensions=768,
                tokens=512,
                text_query_prefix="Represent this sentence for searching relevant passages: ",
            ),
            "Marqo/dunzhang-stella_en_400M_v5": HFModelProperties(
                name="Marqo/dunzhang-stella_en_400M_v5",
                dimensions=1024,
                tokens=512,
                type=ModelType.hf_stella,
                trustRemoteCode=True,
            ),
        }
        pass


@convert_model_properties_to_dict
def _get_hf_properties() -> Dict:
    return HFModelProperties.get_all_model_properties_objects()
