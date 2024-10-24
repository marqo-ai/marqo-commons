"""
Test model properties

This file contains test model properties. It is intended to be used in conjunction with the model registry
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


class SbertTestModelProperties(ModelProperties):
    vector_numeric_type: VectorNumericType = VectorNumericType.float32
    modality: List[Modality] = [Modality.text]
    type: ModelType = ModelType.test
    tokens: int
    notes: str = ""
    text_query_prefix: str = ""
    text_chunk_prefix: str = ""

    @classmethod
    def get_all_model_properties_objects(cls) -> Dict[str, "SbertTestModelProperties"]:
        return {
            "sentence-transformers/test": SbertTestModelProperties(
                name="sentence-transformers/all-MiniLM-L6-v1",
                dimensions=16,
                tokens=128,
            ),
            "test": SbertTestModelProperties(
                name="sentence-transformers/all-MiniLM-L6-v1",
                dimensions=16,
                tokens=128,
            ),
            "test_prefix": SbertTestModelProperties(
                name="sentence-transformers/all-MiniLM-L6-v1",
                dimensions=16,
                tokens=128,
                type=ModelType.test,
                text_query_prefix="test query: ",
                text_chunk_prefix="test passage: ",
            ),
        }


@convert_model_properties_to_dict
def _get_sbert_test_properties() -> Dict:
    return SbertTestModelProperties.get_all_model_properties_objects()
