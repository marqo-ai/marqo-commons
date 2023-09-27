from typing import Dict

from marqo_commons.model_registry.model_properties_object import ModelProperties, VectorNumericType, Modality, ModelType


class SbertTestModelProperties(ModelProperties):
    vector_numeric_type: VectorNumericType = VectorNumericType.float32
    modality: list[Modality] = [Modality.text]
    type: ModelType = ModelType.test
    tokens: int
    notes: str = ""


def _get_sbert_test_properties() -> Dict:
    TEST_MODEL_PROPERTIES = {
        "sentence-transformers/test": vars(SbertTestModelProperties(
            name="sentence-transformers/all-MiniLM-L6-v1",
            memory_size=0.66,
            dimensions=16,
            tokens=128,
        )),
        "test": vars(SbertTestModelProperties(
            name="sentence-transformers/all-MiniLM-L6-v1",
            memory_size=0.66,
            dimensions=16,
            tokens=128,
        )),
    }
    return TEST_MODEL_PROPERTIES
