from typing import Dict

from marqo_commons.model_registry.model_properties_object import ModelProperties, VectorNumericType, Modality


class SbertTestModelProperties(ModelProperties):
    vector_numeric_type: VectorNumericType = VectorNumericType.float32
    modality: list[Modality] = [Modality.text, Modality.image]
    type: str = "random"
    memory_size: int = 0  # TODO: add memory size
    tokens: int
    notes: str = ""


def _get_sbert_test_properties() -> Dict:
    TEST_MODEL_PROPERTIES = {
        "sentence-transformers/test": SbertTestModelProperties(
            name="sentence-transformers/all-MiniLM-L6-v1",
            dimensions=16,
            tokens=128,
        ).__dict__,
        "test": SbertTestModelProperties(
            name="sentence-transformers/all-MiniLM-L6-v1",
            dimensions=16,
            tokens=128,
        ).__dict__,
    }
    return TEST_MODEL_PROPERTIES
