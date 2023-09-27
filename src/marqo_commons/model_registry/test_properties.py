from typing import Dict, List

from marqo_commons.model_registry.model_properties_object import ModelProperties, VectorNumericType, Modality, ModelType


class SbertTestModelProperties(ModelProperties):
    vector_numeric_type: VectorNumericType = VectorNumericType.float32
    modality: List[Modality] = [Modality.text]
    type: ModelType = ModelType.test
    tokens: int
    notes: str = ""


def _get_sbert_test_properties() -> Dict:
    TEST_MODEL_PROPERTIES = {
        "sentence-transformers/test": SbertTestModelProperties(
            name="sentence-transformers/all-MiniLM-L6-v1",
            dimensions=16,
            tokens=128,
        ).to_dict(),
        "test": SbertTestModelProperties(
            name="sentence-transformers/all-MiniLM-L6-v1",
            dimensions=16,
            tokens=128,
        ).to_dict(),
    }
    return TEST_MODEL_PROPERTIES
