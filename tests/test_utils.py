from marqo_commons.model_registry.utils import convert_model_properties_to_dict
from unittest import TestCase
from marqo_commons.model_registry.model_properties_object import ModelProperties, VectorNumericType, Modality, ModelType


class TestUtils(TestCase):
    def test_convert_model_properties_to_dict(self):
        @convert_model_properties_to_dict
        def _get_test_properties():
            return {
                "sentence-transformers/test": ModelProperties(
                    name="sentence-transformers/all-MiniLM-L6-v1",
                    dimensions=16,
                    tokens=128,
                    vector_numeric_type=VectorNumericType.float32,
                    modality=[Modality.text],
                    type=ModelType.test,
                    notes="",
                ),
                "test": ModelProperties(
                    name="sentence-transformers/all-MiniLM-L6-v1",
                    dimensions=16,
                    tokens=128,
                    vector_numeric_type=VectorNumericType.float32,
                    modality=[Modality.text],
                    type=ModelType.test,
                    notes="",
                ),
            }

        properties = _get_test_properties()
        self.assertEqual(type(properties["sentence-transformers/test"]), dict)
        self.assertEqual(type(properties["test"]), dict)
        self.assertNotIn("default_memory_size", properties)
        self.assertEqual(properties["sentence-transformers/test"]["name"], "sentence-transformers/all-MiniLM-L6-v1")