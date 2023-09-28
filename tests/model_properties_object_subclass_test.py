from typing import List

from marqo_commons.model_registry.model_properties_object import ModelProperties, VectorNumericType, Modality, ModelType
from unittest import TestCase
from marqo_commons.shared_utils.constants import MODEL_NAME_SIZE_MAPPING


class ModelPropertiesTestObject(ModelProperties):
    vector_numeric_type: VectorNumericType = VectorNumericType.float32
    modality: List[Modality] = [Modality.text]
    type: ModelType = ModelType.test
    tokens: int
    notes: str = ""
    default_random_field: str = "default_random_field"
    default_memory_size: float = 1.11


class TestModelPropertiesSubclass(TestCase):
    def setUp(self) -> None:
        self.basic_model_properties = ModelPropertiesTestObject(
            name="sentence-transformers/all-MiniLM-L6-v1",
            dimensions=16,
            tokens=128,
        )

    def test_default_values_are_not_returned_by_to_dict(self):
        dict_representation = self.basic_model_properties.to_dict()
        self.assertNotIn("default_random_field", dict_representation)
        self.assertNotIn("default_memory_size", dict_representation)

    def test_subclass_default_memory_size_overrides_parent(self):
        self.assertEqual(1.11, self.basic_model_properties.default_memory_size)

    def test_size_calculation_on_special_model_name(self):
        model_with_special_type = ModelPropertiesTestObject(
            name="ViT-L/14@336px",
            dimensions=768,
            tokens=128,
            type=ModelType.clip,
            modality=[Modality.image, Modality.text],
        )
        dict_representation = model_with_special_type.to_dict()
        self.assertEqual(MODEL_NAME_SIZE_MAPPING["vit-l-14"], dict_representation["memory_size"])

