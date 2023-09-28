import json
from unittest import TestCase

from marqo_commons.model_registry.clip_properties import _get_clip_properties
from marqo_commons.model_registry.fp16_clip_properties import _get_fp16_clip_properties
from marqo_commons.model_registry.hf_properties import _get_hf_properties
from marqo_commons.model_registry.model_registry import get_model_properties_dict, get_model_properties_json
from marqo_commons.model_registry.multilingual_clip_properties import _get_multilingual_clip_properties
from marqo_commons.model_registry.onnx_clip_properties import _get_onnx_clip_properties
from marqo_commons.model_registry.open_clip_properties import _get_open_clip_properties
from marqo_commons.model_registry.random_properties import _get_random_properties
from marqo_commons.model_registry.sbert_onnx_properties import _get_sbert_onnx_properties
from marqo_commons.model_registry.sbert_properties import _get_sbert_properties
from marqo_commons.model_registry.test_properties import _get_sbert_test_properties
from marqo_commons.shared_utils.enums import ModelType


class TestModelProperties(TestCase):
    def test_models_count(self):
        """ this test ensures that model names are not duplicated and thus the model registry is valid """
        model_properties = get_model_properties_dict()
        total_count_from_model_registry = len(model_properties)
        total_count_from_all_model_properties = 0

        sbert_model_properties = _get_sbert_properties()
        sbert_model_properties.update({k.split('/')[-1]: v for k, v in sbert_model_properties.items()})

        total_count_from_all_model_properties += len(sbert_model_properties)
        total_count_from_all_model_properties += len(_get_sbert_onnx_properties())
        total_count_from_all_model_properties += len(_get_clip_properties())
        total_count_from_all_model_properties += len(_get_sbert_test_properties())
        total_count_from_all_model_properties += len(_get_random_properties())
        total_count_from_all_model_properties += len(_get_hf_properties())
        total_count_from_all_model_properties += len(_get_open_clip_properties())
        total_count_from_all_model_properties += len(_get_onnx_clip_properties())
        total_count_from_all_model_properties += len(_get_multilingual_clip_properties())
        total_count_from_all_model_properties += len(_get_fp16_clip_properties())
        self.assertEqual(
            total_count_from_all_model_properties, total_count_from_model_registry,
            "Number of models in get_model_properties_dict is not equal to total "
            "concatenated number of models from all model_properties.py files")

    def test_all_models_have_text_modality(self):
        """ this test ensures that all models have text modality """
        model_properties = get_model_properties_dict()
        for model_name, model_property in model_properties.items():
            self.assertIn("text", model_property["modality"], f"Model {model_name} does not have text modality")

    def test_all_models_have_memory_size(self):
        """ this test ensures that all models have memory_size """
        model_properties = get_model_properties_dict()
        for model_name, model_property in model_properties.items():
            self.assertNotIn(model_property["memory_size"], [None, 0], f"Model {model_name} does not have memory_size")

    def test_all_model_types_are_used(self):
        """ this test ensures that all model types are used """
        model_properties = get_model_properties_dict()
        model_types = [model_property["type"] for model_property in model_properties.values()]
        for model_type in ModelType:
            self.assertIn(model_type, model_types, f"Model type {model_type} is not used")

    def test_serialized_and_deserialized_model_properties_are_equal(self):
        """ this test ensures that all models passed to json are correct and all keys are unique  """
        model_properties = get_model_properties_dict()
        model_properties_json = get_model_properties_json()
        deserialized_model_properties_json = json.loads(model_properties_json)
        for model_name, model_property in model_properties.items():
            # check if model name is in deserialized model properties
            self.assertIn(
                model_name, deserialized_model_properties_json,
                f"Model {model_name} is not in deserialized model properties"
            )
            for key, value in model_property.items():
                # check if key is in deserialized model properties
                self.assertIn(
                    key, deserialized_model_properties_json[model_name],
                    f"Model {model_name} does not have key {key} in deserialized model properties"
                )
                if type(deserialized_model_properties_json[model_name][key]) is list:
                    # check if all items are in deserialized model properties
                    for item in value:
                        self.assertIn(
                            item, deserialized_model_properties_json[model_name][key],
                            f"Model {model_name} has item {item} for key {key} but deserialized model properties does not have it"
                        )
                else:
                    # check if value is equal to deserialized model properties
                    self.assertEqual(
                        value, deserialized_model_properties_json[model_name][key],
                        f"Model {model_name} has value {value} for key {key} but deserialized model properties has value {deserialized_model_properties_json[model_name][key]}"
                    )