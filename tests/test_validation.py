import os
from unittest import TestCase

from marqo_commons.settings_validation.settings_validation import validate_index_settings
from marqo_commons.shared_utils.errors import InvalidSettingsArgError


class TestValidateIndexSettings(TestCase):
    def setUp(self) -> None:
        self.MARQO_MAX_NUMBER_OF_REPLICAS = 1
        self.MARQO_EF_CONSTRUCTION_MAX_VALUE = 4096

    @staticmethod
    def get_good_index_settings():
        return {
            "index_defaults": {
                "treat_urls_and_pointers_as_images": False,
                "model": "hf/all_datasets_v4_MiniLM-L6",
                "normalize_embeddings": True,
                "text_preprocessing": {
                    "split_length": 2,
                    "split_overlap": 0,
                    "split_method": "sentence"
                },
                "image_preprocessing": {
                    "patch_method": None
                }
            },
            "number_of_shards": 5,
            "number_of_replicas": 1
        }

    def test_validate_index_settings(self):

        good_settings = [
            {
                "index_defaults": {
                    "treat_urls_and_pointers_as_images": False,
                    "model": "hf/all_datasets_v4_MiniLM-L6",
                    "normalize_embeddings": True,
                    "text_preprocessing": {
                        "split_length": 2,
                        "split_overlap": 0,
                        "split_method": "sentence"
                    },
                    "image_preprocessing": {
                        "patch_method": None
                    }
                },
                "number_of_shards": 5,
                "number_of_replicas": 1
            },
            {  # extra field in text_preprocessing: OK
                "index_defaults": {
                    "treat_urls_and_pointers_as_images": False,
                    "model": "hf/all_datasets_v4_MiniLM-L6",
                    "normalize_embeddings": True,
                    "text_preprocessing": {
                        "split_length": 2,
                        "split_overlap": 0,
                        "split_method": "sentence",
                        "blah blah blah": "woohoo"
                    },
                    "image_preprocessing": {
                        "patch_method": None
                    }
                },
                "number_of_shards": 5,
                "number_of_replicas": 1
            },
            {  # extra field in image_preprocessing: OK
                "index_defaults": {
                    "treat_urls_and_pointers_as_images": False,
                    "model": "hf/all_datasets_v4_MiniLM-L6",
                    "normalize_embeddings": True,
                    "text_preprocessing": {
                        "split_length": 2,
                        "split_overlap": 0,
                        "split_method": "sentence",
                    },
                    "image_preprocessing": {
                        "patch_method": None,
                        "blah blah blah": "woohoo"
                    }
                },
                "number_of_shards": 5,
                "number_of_replicas": 1
            }
        ]
        for settings in good_settings:
            assert settings == validate_index_settings(settings,
                                                       MAX_EF_CONSTRUCTION_VALUE=
                                                       self.MARQO_EF_CONSTRUCTION_MAX_VALUE,
                                                       MAX_NUMBER_OF_REPLICAS=
                                                       self.MARQO_MAX_NUMBER_OF_REPLICAS,
                                                       )

    def test_validate_index_settings_model_properties(self):
        good_settings = self.get_good_index_settings()
        good_settings['index_defaults']['model_properties'] = dict()
        assert good_settings == validate_index_settings(good_settings,
                                                        MAX_EF_CONSTRUCTION_VALUE=
                                                        self.MARQO_EF_CONSTRUCTION_MAX_VALUE,
                                                        MAX_NUMBER_OF_REPLICAS=
                                                        self.MARQO_MAX_NUMBER_OF_REPLICAS,
                                                        )
        
    def test_validate_index_settings_search_model(self):
        good_settings = self.get_good_index_settings()
        good_settings['index_defaults']['model_properties'] = {
            "dimensions": 123,
            "url": "https://random_site.com"
        }
        good_settings['index_defaults']['search_model'] = "ViT-B/32"
        good_settings['index_defaults']['search_model_properties'] = {
            "dimensions": 456,
            "url": "https://random_site.com"
        }
        assert good_settings == validate_index_settings(good_settings,
                                                        MAX_EF_CONSTRUCTION_VALUE=
                                                        self.MARQO_EF_CONSTRUCTION_MAX_VALUE,
                                                        MAX_NUMBER_OF_REPLICAS=
                                                        self.MARQO_MAX_NUMBER_OF_REPLICAS,
                                                        )

    def test_validate_index_settings_bad(self):
        bad_settings = [{
            "index_defaults": {
                "treat_urls_and_pointers_as_images": False,
                "model": "hf/all_datasets_v4_MiniLM-L6",
                "normalize_embeddings": True,
                "text_preprocessing": {
                    "split_length": "2",
                    "split_overlap": "0",
                    "split_method": "sentence"
                },
                "image_preprocessing": {
                    "patch_method": None
                }
            },
            "number_of_shards": 5,
            "number_of_replicas": -1
        },
            {
                "index_defaults": {
                    "treat_urls_and_pointers_as_images": False,
                    "model": "hf/all_datasets_v4_MiniLM-L6",
                    "normalize_embeddings": True,
                    "text_preprocessing": {
                        "split_length": "2",
                        "split_overlap": "0",
                        "split_method": "sentence"
                    },
                    "image_preprocessing": {
                        "patch_method": None
                    }
                },
                "number_of_shards": 5
            },
        ]
        for bad_setting in bad_settings:
            try:
                validate_index_settings(bad_setting,
                                        MAX_EF_CONSTRUCTION_VALUE=
                                        self.MARQO_EF_CONSTRUCTION_MAX_VALUE,
                                        MAX_NUMBER_OF_REPLICAS=
                                        self.MARQO_MAX_NUMBER_OF_REPLICAS,
                                        )
                raise AssertionError
            except InvalidSettingsArgError as e:
                pass

    def test_validate_index_settings_missing_text_preprocessing(self):
        settings = self.get_good_index_settings()
        # base good settings should be OK
        assert settings == validate_index_settings(settings,
                                                   MAX_EF_CONSTRUCTION_VALUE=
                                                   self.MARQO_EF_CONSTRUCTION_MAX_VALUE,
                                                   MAX_NUMBER_OF_REPLICAS=
                                                   self.MARQO_MAX_NUMBER_OF_REPLICAS,
                                                   )
        del settings['index_defaults']['text_preprocessing']
        try:
            validate_index_settings(settings,
                                    MAX_EF_CONSTRUCTION_VALUE=self.MARQO_EF_CONSTRUCTION_MAX_VALUE,
                                    MAX_NUMBER_OF_REPLICAS=self.MARQO_MAX_NUMBER_OF_REPLICAS,
                                    )
            raise AssertionError
        except InvalidSettingsArgError:
            pass

    def test_validate_index_settings_missing_model(self):
        settings = self.get_good_index_settings()
        # base good settings should be OK
        assert settings == validate_index_settings(settings,
                                                   MAX_EF_CONSTRUCTION_VALUE=
                                                   self.MARQO_EF_CONSTRUCTION_MAX_VALUE,
                                                   MAX_NUMBER_OF_REPLICAS=
                                                   self.MARQO_MAX_NUMBER_OF_REPLICAS,
                                                   )
        del settings['index_defaults']['model']
        try:
            validate_index_settings(settings,
                                    MAX_EF_CONSTRUCTION_VALUE=self.MARQO_EF_CONSTRUCTION_MAX_VALUE,
                                    MAX_NUMBER_OF_REPLICAS=self.MARQO_MAX_NUMBER_OF_REPLICAS,
                                    )
            raise AssertionError
        except InvalidSettingsArgError:
            pass

    def test_validate_index_settings_missing_index_defaults(self):
        settings = self.get_good_index_settings()
        # base good settings should be OK
        assert settings == validate_index_settings(settings,
                                                   MAX_EF_CONSTRUCTION_VALUE=
                                                   self.MARQO_EF_CONSTRUCTION_MAX_VALUE,
                                                   MAX_NUMBER_OF_REPLICAS=
                                                   self.MARQO_MAX_NUMBER_OF_REPLICAS,
                                                   )
        del settings['index_defaults']
        try:
            validate_index_settings(settings,
                                    MAX_EF_CONSTRUCTION_VALUE=self.MARQO_EF_CONSTRUCTION_MAX_VALUE,
                                    MAX_NUMBER_OF_REPLICAS=self.MARQO_MAX_NUMBER_OF_REPLICAS,
                                    )
            raise AssertionError
        except InvalidSettingsArgError:
            pass

    def test_validate_index_settings_bad_number_shards(self):
        settings = self.get_good_index_settings()
        # base good settings should be OK
        assert settings == validate_index_settings(settings,
                                                   MAX_EF_CONSTRUCTION_VALUE=
                                                   self.MARQO_EF_CONSTRUCTION_MAX_VALUE,
                                                   MAX_NUMBER_OF_REPLICAS=
                                                   self.MARQO_MAX_NUMBER_OF_REPLICAS,
                                                   )
        settings['number_of_shards'] = -1
        try:
            validate_index_settings(settings,
                                    MAX_EF_CONSTRUCTION_VALUE=self.MARQO_EF_CONSTRUCTION_MAX_VALUE,
                                    MAX_NUMBER_OF_REPLICAS=self.MARQO_MAX_NUMBER_OF_REPLICAS,
                                    )
            raise AssertionError
        except InvalidSettingsArgError as e:
            pass

    def test_validate_index_settings_bad_number_replicas(self):
        settings = self.get_good_index_settings()
        # base good settings should be OK
        assert settings == validate_index_settings(settings,
                                                   MAX_EF_CONSTRUCTION_VALUE=
                                                   self.MARQO_EF_CONSTRUCTION_MAX_VALUE,
                                                   MAX_NUMBER_OF_REPLICAS=
                                                   self.MARQO_MAX_NUMBER_OF_REPLICAS,
                                                   )
        settings['number_of_replicas'] = -1
        try:
            validate_index_settings(settings,
                                    MAX_EF_CONSTRUCTION_VALUE=self.MARQO_EF_CONSTRUCTION_MAX_VALUE,
                                    MAX_NUMBER_OF_REPLICAS=self.MARQO_MAX_NUMBER_OF_REPLICAS,
                                    )
            raise AssertionError
        except InvalidSettingsArgError as e:
            pass

    def test_validate_index_settings_img_preprocessing(self):
        settings = self.get_good_index_settings()
        # base good settings should be OK
        assert settings == validate_index_settings(settings,
                                                   MAX_EF_CONSTRUCTION_VALUE=
                                                   self.MARQO_EF_CONSTRUCTION_MAX_VALUE,
                                                   MAX_NUMBER_OF_REPLICAS=
                                                   self.MARQO_MAX_NUMBER_OF_REPLICAS,
                                                   )
        settings['index_defaults']['image_preprocessing']["path_method"] = "frcnn"
        assert settings == validate_index_settings(settings,
                                                   MAX_EF_CONSTRUCTION_VALUE=
                                                   self.MARQO_EF_CONSTRUCTION_MAX_VALUE,
                                                   MAX_NUMBER_OF_REPLICAS=
                                                   self.MARQO_MAX_NUMBER_OF_REPLICAS,
                                                   )

    def test_validate_index_settings_misplaced_fields(self):
        bad_settings = [
            {
                "index_defaults": {
                    "treat_urls_and_pointers_as_images": False,
                    "model": "hf/all_datasets_v4_MiniLM-L6",
                    "normalize_embeddings": True,
                    "text_preprocessing": {
                        "split_length": 2,
                        "split_overlap": 0,
                        "split_method": "sentence"
                    },
                    "image_preprocessing": {
                        "patch_method": None
                    }
                },
                "number_of_shards": 5,
                "model": "hf/all_datasets_v4_MiniLM-L6"  # model is also outside, here...
            },
            {
                "index_defaults": {
                    "image_preprocessing": {
                        "patch_method": None  # no models here
                    },
                    "normalize_embeddings": True,
                    "text_preprocessing": {
                        "split_length": 2,
                        "split_method": "sentence",
                        "split_overlap": 0
                    },
                    "treat_urls_and_pointers_as_images": False
                },
                "model": "open_clip/ViT-L-14/laion2b_s32b_b82k",  # model here (bad)
                "number_of_shards": 5,
                "treat_urls_and_pointers_as_images": True
            },
            {
                "index_defaults": {
                    "image_preprocessing": {
                        "patch_method": None,
                        "model": "open_clip/ViT-L-14/laion2b_s32b_b82k",
                    },
                    "normalize_embeddings": True,
                    "text_preprocessing": {
                        "split_length": 2,
                        "split_method": "sentence",
                        "split_overlap": 0
                    },
                    "treat_urls_and_pointers_as_images": False,
                    "number_of_shards": 5,  # shouldn't be here
                },
                "treat_urls_and_pointers_as_images": True
            },
            {  # good, BUT extra field in index_defaults
                "index_defaults": {
                    "number_of_shards": 5,
                    "treat_urls_and_pointers_as_images": False,
                    "model": "hf/all_datasets_v4_MiniLM-L6",
                    "normalize_embeddings": True,
                    "text_preprocessing": {
                        "split_length": 2,
                        "split_overlap": 0,
                        "split_method": "sentence"
                    },
                    "image_preprocessing": {
                        "patch_method": None
                    }
                },
                "number_of_shards": 5
            },
            {  # good, BUT extra field in root
                "model": "hf/all_datasets_v4_MiniLM-L6",
                "index_defaults": {
                    "treat_urls_and_pointers_as_images": False,
                    "model": "hf/all_datasets_v4_MiniLM-L6",
                    "normalize_embeddings": True,
                    "text_preprocessing": {
                        "split_length": 2,
                        "split_overlap": 0,
                        "split_method": "sentence"
                    },
                    "image_preprocessing": {
                        "patch_method": None
                    }
                },
                "number_of_shards": 5
            }
        ]
        for bad_set in bad_settings:
            try:
                validate_index_settings(bad_set,
                                        MAX_EF_CONSTRUCTION_VALUE=
                                        self.MARQO_EF_CONSTRUCTION_MAX_VALUE,
                                        MAX_NUMBER_OF_REPLICAS=
                                        self.MARQO_MAX_NUMBER_OF_REPLICAS,
                                        )
                raise AssertionError
            except InvalidSettingsArgError as e:
                pass

    def test_validate_index_settings_replicas_exceed_limit(self):
        settings = self.get_good_index_settings()
        settings['number_of_replicas'] = self.MARQO_MAX_NUMBER_OF_REPLICAS + 1
        try:
            validate_index_settings(settings,
                                    MAX_EF_CONSTRUCTION_VALUE=self.MARQO_EF_CONSTRUCTION_MAX_VALUE,
                                    MAX_NUMBER_OF_REPLICAS=self.MARQO_MAX_NUMBER_OF_REPLICAS,
                                    )
            raise AssertionError
        except InvalidSettingsArgError as e:
            pass
