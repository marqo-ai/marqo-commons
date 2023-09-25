import jsonschema

from marqo_commons.shared_utils.enums import IndexSettingsField as NsFields
from marqo_commons.shared_utils.errors import InvalidArgError


def validate_index_settings(settings_to_validate: dict, MAX_NUMBER_OF_REPLICAS: int, MAX_EF_CONSTRUCTION_VALUE: int):
    """validates index settings.
    Returns
        The given index settings if validation has passed

    Raises an InvalidArgError if the settings object is badly formatted
    """
    try:
        settings_schema = {
            "$schema": "https://json-schema.org/draft/2019-09/schema",
            "type": "object",
            "required": [
                NsFields.index_defaults,
                NsFields.number_of_shards,
                NsFields.number_of_replicas
            ],
            "additionalProperties": False,
            "properties": {
                NsFields.index_defaults: {
                    "type": "object",
                    "required": [
                        NsFields.treat_urls_and_pointers_as_images,
                        NsFields.model,
                        NsFields.normalize_embeddings,
                        NsFields.text_preprocessing,
                        NsFields.image_preprocessing
                    ],
                    "additionalProperties": False,
                    "properties": {
                        NsFields.treat_urls_and_pointers_as_images: {
                            "type": "boolean",
                            "examples": [
                                False
                            ]
                        },
                        NsFields.model: {
                            "type": "string",
                            "examples": [
                                "hf/all_datasets_v4_MiniLM-L6"
                            ]
                        },
                        NsFields.model_properties: {
                            "type": "object",
                        },
                        NsFields.normalize_embeddings: {
                            "type": "boolean",
                            "examples": [
                                True
                            ]
                        },
                        NsFields.text_preprocessing: {
                            "type": "object",
                            "required": [
                                NsFields.split_length,
                                NsFields.split_overlap,
                                NsFields.split_method
                            ],
                            "properties": {
                                NsFields.split_length: {
                                    "type": "integer",
                                    "examples": [
                                        2
                                    ]
                                },
                                NsFields.split_overlap: {
                                    "type": "integer",
                                    "examples": [
                                        0
                                    ]
                                },
                                NsFields.split_method: {
                                    "type": "string",
                                    "examples": [
                                        "sentence"
                                    ]
                                }
                            },
                            "examples": [{
                                NsFields.split_length: 2,
                                NsFields.split_overlap: 0,
                                NsFields.split_method: "sentence"
                            }]
                        },
                        NsFields.image_preprocessing: {
                            "type": "object",
                            "required": [
                                NsFields.patch_method
                            ],
                            "properties": {
                                NsFields.patch_method: {
                                    "type": ["null", "string"],
                                    "examples": [
                                        None
                                    ]
                                }
                            },
                            "examples": [{
                                NsFields.patch_method: None
                            }]
                        },
                        NsFields.ann_parameters: {
                            "type": "object",
                            "required": [
                                # Non required for backwards compatibility
                            ],
                            "properties": {
                                NsFields.ann_method: {
                                    "type": "string",
                                    "enum": ["hnsw"],
                                    "examples": [
                                        "hnsw"
                                    ]
                                },
                                NsFields.ann_engine: {
                                    "type": "string",
                                    "enum": ["lucene"],
                                    "examples": [
                                        "lucene"
                                    ]
                                },
                                NsFields.ann_metric: {
                                    "type": "string",
                                    "enum": ["l1", "l2", "linf", "cosinesimil"],
                                    "examples": [
                                        "cosinesimil"
                                    ]
                                },
                                NsFields.ann_method_parameters: {
                                    "type": "object",
                                    "required": [],
                                    "properties": {
                                        NsFields.hnsw_ef_construction: {
                                            "type": "integer",
                                            "minimum": 1,
                                            "maximum": MAX_EF_CONSTRUCTION_VALUE,
                                            "examples": [
                                                128
                                            ]
                                        },
                                        NsFields.hnsw_m: {
                                            "type": "integer",
                                            "minimum": 2,
                                            "maximum": 100,
                                            "examples": [
                                                16
                                            ]
                                        },
                                    },
                                    "examples": [{
                                        NsFields.hnsw_ef_construction: 128,
                                        NsFields.hnsw_m: 16
                                    }]
                                }
                            },
                            "examples": [{
                                NsFields.ann_method: "hnsw",
                                NsFields.ann_engine: "lucene",
                                NsFields.ann_metric: "cosinesimil",
                                NsFields.ann_method_parameters: {
                                    NsFields.hnsw_ef_construction: 128,
                                    NsFields.hnsw_m: 16
                                }
                            }]
                        }
                    },
                    "examples": [{
                        NsFields.treat_urls_and_pointers_as_images: False,
                        NsFields.model: "hf/all_datasets_v4_MiniLM-L6",
                        NsFields.normalize_embeddings: True,
                        NsFields.text_preprocessing: {
                            NsFields.split_length: 2,
                            NsFields.split_overlap: 0,
                            NsFields.split_method: "sentence"
                        },
                        NsFields.image_preprocessing: {
                            NsFields.patch_method: None
                        },
                        NsFields.ann_parameters: {
                            NsFields.ann_method: "hnsw",
                            NsFields.ann_engine: "lucene",
                            NsFields.ann_metric: "cosinesimil",
                            NsFields.ann_method_parameters: {
                                NsFields.hnsw_ef_construction: 128,
                                NsFields.hnsw_m: 16
                            }
                        }
                    }]
                },
                NsFields.number_of_shards: {
                    "type": "integer",
                    "minimum": 1,
                    "examples": [
                        5
                    ]
                },
                NsFields.number_of_replicas: {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": MAX_NUMBER_OF_REPLICAS,
                    "examples": [
                        1
                    ]
                },
            },
            "examples": [{
                NsFields.index_defaults: {
                    NsFields.treat_urls_and_pointers_as_images: False,
                    NsFields.model: "hf/all_datasets_v4_MiniLM-L6",
                    NsFields.normalize_embeddings: True,
                    NsFields.text_preprocessing: {
                        NsFields.split_length: 2,
                        NsFields.split_overlap: 0,
                        NsFields.split_method: "sentence"
                    },
                    NsFields.image_preprocessing: {
                        NsFields.patch_method: None
                    },
                    NsFields.ann_parameters: {
                        NsFields.ann_method: "hnsw",
                        NsFields.ann_engine: "lucene",
                        NsFields.ann_metric: "cosinesimil",
                        NsFields.ann_method_parameters: {
                            NsFields.hnsw_ef_construction: 128,
                            NsFields.hnsw_m: 16
                        }
                    }
                },
                NsFields.number_of_shards: 3,
                NsFields.number_of_replicas: 0
            }]
        }
        jsonschema.validate(instance=dict, schema=settings_schema)
        return settings_to_validate
    except jsonschema.ValidationError as e:
        raise InvalidArgError(
            f"Error validating index settings object. Reason: \n{str(e)}"
            f"\nRead about the index settings object here: https://docs.marqo.ai/0.0.13/API-Reference/indexes/#body"
        )
