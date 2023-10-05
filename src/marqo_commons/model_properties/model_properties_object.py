from typing import List

from pydantic import BaseModel, Field
from marqo_commons.shared_utils.enums import Modality, VectorNumericType, ModelType
from marqo_commons.shared_utils import constants


class ModelProperties(BaseModel):
    name: str = Field(..., title="Model name")
    dimensions: int = Field(..., title="Model dimensions")
    notes: str = Field(..., title="Model notes")
    type: ModelType = Field(..., title="Model types")
    default_memory_size: float = Field(0.66, title="Model default memory size in GB")
    memory_size: float = Field(..., title="Model memory size")
    modality: List[Modality] = Field(..., title="Model modality")
    vector_numeric_type: VectorNumericType = Field(..., title="Model vector numeric type")

    def __init__(self, **kwargs):
        if "memory_size" not in kwargs:
            kwargs["memory_size"] = self._get_default_model_size(kwargs["name"])
        super().__init__(**kwargs)

    @classmethod
    def _get_default_model_size(cls, name) -> float:
        """
        Calculates default model memory size in the following order:
        1. Use default from `constants.MODEL_NAME_SIZE_MAPPING` if a name there is in the model name.
        2. Use default from `default_memory_size` field in subclass (eg `ClipProperties`) if exists.
        3. Use default from `default_memory_size` field in `ModelProperties`.
        """
        name_info = name.lower().replace("/", "-")
        for name, size in constants.MODEL_NAME_SIZE_MAPPING.items():
            if name in name_info:
                return size
        return cls.__fields__["default_memory_size"].default

    def to_dict(self):
        """ Function returns a dict of the model properties without the default values.
        Handles deletion of default values by collecting the keys of the dict that start with "default".
        And then deleting them from the dict.
        Implemented this way to avoid dict size changes during iteration.
        """
        dict_of_model_properties = vars(self)
        properties_to_remove = []  # we need to remove the default values from the dict
        for key in dict_of_model_properties.keys():
            if key.startswith("default"):
                properties_to_remove.append(key)
        for key in properties_to_remove:
            del dict_of_model_properties[key]
        return dict_of_model_properties

