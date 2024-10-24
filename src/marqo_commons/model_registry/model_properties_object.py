from abc import ABC, abstractmethod
from typing import Dict, List, TypeVar

from pydantic import BaseModel, Field

from marqo_commons.shared_utils import constants
from marqo_commons.shared_utils.enums import Modality, ModelType, VectorNumericType

T = TypeVar("T", bound="ModelProperties")


class ModelProperties(BaseModel, ABC):
    name: str = Field(..., title="Model name")
    dimensions: int = Field(..., title="Model dimensions")
    notes: str = Field(..., title="Model notes")
    type: ModelType = Field(..., title="Model types")

    # Model memory size is in gigabytes (GB).
    default_memory_size: float = 0.66
    memory_size: float = Field(..., title="Model memory size")

    modality: List[Modality] = Field(..., title="Model modality")
    vector_numeric_type: VectorNumericType = Field(
        ..., title="Model vector numeric type"
    )
    trustRemoteCode: bool = Field(default=False, title="Trust remote code")

    def __init__(self, **kwargs):
        if "memory_size" not in kwargs:
            kwargs["memory_size"] = self._get_default_model_size(kwargs["name"])
        super().__init__(**kwargs)

    @classmethod
    @abstractmethod
    def get_all_model_properties_objects(cls) -> Dict[str, T]:
        """
        Get all model properties associated with the model type class. This returns a dictionary
        of typed objects that are subclasses of ModelProperties keyed by their model names.
        """
        pass

    @classmethod
    def _get_default_model_size(cls, name) -> float:
        """
        Calculates default model memory size in the following order:
        1. Use default from `constants.MODEL_NAME_SIZE_MAPPING` if a name there is in the model name.
        2. Use default from `default_memory_size` field in subclass (eg `ClipProperties`) if exists.
        3. Use default from `default_memory_size` field in `ModelProperties`.

        Model memory size is in gigabytes (GB).
        """
        name_info = name.lower().replace("/", "-")
        for name, size in constants.MODEL_NAME_SIZE_MAPPING.items():
            if name in name_info:
                return size
        return cls.model_fields["default_memory_size"].default

    def to_dict(self):
        """Function returns a dict of the model properties without the default values.
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
