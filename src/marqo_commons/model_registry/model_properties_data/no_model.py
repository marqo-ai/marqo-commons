"""
No-model properties

This file contains random model properties. It is intended to be used in conjunction with the model registry
and should not be used in isolation.
"""
from typing import Dict, List

from marqo_commons.model_registry.model_properties_object import ModelProperties, VectorNumericType, Modality, \
    ModelType, T
from marqo_commons.model_registry.utils import convert_model_properties_to_dict


class NoModelProperties(ModelProperties):
    vector_numeric_type: VectorNumericType = VectorNumericType.float32
    default_memory_size: float = 0.1
    modality: List[Modality] = [Modality.text, Modality.image]
    type: ModelType = ModelType.no_model
    tokens: int
    notes: str = ""

    @classmethod
    def get_all_model_properties_objects(cls) -> Dict[str, T]:
        return {
            'no_model': NoModelProperties(
                name='no_model',
                dimensions=0,
                notes="This is a special model no_model that requires users to provide 'dimensions'",
                type=ModelType.no_model,
                tokens=0  # Assuming default value for tokens
            )
        }

@convert_model_properties_to_dict
def _get_no_model_properties() -> Dict:
    return NoModelProperties.get_all_model_properties_objects()
