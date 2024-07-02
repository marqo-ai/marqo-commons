"""
Random model properties

This file contains random model properties. It is intended to be used in conjunction with the model registry
and should not be used in isolation.
"""
from typing import Dict, List

from marqo_commons.model_registry.model_properties_object import ModelProperties, VectorNumericType, Modality, ModelType
from marqo_commons.model_registry.utils import convert_model_properties_to_dict


class RandomModelProperties(ModelProperties):
    vector_numeric_type: VectorNumericType = VectorNumericType.float32
    default_memory_size: float = 0.1
    modality: List[Modality] = [Modality.text, Modality.image]
    type: ModelType = ModelType.random
    tokens: int
    notes: str = ""


@convert_model_properties_to_dict
def _get_random_properties() -> Dict:
    RANDOM_MODEL_PROPERTIES = {
        "random": RandomModelProperties(
            name="random",
            dimensions=384,
            tokens=128,
            notes=""
        ),
        "random/large": RandomModelProperties(
            name="random/large",
            dimensions=768,
            tokens=128,
            notes=""
        ),
        "random/small": RandomModelProperties(
            name="random/small",
            dimensions=32,
            tokens=128,
            notes=""
        ),
        "random/medium": RandomModelProperties(
            name="random/medium",
            dimensions=128,
            tokens=128,
            notes=""
        ),
    }
    return RANDOM_MODEL_PROPERTIES
