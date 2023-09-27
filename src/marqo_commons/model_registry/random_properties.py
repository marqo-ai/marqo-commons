from typing import Dict, List

from marqo_commons.model_registry.model_properties_object import ModelProperties, VectorNumericType, Modality, ModelType


class RandomModelProperties(ModelProperties):
    vector_numeric_type: VectorNumericType = VectorNumericType.float32
    default_memory_size: float = 0.1
    modality: List[Modality] = [Modality.text, Modality.image]
    type: ModelType = ModelType.random
    tokens: int
    notes: str = ""

def _get_random_properties() -> Dict:
    RANDOM_MODEL_PROPERTIES = {
        "random": RandomModelProperties(
            name="random",
            dimensions=384,
            tokens=128,
        ).to_dict(),
        "random/large": RandomModelProperties(
            name="random/large",
            dimensions=768,
            tokens=128,
        ).to_dict(),
        "random/small": RandomModelProperties(
            name="random/small",
            dimensions=32,
            tokens=128,
        ).to_dict(),
        "random/medium": RandomModelProperties(
            name="random/medium",
            dimensions=128,
            tokens=128,
        ).to_dict(),
    }

    return RANDOM_MODEL_PROPERTIES
