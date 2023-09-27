from typing import Dict

from marqo_commons.model_registry.model_properties_object import ModelProperties, VectorNumericType, Modality, ModelType


class RandomModelProperties(ModelProperties):
    vector_numeric_type: VectorNumericType = VectorNumericType.float32
    modality: list[Modality] = [Modality.text, Modality.image]
    type: ModelType = ModelType.random
    tokens: int
    notes: str = ""

def _get_random_properties() -> Dict:
    RANDOM_MODEL_PROPERTIES = {
        "random": vars(RandomModelProperties(
            name="random",
            memory_size=0.1,
            dimensions=384,
            tokens=128,
        )),
        "random/large": vars(RandomModelProperties(
            name="random/large",
            memory_size=0.1,
            dimensions=768,
            tokens=128,
        )),
        "random/small": vars(RandomModelProperties(
            name="random/small",
            memory_size=0.1,
            dimensions=32,
            tokens=128,
        )),
        "random/medium": vars(RandomModelProperties(
            name="random/medium",
            memory_size=0.1,
            dimensions=128,
            tokens=128,
        )),
    }

    return RANDOM_MODEL_PROPERTIES
