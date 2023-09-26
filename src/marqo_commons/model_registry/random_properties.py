from typing import Dict

from marqo_commons.model_registry.model_properties_object import ModelProperties, VectorNumericType, Modality


class RandomModelProperties(ModelProperties):
    vector_numeric_type: VectorNumericType = VectorNumericType.float32
    modality: list[Modality] = [Modality.text, Modality.image]
    type: str = "random"
    memory_size: float = 0  # TODO: add memory size
    tokens: int
    notes: str = ""

def _get_random_properties() -> Dict:
    RANDOM_MODEL_PROPERTIES = {
        "random": RandomModelProperties(
            name="random",
            dimensions=384,
            tokens=128,
        ).__dict__,
        "random/large": RandomModelProperties(
            name="random/large",
            dimensions=768,
            tokens=128,
        ).__dict__,
        "random/small": RandomModelProperties(
            name="random/small",
            dimensions=32,
            tokens=128,
        ).__dict__,
        "random/medium": RandomModelProperties(
            name="random/medium",
            dimensions=128,
            tokens=128,
        ).__dict__,
    }

    return RANDOM_MODEL_PROPERTIES


_get_random_properties()