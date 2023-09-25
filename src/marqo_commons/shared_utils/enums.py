from enum import Enum


class Modality(str, Enum):
    image = "image"
    text = "text"


class VectorNumericType(str, Enum):
    float32 = "float32"
