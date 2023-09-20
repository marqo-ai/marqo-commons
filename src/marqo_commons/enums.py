from enum import Enum


class Modality(str, Enum):
    IMAGE = "image"
    TEXT = "text"


class VectorNumericType(str, Enum):
    FLOAT32 = "float32"
