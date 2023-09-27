from typing import List

from pydantic import BaseModel, Field
from marqo_commons.shared_utils.enums import Modality, VectorNumericType, ModelType


class ModelProperties(BaseModel):
    name: str = Field(..., title="Model name")
    dimensions: int = Field(..., title="Model dimensions")
    notes: str = Field(..., title="Model notes")
    type: ModelType = Field(..., title="Model types")
    memory_size: float = Field(..., title="Model memory size")
    modality: List[Modality] = Field(..., title="Model modality")
    vector_numeric_type: VectorNumericType = Field(..., title="Model vector numeric type")

