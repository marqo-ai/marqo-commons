from pydantic import BaseModel, Field
from marqo_commons.enums import Modality, VectorNumericType


class ModelProperties(BaseModel):
    name: str = Field(..., title="Model name")
    dimensions: int = Field(..., title="Model dimensions")
    notes: str = Field(..., title="Model notes")
    type: str = Field(..., title="Model types")
    memory_size: int = Field(..., title="Model memory size")
    modality: list[Modality] = Field(..., title="Model modality")
    vector_numeric_type: VectorNumericType = Field(..., title="Model vector numeric type")

