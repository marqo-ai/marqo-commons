from marqo_commons.model_registry.model_registry import get_model_properties_dict

MODEL_PROPERTIES = get_model_properties_dict()


def calculate_maximum_number_of_vectors_in_memory(
    memory_size_in_gb: float, model_name: str, m: int = 16
) -> int:
    """
    Calculates the maximum number of vectors that can be stored in memory.
    """
    memory_size_in_bytes = memory_size_in_gb * 1024 * 1024 * 1024
    vector_size_in_bytes = calculate_memory_usage_per_vector(
        MODEL_PROPERTIES[model_name]["dimensions"],
        MODEL_PROPERTIES[model_name]["vector_numeric_type"],
        m
    )
    return int(memory_size_in_bytes / vector_size_in_bytes)


def calculate_memory_usage_per_vector(dimensions: int, vector_numeric_type: str, m) -> float:
    """
    Calculates the memory usage per vector in bytes.
    """
    vector_type_multiplier = 4 if vector_numeric_type == "float32" else 2
    return (dimensions * vector_type_multiplier + (m * 8)) * 1.1
