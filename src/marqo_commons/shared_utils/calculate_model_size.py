from marqo_commons.shared_utils import constants


def get_model_size(model_name: str, model_properties: dict) -> (int, float):
    '''
    Return the model size for given model
    Note that the priorities are size_in_properties -> model_name -> model_type -> default size
    '''
    if "model_size" in model_properties:
        return model_properties["model_size"]

    name_info = (model_name + model_properties.get("name", "")).lower().replace("/", "-")
    for name, size in constants.MODEL_NAME_SIZE_MAPPING.items():
        if name in name_info:
            return size

    type = model_properties.get("type", None)
    return constants.MODEL_TYPE_SIZE_MAPPING.get(type, constants.DEFAULT_MODEL_SIZE)