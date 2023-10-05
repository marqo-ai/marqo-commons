""" decorator to convert model properties to dict
uses model's to_dict method and iterates over all model properties"""
from functools import wraps


def convert_model_properties_to_dict(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        model_properties = func(*args, **kwargs)
        model_properties_dict = dict()
        for key, val in model_properties.items():
            model_properties_dict[key] = val.to_dict()
        return model_properties_dict
    return wrapper
