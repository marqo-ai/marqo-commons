from enum import Enum


class Modality(str, Enum):
    image = "image"
    text = "text"


class ModelType(str, Enum):
    clip = "clip"
    fp16_clip = "fp16_clip"
    hf = "hf"
    multilingual_clip = "multilingual_clip"
    clip_onnx = "clip_onnx"
    open_clip = "open_clip"
    random = "random"
    sbert_onnx = "sbert_onnx"
    sbert = "sbert"
    test = "test"


class VectorNumericType(str, Enum):
    float32 = "float32"


class IndexSettingsField:
    index_settings = "index_settings"
    index_defaults = "index_defaults"
    treat_urls_and_pointers_as_images = "treat_urls_and_pointers_as_images"
    model = "model"
    model_properties = "model_properties"
    search_model = "search_model"
    search_model_properties = "search_model_properties"
    normalize_embeddings = "normalize_embeddings"

    text_preprocessing = "text_preprocessing"
    split_length = "split_length"
    split_overlap = "split_overlap"
    split_method = "split_method"

    image_preprocessing = "image_preprocessing"
    patch_method = "patch_method"

    number_of_shards = "number_of_shards"
    number_of_replicas = "number_of_replicas"

    ann_parameters = "ann_parameters"
    ann_method = "method"
    ann_method_name = "name"
    ann_metric = "space_type"
    ann_engine = "engine"
    ann_method_parameters = "parameters"

    # method_parameters keys for "method"="hnsw"
    hnsw_ef_construction = "ef_construction"
    hnsw_m = "m"
