
![Marqo Commons Logo](https://uploads-ssl.webflow.com/62dfa8e3960a6e2b47dc7fae/62fdf9cef684e6f16158b094_MARQO%20LOGO-UPDATED-GREEN.svg)

<p align="center">
  <b>
    <a href="https://www.marqo.ai">Website</a> |
    <a href="https://docs.marqo.ai">Documentation</a> |
    <a href="https://demo.marqo.ai">Demos</a> |
    <a href="https://community.marqo.ai">Discourse</a> |
    <a href="https://bit.ly/marqo-slack">Slack Community</a> |
    <a href="https://www.marqo.ai/cloud">Marqo Cloud</a>
  </b>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg">
  </a>
  <a align="center" href="https://bit.ly/marqo-slack">
    <img src="https://img.shields.io/badge/Slack-blueviolet?logo=slack&logoColor=white">
  </a>
</p>

Marqo Commons is a Python package that provides a set of utilities and tools for common tasks related to the Marqo project. It includes functionalities for model management, settings validation, and calculations such as estimating how many vectors can fit in a given memory size.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## Features

- **Model Registry**: Easily fetch model properties of the models used in the Marqo project. Supports popular models from PyTorch, Huggingface, OpenAI, and more. Model size in memory, represented by the key `memory_size` is always in gigabytes (GB).

- **Settings Validation**: Validate project settings and configurations to ensure they meet Marqo's requirements and standards.

## Installation

To install Marqo Commons, you can use pip:

```bash
pip install git+https://github.com/marqo-ai/marqo-commons
```

### Usage
To get the model registry, use:
```python
from marqo_commons.model_registry.model_registry import get_model_properties_dict, get_model_properties_json

# Use either of these (python dict or json format)
MODEL_PROPERTIES = get_model_properties_dict()
MODEL_PROPERTIES_JSON_FORMAT = get_model_properties_json()
```

To use the settings validation function, import like so:
```python
from marqo_commons.settings_validation.settings_validation import validate_index_settings
validate_index_settings(settings_to_validate=my_index_settings_object)
```

### Examples
The model registry will return a dict or json containing each model as a key. The value for each model key is a dict containing its properties.

The following are **required** for all models:
```python
name: str
dimensions: int
notes: str
type: ModelType
memory_size: float
modality: List[Modality]
vector_numeric_type: VectorNumericType
```
Some properties are specific to certain model types.

Here are the current model types, along with example objects for each:
### CLIP:
```python
 'ViT-B/32': {'dimensions': 512,
              'memory_size': 1.0,
              'modality': ['image', 'text'],
              'name': 'ViT-B/32',
              'notes': 'CLIP ViT-B/32',
              'type': 'clip',
              'vector_numeric_type': 'float32'}
```
### FP16_CLIP:
```python
 'fp16/ViT-B/16': {'dimensions': 512,
                   'memory_size': 0.66,
                   'modality': ['image',
                                'text'],
                   'name': 'fp16/ViT-B/16',
                   'notes': 'The faster version (fp16, load from `cuda`) of '
                            'openai clip model',
                   'type': 'fp16_clip',
                   'vector_numeric_type': 'float32'}
```
### HF:
```python
 'hf/all-mpnet-base-v1': {'dimensions': 768,
                          'memory_size': 1.0,
                          'modality': ['text'],
                          'name': 'sentence-transformers/all-mpnet-base-v1',
                          'notes': '',
                          'tokens': 128,
                          'type': 'hf',
                          'vector_numeric_type': 'float32'}
```
### MULTILINGUAL_CLIP
```python
'multilingual-clip/XLM-R Large Vit-B/16+': {'dimensions': 640,
                                             'memory_size': 5.0,
                                             'modality': ['text',
                                                          'image'],
                                             'name': 'multilingual-clip/XLM-R '
                                                     'Large Vit-B/16+',
                                             'notes': '',
                                             'textual_model': 'M-CLIP/XLM-Roberta-Large-Vit-B-16Plus',
                                             'type': 'multilingual_clip',
                                             'vector_numeric_type': 'float32',
                                             'visual_model': 'open_clip/ViT-B-16-plus-240/laion400m_e32'}
```
### ONNX_CLIP
```python
'onnx16/open_clip/RN101-quickgelu/openai': {'dimensions': 512,
                                             'image_mean': None,
                                             'image_std': None,
                                             'memory_size': 1.0,
                                             'modality': ['text',
                                                          'image'],
                                             'name': 'onnx16/open_clip/RN101-quickgelu/openai',
                                             'notes': 'the onnx float16 '
                                                      'version of open_clip '
                                                      'RN101-quickgelu/openai',
                                             'pretrained': 'openai',
                                             'repo_id': 'Marqo/onnx-open_clip-RN101-quickgelu',
                                             'resolution': 224,
                                             'textual_file': 'onnx16-open_clip-RN101-quickgelu-openai-textual.onnx',
                                             'type': 'clip_onnx',
                                             'vector_numeric_type': 'float32',
                                             'visual_file': 'onnx16-open_clip-RN101-quickgelu-openai-visual.onnx'}
```

### OPEN_CLIP
```python
 'open_clip/ViT-B-32/laion2b_e16': {'dimensions': 512,
                                    'memory_size': 1.0,
                                    'modality': ['text',
                                                 'image'],
                                    'name': 'open_clip/ViT-B-32/laion2b_e16',
                                    'notes': 'open_clip models',
                                    'pretrained': 'laion2b_e16',
                                    'type': 'open_clip',
                                    'vector_numeric_type': 'float32'}
```

### SBERT_ONNX:
```python
'onnx/all-MiniLM-L6-v1': {'dimensions': 384,
                           'memory_size': 0.7,
                           'modality': ['text'],
                           'name': 'sentence-transformers/all-MiniLM-L6-v1',
                           'notes': '',
                           'tokens': 128,
                           'type': 'sbert_onnx',
                           'vector_numeric_type': 'float32'}

```
### SBERT:
```python
'all-MiniLM-L6-v1': {'dimensions': 384,
                      'memory_size': 0.7,
                      'modality': ['text'],
                      'name': 'sentence-transformers/all-MiniLM-L6-v1',
                      'notes': '',
                      'tokens': 128,
                      'type': 'sbert',
                      'vector_numeric_type': 'float32'}
```

### RANDOM (for testing purposes):
```python
'random': {'dimensions': 384,
            'memory_size': 0.1,
            'modality': ['text', 'image'],
            'name': 'random',
            'notes': '',
            'tokens': 128,
            'type': 'random',
            'vector_numeric_type': 'float32'}

```
### TEST (for testing purposes):
```python
 'test': {'dimensions': 16,
          'memory_size': 0.66,
          'modality': ['text'],
          'name': 'sentence-transformers/all-MiniLM-L6-v1',
          'notes': '',
          'tokens': 128,
          'type': 'test',
          'vector_numeric_type': 'float32'}
```

### Contributing
We welcome contributions to Marqo Commons! If you'd like to contribute code, report issues, or suggest improvements, please follow these guidelines:

- For bug reports or feature requests, open an issue [here](https://github.com/marqo-ai/marqo-commons/issues).
- To contribute code, fork the repository and create a pull request with your changes.

We appreciate your contributions to make Marqo Commons even better!