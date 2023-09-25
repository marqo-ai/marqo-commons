from typing import Dict

from marqo_commons.model_registry.model_properties_object import ModelProperties, VectorNumericType, Modality


class OnnxClipModelPropeties(ModelProperties):
    vector_numeric_type: VectorNumericType = VectorNumericType.float32
    modality: list[Modality] = [Modality.text, Modality.image]
    type: str = "clip_onnx"
    memory_size: int = 0  # TODO: add memory size
    repo_id: str
    visual_file: str
    textual_file: str
    tokens: int
    notes: str = ""


def _get_onnx_clip_properties() -> Dict:
    ONNX_CLIP_MODEL_PROPERTIES = {
        "onnx32/openai/ViT-L/14":
            {
                "name":"onnx32/openai/ViT-L/14",
                "dimensions" : 768,
                "type":"clip_onnx",
                "notes":"the onnx float32 version of openai ViT-L/14",
                "repo_id": "Marqo/onnx-openai-ViT-L-14",
                "visual_file": "onnx32-openai-ViT-L-14-visual.onnx",
                "textual_file": "onnx32-openai-ViT-L-14-textual.onnx",
                "token": None,
                "resolution" : 224,
            },
        "onnx16/openai/ViT-L/14":
            {
                "name": "onnx16/openai/ViT-L/14",
                "dimensions": 768,
                "type": "clip_onnx",
                "notes": "the onnx float16 version of openai ViT-L/14",
                "repo_id": "Marqo/onnx-openai-ViT-L-14",
                "visual_file": "onnx16-openai-ViT-L-14-visual.onnx",
                "textual_file": "onnx16-openai-ViT-L-14-textual.onnx",
                "token": None,
                "resolution" : 224,
            },
        "onnx32/open_clip/ViT-L-14/openai":
            {
                "name": "onnx32/open_clip/ViT-L-14/openai",
                "dimensions": 768,
                "type": "clip_onnx",
                "notes": "the onnx float32 version of open_clip ViT-L-14/openai",
                "repo_id": "Marqo/onnx-open_clip-ViT-L-14",
                "visual_file": "onnx32-open_clip-ViT-L-14-openai-visual.onnx",
                "textual_file": "onnx32-open_clip-ViT-L-14-openai-textual.onnx",
                "token": None,
                "resolution": 224,
                "pretrained": "openai"
            },
        "onnx16/open_clip/ViT-L-14/openai":
            {
                "name": "onnx16/open_clip/ViT-L-14/openai",
                "dimensions": 768,
                "type": "clip_onnx",
                "notes": "the onnx float16 version of open_clip ViT-L-14/openai",
                "repo_id": "Marqo/onnx-open_clip-ViT-L-14",
                "visual_file": "onnx16-open_clip-ViT-L-14-openai-visual.onnx",
                "textual_file": "onnx16-open_clip-ViT-L-14-openai-textual.onnx",
                "token": None,
                "resolution": 224,
                "pretrained": "openai"
            },
        "onnx32/open_clip/ViT-L-14/laion400m_e32":
            {
                "name" : "onnx32/open_clip/ViT-L-14/laion400m_e32",
                "dimensions" : 768,
                "type" : "clip_onnx",
                "notes": "the onnx float32 version of open_clip ViT-L-14/lainon400m_e32",
                "repo_id" : "Marqo/onnx-open_clip-ViT-L-14",
                "visual_file" : "onnx32-open_clip-ViT-L-14-laion400m_e32-visual.onnx",
                "textual_file" : "onnx32-open_clip-ViT-L-14-laion400m_e32-textual.onnx",
                "token" : None,
                "resolution" : 224,
                "pretrained" : "laion400m_e32"
            },
        "onnx16/open_clip/ViT-L-14/laion400m_e32":
            {
                "name": "onnx16/open_clip/ViT-L-14/laion400m_e32",
                "dimensions": 768,
                "type": "clip_onnx",
                "notes": "the onnx float16 version of open_clip ViT-L-14/lainon400m_e32",
                "repo_id": "Marqo/onnx-open_clip-ViT-L-14",
                "visual_file": "onnx16-open_clip-ViT-L-14-laion400m_e32-visual.onnx",
                "textual_file": "onnx16-open_clip-ViT-L-14-laion400m_e32-textual.onnx",
                "token": None,
                "resolution": 224,
                "pretrained": "laion400m_e32"
            },
        "onnx32/open_clip/ViT-L-14/laion2b_s32b_b82k":
            {
                "name": "onnx32/open_clip/ViT-L-14/laion2b_s32b_b82k",
                "dimensions": 768,
                "type": "clip_onnx",
                "notes": "the onnx float32 version of open_clip ViT-L-14/laion2b_s32b_b82k",
                "repo_id": "Marqo/onnx-open_clip-ViT-L-14",
                "visual_file": "onnx32-open_clip-ViT-L-14-laion2b_s32b_b82k-visual.onnx",
                "textual_file": "onnx32-open_clip-ViT-L-14-laion2b_s32b_b82k-textual.onnx",
                "token": None,
                "resolution": 224,
                "pretrained": "laionb_s32b_b82k",
                "image_mean" : (0.5, 0.5, 0.5),
                "image_std" : (0.5, 0.5, 0.5),

            },
        "onnx16/open_clip/ViT-L-14/laion2b_s32b_b82k":
            {
                "name": "onnx16/open_clip/ViT-L-14/laion2b_s32b_b82k",
                "dimensions": 768,
                "type": "clip_onnx",
                "notes": "the onnx float16 version of open_clip ViT-L-14/laion2b_s32b_b82k",
                "repo_id": "Marqo/onnx-open_clip-ViT-L-14",
                "visual_file": "onnx16-open_clip-ViT-L-14-laion2b_s32b_b82k-visual.onnx",
                "textual_file": "onnx16-open_clip-ViT-L-14-laion2b_s32b_b82k-textual.onnx",
                "token": None,
                "resolution": 224,
                "pretrained": "laionb_s32b_b82k",
                "image_mean": (0.5, 0.5, 0.5),
                 "image_std": (0.5, 0.5, 0.5),
            },
        "onnx32/open_clip/ViT-L-14-336/openai":
            {
                "name": "onnx32/open_clip/ViT-L-14-336/openai",
                "dimensions": 768,
                "type": "clip_onnx",
                "notes": "the onnx float32 version of open_clip ViT-L-14-336/openai",
                "repo_id": "Marqo/onnx-open_clip-ViT-L-14-336",
                "visual_file": "onnx32-open_clip-ViT-L-14-336-openai-visual.onnx",
                "textual_file": "onnx32-open_clip-ViT-L-14-336-openai-textual.onnx",
                "token": None,
                "resolution": 336,
                "pretrained": "openai",
                "image_mean": None,
                "image_std": None,

            },
        "onnx16/open_clip/ViT-L-14-336/openai":
            {
                "name": "onnx16/open_clip/ViT-L-14-336/openai",
                "dimensions": 768,
                "type": "clip_onnx",
                "notes": "the onnx float16 version of open_clip ViT-L-14-336/openai",
                "repo_id": "Marqo/onnx-open_clip-ViT-L-14-336",
                "visual_file": "onnx16-open_clip-ViT-L-14-336-openai-visual.onnx",
                "textual_file": "onnx16-open_clip-ViT-L-14-336-openai-textual.onnx",
                "token": None,
                "resolution": 336,
                "pretrained": "openai",
                "image_mean": None,
                "image_std": None,
            },

        "onnx32/open_clip/ViT-B-32/openai":
            {
                "name": "onnx32/open_clip/ViT-B-32/openai",
                "dimensions": 512,
                "type": "clip_onnx",
                "notes": "the onnx float32 version of open_clip ViT-B-32/openai",
                "repo_id": "Marqo/onnx-open_clip-ViT-B-32",
                "visual_file": "onnx32-open_clip-ViT-B-32-openai-visual.onnx",
                "textual_file": "onnx32-open_clip-ViT-B-32-openai-textual.onnx",
                "token": None,
                "resolution": 224,
                "pretrained": "openai",
                "image_mean": None,
                "image_std": None,
            },

        "onnx16/open_clip/ViT-B-32/openai":
            {
                "name": "onnx16/open_clip/ViT-B-32/openai",
                "dimensions": 512,
                "type": "clip_onnx",
                "notes": "the onnx float16 version of open_clip ViT-B-32/openai",
                "repo_id": "Marqo/onnx-open_clip-ViT-B-32",
                "visual_file": "onnx16-open_clip-ViT-B-32-openai-visual.onnx",
                "textual_file": "onnx16-open_clip-ViT-B-32-openai-textual.onnx",
                "token": None,
                "resolution": 224,
                "pretrained": "openai",
                "image_mean": None,
                "image_std": None,
            },

        "onnx32/open_clip/ViT-B-32/laion400m_e31":
            {
                "name": "onnx32/open_clip/ViT-B-32/laion400m_e31",
                "dimensions": 512,
                "type": "clip_onnx",
                "notes": "the onnx float32 version of open_clip ViT-B-32/laion400m_e31",
                "repo_id": "Marqo/onnx-open_clip-ViT-B-32",
                "visual_file": "onnx32-open_clip-ViT-B-32-laion400m_e31-visual.onnx",
                "textual_file": "onnx32-open_clip-ViT-B-32-laion400m_e31-textual.onnx",
                "token": None,
                "resolution": 224,
                "pretrained": "laion400m_e31",
                "image_mean": None,
                "image_std": None,
            },

        "onnx16/open_clip/ViT-B-32/laion400m_e31":
            {
                "name": "onnx16/open_clip/ViT-B-32/laion400m_e31",
                "dimensions": 512,
                "type": "clip_onnx",
                "notes": "the onnx float16 version of open_clip ViT-B-32/laion400m_e31",
                "repo_id": "Marqo/onnx-open_clip-ViT-B-32",
                "visual_file": "onnx16-open_clip-ViT-B-32-laion400m_e31-visual.onnx",
                "textual_file": "onnx16-open_clip-ViT-B-32-laion400m_e31-textual.onnx",
                "token": None,
                "resolution": 224,
                "pretrained": "laion400m_e31",
                "image_mean": None,
                "image_std": None,
            },

        "onnx32/open_clip/ViT-B-32/laion400m_e32":
            {
                "name": "onnx32/open_clip/ViT-B-32/laion400m_e32",
                "dimensions": 512,
                "type": "clip_onnx",
                "notes": "the onnx float32 version of open_clip ViT-B-32/laion400m_e32",
                "repo_id": "Marqo/onnx-open_clip-ViT-B-32",
                "visual_file": "onnx32-open_clip-ViT-B-32-laion400m_e32-visual.onnx",
                "textual_file": "onnx32-open_clip-ViT-B-32-laion400m_e32-textual.onnx",
                "token": None,
                "resolution": 224,
                "pretrained": "laion400m_e32",
                "image_mean": None,
                "image_std": None,
            },
        "onnx16/open_clip/ViT-B-32/laion400m_e32":
            {
                "name": "onnx16/open_clip/ViT-B-32/laion400m_e32",
                "dimensions": 512,
                "type": "clip_onnx",
                "notes": "the onnx float16 version of open_clip ViT-B-32/laion400m_e32",
                "repo_id": "Marqo/onnx-open_clip-ViT-B-32",
                "visual_file": "onnx16-open_clip-ViT-B-32-laion400m_e32-visual.onnx",
                "textual_file": "onnx16-open_clip-ViT-B-32-laion400m_e32-textual.onnx",
                "token": None,
                "resolution": 224,
                "pretrained": "laion400m_e32",
                "image_mean": None,
                "image_std": None,
            },

        "onnx32/open_clip/ViT-B-32/laion2b_e16":
            {
                "name": "onnx32/open_clip/ViT-B-32/laion2b_e16",
                "dimensions": 512,
                "type": "clip_onnx",
                "notes": "the onnx float32 version of open_clip ViT-B-32/laion2b_e16",
                "repo_id": "Marqo/onnx-open_clip-ViT-B-32",
                "visual_file": "onnx32-open_clip-ViT-B-32-laion2b_e16-visual.onnx",
                "textual_file": "onnx32-open_clip-ViT-B-32-laion2b_e16-textual.onnx",
                "token": None,
                "resolution": 224,
                "pretrained": "laion2b_e16",
                "image_mean": None,
                "image_std": None,
            },

        "onnx16/open_clip/ViT-B-32/laion2b_e16":
            {
                "name": "onnx16/open_clip/ViT-B-32/laion2b_e16",
                "dimensions": 512,
                "type": "clip_onnx",
                "notes": "the onnx float16 version of open_clip ViT-B-32/laion2b_e16",
                "repo_id": "Marqo/onnx-open_clip-ViT-B-32",
                "visual_file": "onnx16-open_clip-ViT-B-32-laion2b_e16-visual.onnx",
                "textual_file": "onnx16-open_clip-ViT-B-32-laion2b_e16-textual.onnx",
                "token": None,
                "resolution": 224,
                "pretrained": "laion2b_e16",
                "image_mean": None,
                "image_std": None,
            },

        'onnx32/open_clip/ViT-B-32-quickgelu/openai':
            {
              'name': 'onnx32/open_clip/ViT-B-32-quickgelu/openai',
              'dimensions': 512,
              'type': 'clip_onnx',
              'notes': 'the onnx float32 version of open_clip ViT-B-32-quickgelu/openai',
              'repo_id': 'Marqo/onnx-open_clip-ViT-B-32-quickgelu',
              'visual_file': 'onnx32-open_clip-ViT-B-32-quickgelu-openai-visual.onnx',
              'textual_file': 'onnx32-open_clip-ViT-B-32-quickgelu-openai-textual.onnx',
              'token': None,
              'resolution': 224, 'pretrained': 'openai',
              'image_mean': None,
              'image_std': None
             },

        'onnx16/open_clip/ViT-B-32-quickgelu/openai':
            {
               'name': 'onnx16/open_clip/ViT-B-32-quickgelu/openai',
               'dimensions': 512,
               'type': 'clip_onnx',
               'notes': 'the onnx float16 version of open_clip ViT-B-32-quickgelu/openai',
               'repo_id': 'Marqo/onnx-open_clip-ViT-B-32-quickgelu',
               'visual_file': 'onnx16-open_clip-ViT-B-32-quickgelu-openai-visual.onnx',
               'textual_file': 'onnx16-open_clip-ViT-B-32-quickgelu-openai-textual.onnx',
               'token': None,
               'resolution': 224,
               'pretrained': 'openai',
               'image_mean': None,
               'image_std': None
            },

        'onnx32/open_clip/ViT-B-32-quickgelu/laion400m_e31':
            {
                'name': 'onnx32/open_clip/ViT-B-32-quickgelu/laion400m_e31',
                'dimensions': 512,
                'type': 'clip_onnx',
                'notes': 'the onnx float32 version of open_clip ViT-B-32-quickgelu/laion400m_e31',
                'repo_id': 'Marqo/onnx-open_clip-ViT-B-32-quickgelu',
                'visual_file': 'onnx32-open_clip-ViT-B-32-quickgelu-laion400m_e31-visual.onnx',
                'textual_file': 'onnx32-open_clip-ViT-B-32-quickgelu-laion400m_e31-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'laion400m_e31',
                'image_mean': None,
                'image_std': None
            },

        'onnx16/open_clip/ViT-B-32-quickgelu/laion400m_e31':
            {
                'name': 'onnx16/open_clip/ViT-B-32-quickgelu/laion400m_e31',
                'dimensions': 512,
                'type': 'clip_onnx',
                'notes': 'the onnx float16 version of open_clip ViT-B-32-quickgelu/laion400m_e31',
                'repo_id': 'Marqo/onnx-open_clip-ViT-B-32-quickgelu',
                'visual_file': 'onnx16-open_clip-ViT-B-32-quickgelu-laion400m_e31-visual.onnx',
                'textual_file': 'onnx16-open_clip-ViT-B-32-quickgelu-laion400m_e31-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'laion400m_e31',
                'image_mean': None,
                'image_std': None
            },

        'onnx16/open_clip/ViT-B-32-quickgelu/laion400m_e32':
            {
                'name': 'onnx16/open_clip/ViT-B-32-quickgelu/laion400m_e32',
                'dimensions': 512,
                'type': 'clip_onnx',
                'notes': 'the onnx float16 version of open_clip ViT-B-32-quickgelu/laion400m_e32',
                'repo_id': 'Marqo/onnx-open_clip-ViT-B-32-quickgelu',
                'visual_file': 'onnx16-open_clip-ViT-B-32-quickgelu-laion400m_e32-visual.onnx',
                'textual_file': 'onnx16-open_clip-ViT-B-32-quickgelu-laion400m_e32-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'laion400m_e32',
                'image_mean': None,
                'image_std': None
            },

        'onnx32/open_clip/ViT-B-32-quickgelu/laion400m_e32':
            {
                'name': 'onnx32/open_clip/ViT-B-32-quickgelu/laion400m_e32',
                'dimensions': 512,
                'type': 'clip_onnx',
                'notes': 'the onnx float32 version of open_clip ViT-B-32-quickgelu/laion400m_e32',
                'repo_id': 'Marqo/onnx-open_clip-ViT-B-32-quickgelu',
                'visual_file': 'onnx32-open_clip-ViT-B-32-quickgelu-laion400m_e32-visual.onnx',
                'textual_file': 'onnx32-open_clip-ViT-B-32-quickgelu-laion400m_e32-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'laion400m_e32',
                'image_mean': None,
                'image_std': None
            },

        'onnx16/open_clip/ViT-B-16/openai':
            {
                'name': 'onnx16/open_clip/ViT-B-16/openai',
                'dimensions': 512,
                'type': 'clip_onnx',
                'notes': 'the onnx float16 version of open_clip ViT-B-16/openai',
                'repo_id': 'Marqo/onnx-open_clip-ViT-B-16',
                'visual_file': 'onnx16-open_clip-ViT-B-16-openai-visual.onnx',
                'textual_file': 'onnx16-open_clip-ViT-B-16-openai-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'openai',
                'image_mean': None,
                'image_std': None
             },

        'onnx32/open_clip/ViT-B-16/openai':
            {
                'name': 'onnx32/open_clip/ViT-B-16/openai',
                'dimensions': 512,
                'type': 'clip_onnx',
                'notes': 'the onnx float32 version of open_clip ViT-B-16/openai',
                'repo_id': 'Marqo/onnx-open_clip-ViT-B-16',
                'visual_file': 'onnx32-open_clip-ViT-B-16-openai-visual.onnx',
                'textual_file': 'onnx32-open_clip-ViT-B-16-openai-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'openai',
                'image_mean': None,
                'image_std': None
            },

        'onnx16/open_clip/ViT-B-16/laion400m_e31':
            {
                'name': 'onnx16/open_clip/ViT-B-16/laion400m_e31',
                'dimensions': 512,
                'type': 'clip_onnx',
                'notes': 'the onnx float16 version of open_clip ViT-B-16/laion400m_e31',
                'repo_id': 'Marqo/onnx-open_clip-ViT-B-16',
                'visual_file': 'onnx16-open_clip-ViT-B-16-laion400m_e31-visual.onnx',
                'textual_file': 'onnx16-open_clip-ViT-B-16-laion400m_e31-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'laion400m_e31',
                'image_mean': None,
                'image_std': None
            },

        'onnx32/open_clip/ViT-B-16/laion400m_e31':
            {
                'name': 'onnx32/open_clip/ViT-B-16/laion400m_e31',
                'dimensions': 512,
                'type': 'clip_onnx',
                'notes': 'the onnx float32 version of open_clip ViT-B-16/laion400m_e31',
                'repo_id': 'Marqo/onnx-open_clip-ViT-B-16',
                'visual_file': 'onnx32-open_clip-ViT-B-16-laion400m_e31-visual.onnx',
                'textual_file': 'onnx32-open_clip-ViT-B-16-laion400m_e31-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'laion400m_e31',
                'image_mean': None,
                'image_std': None
            },

        'onnx16/open_clip/ViT-B-16/laion400m_e32':
            {
                'name': 'onnx16/open_clip/ViT-B-16/laion400m_e32',
                'dimensions': 512,
                'type': 'clip_onnx',
                'notes': 'the onnx float16 version of open_clip ViT-B-16/laion400m_e32',
                'repo_id': 'Marqo/onnx-open_clip-ViT-B-16',
                'visual_file': 'onnx16-open_clip-ViT-B-16-laion400m_e32-visual.onnx',
                'textual_file': 'onnx16-open_clip-ViT-B-16-laion400m_e32-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'laion400m_e32',
                'image_mean': None,
                'image_std': None
            },

        'onnx32/open_clip/ViT-B-16/laion400m_e32':
            {
                'name': 'onnx32/open_clip/ViT-B-16/laion400m_e32',
                'dimensions': 512,
                'type': 'clip_onnx',
                'notes': 'the onnx float32 version of open_clip ViT-B-16/laion400m_e32',
                'repo_id': 'Marqo/onnx-open_clip-ViT-B-16',
                'visual_file': 'onnx32-open_clip-ViT-B-16-laion400m_e32-visual.onnx',
                'textual_file': 'onnx32-open_clip-ViT-B-16-laion400m_e32-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'laion400m_e32',
                'image_mean': None,
                'image_std': None
            },

        'onnx16/open_clip/ViT-B-16-plus-240/laion400m_e31':
            {
                'name': 'onnx16/open_clip/ViT-B-16-plus-240/laion400m_e31',
                'dimensions': 640,
                'type': 'clip_onnx',
                'notes': 'the onnx float16 version of open_clip ViT-B-16-plus-240/laion400m_e31',
                'repo_id': 'Marqo/onnx-open_clip-ViT-B-16-plus-240',
                'visual_file': 'onnx16-open_clip-ViT-B-16-plus-240-laion400m_e31-visual.onnx',
                'textual_file': 'onnx16-open_clip-ViT-B-16-plus-240-laion400m_e31-textual.onnx',
                'token': None,
                'resolution': 240,
                'pretrained': 'laion400m_e31',
                'image_mean': None,
                'image_std': None
            },

        'onnx32/open_clip/ViT-B-16-plus-240/laion400m_e31':
            {
                'name': 'onnx32/open_clip/ViT-B-16-plus-240/laion400m_e31',
                'dimensions': 640, 'type': 'clip_onnx',
                'notes': 'the onnx float32 version of open_clip ViT-B-16-plus-240/laion400m_e31',
                'repo_id': 'Marqo/onnx-open_clip-ViT-B-16-plus-240',
                'visual_file': 'onnx32-open_clip-ViT-B-16-plus-240-laion400m_e31-visual.onnx',
                'textual_file': 'onnx32-open_clip-ViT-B-16-plus-240-laion400m_e31-textual.onnx',
                'token': None,
                'resolution': 240,
                'pretrained': 'laion400m_e31',
                'image_mean': None,
                'image_std': None
            },

        'onnx16/open_clip/ViT-B-16-plus-240/laion400m_e32':
            {
                'name': 'onnx16/open_clip/ViT-B-16-plus-240/laion400m_e32',
                'dimensions': 640,
                'type': 'clip_onnx',
                'notes': 'the onnx float16 version of open_clip ViT-B-16-plus-240/laion400m_e32',
                'repo_id': 'Marqo/onnx-open_clip-ViT-B-16-plus-240',
                'visual_file': 'onnx16-open_clip-ViT-B-16-plus-240-laion400m_e32-visual.onnx',
                'textual_file': 'onnx16-open_clip-ViT-B-16-plus-240-laion400m_e32-textual.onnx',
                'token': None,
                'resolution': 240,
                'pretrained': 'laion400m_e32',
                'image_mean': None,
                'image_std': None
            },

        'onnx32/open_clip/ViT-B-16-plus-240/laion400m_e32':
            {
                'name': 'onnx32/open_clip/ViT-B-16-plus-240/laion400m_e32',
                'dimensions': 640, 'type': 'clip_onnx',
                'notes': 'the onnx float32 version of open_clip ViT-B-16-plus-240/laion400m_e32',
                'repo_id': 'Marqo/onnx-open_clip-ViT-B-16-plus-240',
                'visual_file': 'onnx32-open_clip-ViT-B-16-plus-240-laion400m_e32-visual.onnx',
                'textual_file': 'onnx32-open_clip-ViT-B-16-plus-240-laion400m_e32-textual.onnx',
                'token': None,
                'resolution': 240,
                'pretrained': 'laion400m_e32',
                'image_mean': None,
                'image_std': None
            },

        'onnx16/open_clip/ViT-H-14/laion2b_s32b_b79k':
            {
                'name': 'onnx16/open_clip/ViT-H-14/laion2b_s32b_b79k',
                'dimensions': 1024,
                'type': 'clip_onnx',
                'notes': 'the onnx float16 version of open_clip ViT-H-14/laion2b_s32b_b79k',
                'repo_id': 'Marqo/onnx-open_clip-ViT-H-14',
                'visual_file': 'onnx16-open_clip-ViT-H-14-laion2b_s32b_b79k-visual.onnx',
                'textual_file': 'onnx16-open_clip-ViT-H-14-laion2b_s32b_b79k-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'laion2b_s32b_b79k',
                'image_mean': None,
                'image_std': None,
            },

        'onnx32/open_clip/ViT-H-14/laion2b_s32b_b79k':
            {
                'name': 'onnx32/open_clip/ViT-H-14/laion2b_s32b_b79k',
                'dimensions': 1024,
                'type': 'clip_onnx',
                'notes': 'the onnx float32 version of open_clip ViT-H-14/laion2b_s32b_b79k',
                'repo_id': 'Marqo/onnx-open_clip-ViT-H-14',
                'visual_file': 'onnx32-open_clip-ViT-H-14-laion2b_s32b_b79k-visual.zip',
                'textual_file': 'onnx32-open_clip-ViT-H-14-laion2b_s32b_b79k-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'laion2b_s32b_b79k',
                'image_mean': None,
                'image_std': None
            },

        'onnx16/open_clip/ViT-g-14/laion2b_s12b_b42k':
            {
                'name': 'onnx16/open_clip/ViT-g-14/laion2b_s12b_b42k',
                'dimensions': 1024,
                'type': 'clip_onnx',
                'notes': 'the onnx float16 version of open_clip ViT-g-14/laion2b_s12b_b42k',
                'repo_id': 'Marqo/onnx-open_clip-ViT-g-14',
                'visual_file': 'onnx16-open_clip-ViT-g-14-laion2b_s12b_b42k-visual.onnx',
                'textual_file': 'onnx16-open_clip-ViT-g-14-laion2b_s12b_b42k-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'laion2b_s12b_b42k',
                'image_mean': None,
                'image_std': None
            },

        'onnx32/open_clip/ViT-g-14/laion2b_s12b_b42k':
            {
                'name': 'onnx32/open_clip/ViT-g-14/laion2b_s12b_b42k',
                'dimensions': 1024,
                'type': 'clip_onnx',
                'notes': 'the onnx float32 version of open_clip ViT-g-14/laion2b_s12b_b42k',
                'repo_id': 'Marqo/onnx-open_clip-ViT-g-14',
                'visual_file': 'onnx32-open_clip-ViT-g-14-laion2b_s12b_b42k-visual.zip',
                'textual_file': 'onnx32-open_clip-ViT-g-14-laion2b_s12b_b42k-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'laion2b_s12b_b42k',
                'image_mean': None,
                'image_std': None
            },

        'onnx16/open_clip/RN50/openai':
            {
                'name': 'onnx16/open_clip/RN50/openai',
                'dimensions': 1024,
                'type': 'clip_onnx',
                'notes': 'the onnx float16 version of open_clip RN50/openai',
                'repo_id': 'Marqo/onnx-open_clip-RN50',
                'visual_file': 'onnx16-open_clip-RN50-openai-visual.onnx',
                'textual_file': 'onnx16-open_clip-RN50-openai-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'openai',
                'image_mean': None,
                'image_std': None
            },

        'onnx32/open_clip/RN50/openai':
            {
                'name': 'onnx32/open_clip/RN50/openai',
                'dimensions': 1024,
                'type': 'clip_onnx',
                'notes': 'the onnx float32 version of open_clip RN50/openai',
                'repo_id': 'Marqo/onnx-open_clip-RN50',
                'visual_file': 'onnx32-open_clip-RN50-openai-visual.onnx',
                'textual_file': 'onnx32-open_clip-RN50-openai-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'openai',
                'image_mean': None,
                'image_std': None
            },

        'onnx16/open_clip/RN50/yfcc15m':
            {
                'name': 'onnx16/open_clip/RN50/yfcc15m',
                'dimensions': 1024,
                'type': 'clip_onnx',
                'notes': 'the onnx float16 version of open_clip RN50/yfcc15m',
                'repo_id': 'Marqo/onnx-open_clip-RN50',
                'visual_file': 'onnx16-open_clip-RN50-yfcc15m-visual.onnx',
                'textual_file': 'onnx16-open_clip-RN50-yfcc15m-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'yfcc15m',
                'image_mean': None,
                'image_std': None
            },

        'onnx32/open_clip/RN50/yfcc15m':
            {
                'name': 'onnx32/open_clip/RN50/yfcc15m',
                'dimensions': 1024,
                'type': 'clip_onnx',
                'notes': 'the onnx float32 version of open_clip RN50/yfcc15m',
                'repo_id': 'Marqo/onnx-open_clip-RN50',
                'visual_file': 'onnx32-open_clip-RN50-yfcc15m-visual.onnx',
                'textual_file': 'onnx32-open_clip-RN50-yfcc15m-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'yfcc15m',
                'image_mean': None,
                'image_std': None
            },

        'onnx16/open_clip/RN50/cc12m':
            {
                'name': 'onnx16/open_clip/RN50/cc12m',
                'dimensions': 1024,
                'type': 'clip_onnx',
                'notes': 'the onnx float16 version of open_clip RN50/cc12m',
                'repo_id': 'Marqo/onnx-open_clip-RN50',
                'visual_file': 'onnx16-open_clip-RN50-cc12m-visual.onnx',
                'textual_file': 'onnx16-open_clip-RN50-cc12m-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'cc12m',
                'image_mean': None,
                'image_std': None
            },

        'onnx32/open_clip/RN50/cc12m':
            {
                'name': 'onnx32/open_clip/RN50/cc12m',
                'dimensions': 1024,
                'type': 'clip_onnx',
                'notes': 'the onnx float32 version of open_clip RN50/cc12m',
                'repo_id': 'Marqo/onnx-open_clip-RN50',
                'visual_file': 'onnx32-open_clip-RN50-cc12m-visual.onnx',
                'textual_file': 'onnx32-open_clip-RN50-cc12m-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'cc12m',
                'image_mean': None,
                'image_std': None
            },

        'onnx16/open_clip/RN50-quickgelu/openai':
            {
                'name': 'onnx16/open_clip/RN50-quickgelu/openai',
                'dimensions': 1024,
                'type': 'clip_onnx',
                'notes': 'the onnx float16 version of open_clip RN50-quickgelu/openai',
                'repo_id': 'Marqo/onnx-open_clip-RN50-quickgelu',
                'visual_file': 'onnx16-open_clip-RN50-quickgelu-openai-visual.onnx',
                'textual_file': 'onnx16-open_clip-RN50-quickgelu-openai-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'openai',
                'image_mean': None,
                'image_std': None
            },

        'onnx32/open_clip/RN50-quickgelu/openai':
            {
                'name': 'onnx32/open_clip/RN50-quickgelu/openai',
                'dimensions': 1024,
                'type': 'clip_onnx',
                'notes': 'the onnx float32 version of open_clip RN50-quickgelu/openai',
                'repo_id': 'Marqo/onnx-open_clip-RN50-quickgelu',
                'visual_file': 'onnx32-open_clip-RN50-quickgelu-openai-visual.onnx',
                'textual_file': 'onnx32-open_clip-RN50-quickgelu-openai-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'openai',
                'image_mean': None,
                'image_std': None
            },

        'onnx16/open_clip/RN50-quickgelu/yfcc15m':
            {
                'name': 'onnx16/open_clip/RN50-quickgelu/yfcc15m',
                'dimensions': 1024,
                'type': 'clip_onnx',
                'notes': 'the onnx float16 version of open_clip RN50-quickgelu/yfcc15m',
                'repo_id': 'Marqo/onnx-open_clip-RN50-quickgelu',
                'visual_file': 'onnx16-open_clip-RN50-quickgelu-yfcc15m-visual.onnx',
                'textual_file': 'onnx16-open_clip-RN50-quickgelu-yfcc15m-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'yfcc15m',
                'image_mean': None,
                'image_std': None
            },

        'onnx32/open_clip/RN50-quickgelu/yfcc15m':
            {
                'name': 'onnx32/open_clip/RN50-quickgelu/yfcc15m',
                'dimensions': 1024,
                'type': 'clip_onnx',
                'notes': 'the onnx float32 version of open_clip RN50-quickgelu/yfcc15m',
                'repo_id': 'Marqo/onnx-open_clip-RN50-quickgelu',
                'visual_file': 'onnx32-open_clip-RN50-quickgelu-yfcc15m-visual.onnx',
                'textual_file': 'onnx32-open_clip-RN50-quickgelu-yfcc15m-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'yfcc15m',
                'image_mean': None,
                'image_std': None
            },

        'onnx16/open_clip/RN50-quickgelu/cc12m':
            {
                'name': 'onnx16/open_clip/RN50-quickgelu/cc12m',
                'dimensions': 1024,
                'type': 'clip_onnx',
                'notes': 'the onnx float16 version of open_clip RN50-quickgelu/cc12m',
                'repo_id': 'Marqo/onnx-open_clip-RN50-quickgelu',
                'visual_file': 'onnx16-open_clip-RN50-quickgelu-cc12m-visual.onnx',
                'textual_file': 'onnx16-open_clip-RN50-quickgelu-cc12m-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'cc12m',
                'image_mean': None,
                'image_std': None
            },

        'onnx32/open_clip/RN50-quickgelu/cc12m':
            {
                'name': 'onnx32/open_clip/RN50-quickgelu/cc12m',
                'dimensions': 1024,
                'type': 'clip_onnx',
                'notes': 'the onnx float32 version of open_clip RN50-quickgelu/cc12m',
                'repo_id': 'Marqo/onnx-open_clip-RN50-quickgelu',
                'visual_file': 'onnx32-open_clip-RN50-quickgelu-cc12m-visual.onnx',
                'textual_file': 'onnx32-open_clip-RN50-quickgelu-cc12m-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'cc12m',
                'image_mean': None,
                'image_std': None
            },

        'onnx16/open_clip/RN101/openai':
            {
                'name': 'onnx16/open_clip/RN101/openai',
                'dimensions': 512,
                'type': 'clip_onnx',
                'notes': 'the onnx float16 version of open_clip RN101/openai',
                'repo_id': 'Marqo/onnx-open_clip-RN101',
                'visual_file': 'onnx16-open_clip-RN101-openai-visual.onnx',
                'textual_file': 'onnx16-open_clip-RN101-openai-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'openai',
                'image_mean': None,
                'image_std': None
            },

        'onnx32/open_clip/RN101/openai':
            {
                'name': 'onnx32/open_clip/RN101/openai',
                'dimensions': 512,
                'type': 'clip_onnx',
                'notes': 'the onnx float32 version of open_clip RN101/openai',
                'repo_id': 'Marqo/onnx-open_clip-RN101',
                'visual_file': 'onnx32-open_clip-RN101-openai-visual.onnx',
                'textual_file': 'onnx32-open_clip-RN101-openai-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'openai',
                'image_mean': None,
                'image_std': None,
            },

        'onnx16/open_clip/RN101/yfcc15m':
            {
                'name': 'onnx16/open_clip/RN101/yfcc15m',
                'dimensions': 512,
                'type': 'clip_onnx',
                'notes': 'the onnx float16 version of open_clip RN101/yfcc15m',
                'repo_id': 'Marqo/onnx-open_clip-RN101',
                'visual_file': 'onnx16-open_clip-RN101-yfcc15m-visual.onnx',
                'textual_file': 'onnx16-open_clip-RN101-yfcc15m-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'yfcc15m',
                'image_mean': None,
                'image_std': None,
            },

        'onnx32/open_clip/RN101/yfcc15m':
            {
                'name': 'onnx32/open_clip/RN101/yfcc15m',
                'dimensions': 512,
                'type': 'clip_onnx',
                'notes': 'the onnx float32 version of open_clip RN101/yfcc15m',
                'repo_id': 'Marqo/onnx-open_clip-RN101',
                'visual_file': 'onnx32-open_clip-RN101-yfcc15m-visual.onnx',
                'textual_file': 'onnx32-open_clip-RN101-yfcc15m-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'yfcc15m',
                'image_mean': None,
                'image_std': None
            },

        'onnx16/open_clip/RN101-quickgelu/openai':
            {
                'name': 'onnx16/open_clip/RN101-quickgelu/openai',
                'dimensions': 512,
                'type': 'clip_onnx',
                'notes': 'the onnx float16 version of open_clip RN101-quickgelu/openai',
                'repo_id': 'Marqo/onnx-open_clip-RN101-quickgelu',
                'visual_file': 'onnx16-open_clip-RN101-quickgelu-openai-visual.onnx',
                'textual_file': 'onnx16-open_clip-RN101-quickgelu-openai-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'openai',
                'image_mean': None,
                'image_std': None
            },

        'onnx32/open_clip/RN101-quickgelu/openai':
            {
                'name': 'onnx32/open_clip/RN101-quickgelu/openai',
                'dimensions': 512,
                'type': 'clip_onnx',
                'notes': 'the onnx float32 version of open_clip RN101-quickgelu/openai',
                'repo_id': 'Marqo/onnx-open_clip-RN101-quickgelu',
                'visual_file': 'onnx32-open_clip-RN101-quickgelu-openai-visual.onnx',
                'textual_file': 'onnx32-open_clip-RN101-quickgelu-openai-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'openai',
                'image_mean': None,
                'image_std': None
            },

        'onnx16/open_clip/RN101-quickgelu/yfcc15m':
            {'name': 'onnx16/open_clip/RN101-quickgelu/yfcc15m',
             'dimensions': 512,
             'type': 'clip_onnx',
             'notes': 'the onnx float16 version of open_clip RN101-quickgelu/yfcc15m',
             'repo_id': 'Marqo/onnx-open_clip-RN101-quickgelu',
             'visual_file': 'onnx16-open_clip-RN101-quickgelu-yfcc15m-visual.onnx',
             'textual_file': 'onnx16-open_clip-RN101-quickgelu-yfcc15m-textual.onnx',
             'token': None,
             'resolution': 224,
             'pretrained': 'yfcc15m',
             'image_mean': None,
             'image_std': None
             },

        'onnx32/open_clip/RN101-quickgelu/yfcc15m':
            {
                'name': 'onnx32/open_clip/RN101-quickgelu/yfcc15m',
                'dimensions': 512,
                'type': 'clip_onnx',
                'notes': 'the onnx float32 version of open_clip RN101-quickgelu/yfcc15m',
                'repo_id': 'Marqo/onnx-open_clip-RN101-quickgelu',
                'visual_file': 'onnx32-open_clip-RN101-quickgelu-yfcc15m-visual.onnx',
                'textual_file': 'onnx32-open_clip-RN101-quickgelu-yfcc15m-textual.onnx',
                'token': None,
                'resolution': 224,
                'pretrained': 'yfcc15m',
                'image_mean': None,
                'image_std': None
            },

        'onnx16/open_clip/RN50x4/openai':
            {
                'name': 'onnx16/open_clip/RN50x4/openai',
                'dimensions': 640,
                'type': 'clip_onnx',
                'notes': 'the onnx float16 version of open_clip RN50x4/openai',
                'repo_id': 'Marqo/onnx-open_clip-RN50x4',
                'visual_file': 'onnx16-open_clip-RN50x4-openai-visual.onnx',
                'textual_file': 'onnx16-open_clip-RN50x4-openai-textual.onnx',
                'token': None,
                'resolution': 288,
                'pretrained': 'openai',
                'image_mean': None,
                'image_std': None
            },

        'onnx32/open_clip/RN50x4/openai':
            {
                'name': 'onnx32/open_clip/RN50x4/openai',
                'dimensions': 640,
                'type': 'clip_onnx',
                'notes': 'the onnx float32 version of open_clip RN50x4/openai',
                'repo_id': 'Marqo/onnx-open_clip-RN50x4',
                'visual_file': 'onnx32-open_clip-RN50x4-openai-visual.onnx',
                'textual_file': 'onnx32-open_clip-RN50x4-openai-textual.onnx',
                'token': None,
                'resolution': 288,
                'pretrained': 'openai',
                'image_mean': None,
                'image_std': None
            },

        'onnx16/open_clip/RN50x16/openai':
            {
                'name': 'onnx16/open_clip/RN50x16/openai',
                'dimensions': 768,
                'type': 'clip_onnx',
                'notes': 'the onnx float16 version of open_clip RN50x16/openai',
                'repo_id': 'Marqo/onnx-open_clip-RN50x16',
                'visual_file': 'onnx16-open_clip-RN50x16-openai-visual.onnx',
                'textual_file': 'onnx16-open_clip-RN50x16-openai-textual.onnx',
                'token': None,
                'resolution': 384,
                'pretrained': 'openai',
                'image_mean': None,
                'image_std': None
            },

        'onnx32/open_clip/RN50x16/openai':
            {
                'name': 'onnx32/open_clip/RN50x16/openai',
                'dimensions': 768,
                'type': 'clip_onnx',
                'notes': 'the onnx float32 version of open_clip RN50x16/openai',
                'repo_id': 'Marqo/onnx-open_clip-RN50x16',
                'visual_file': 'onnx32-open_clip-RN50x16-openai-visual.onnx',
                'textual_file': 'onnx32-open_clip-RN50x16-openai-textual.onnx',
                'token': None,
                'resolution': 384,
                'pretrained': 'openai',
                'image_mean': None,
                'image_std': None
            },

        'onnx16/open_clip/RN50x64/openai':
            {
                'name': 'onnx16/open_clip/RN50x64/openai',
                'dimensions': 1024,
                'type': 'clip_onnx',
                'notes': 'the onnx float16 version of open_clip RN50x64/openai',
                'repo_id': 'Marqo/onnx-open_clip-RN50x64',
                'visual_file': 'onnx16-open_clip-RN50x64-openai-visual.onnx',
                'textual_file': 'onnx16-open_clip-RN50x64-openai-textual.onnx',
                'token': None,
                'resolution': 448,
                'pretrained': 'openai',
                'image_mean': None,
                'image_std': None
            },

        'onnx32/open_clip/RN50x64/openai':
            {
                'name': 'onnx32/open_clip/RN50x64/openai',
                'dimensions': 1024,
                'type': 'clip_onnx',
                'notes': 'the onnx float32 version of open_clip RN50x64/openai',
                'repo_id': 'Marqo/onnx-open_clip-RN50x64',
                'visual_file': 'onnx32-open_clip-RN50x64-openai-visual.onnx',
                'textual_file': 'onnx32-open_clip-RN50x64-openai-textual.onnx',
                'token': None,
                'resolution': 448,
                'pretrained': 'openai',
                'image_mean': None,
                'image_std': None
            },
    }

    # new_properties = {}
    # for key, value in ONNX_CLIP_MODEL_PROPERTIES.items():
    #     del value["type"]
    #     print(value)
    #     new_properties[key] = OnnxClipModelPropeties(**value)
    # for key, value in new_properties.items():
    #     print(f'"{key}": MultilingualClipProperties(')
    #     print(f'    name="{value.name}",')
    #     print(f'    dimensions={value.dimensions},')
    #     print(').__dict__,')

    return ONNX_CLIP_MODEL_PROPERTIES
