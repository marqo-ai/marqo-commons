from typing import Dict


def _get_multilingual_clip_properties() -> Dict:
    """This is moved here from the model registry to avoid a circular import"""
    # Models are from github repo
    # https://github.com/FreddeFrallan/Multilingual-CLIP
    MULTILINGUAL_CLIP_PROPERTIES = {
        "multilingual-clip/XLM-Roberta-Large-Vit-L-14":
            {
                "name": "multilingual-clip/XLM-Roberta-Large-Vit-L-14",
                "visual_model": "openai/ViT-L/14",
                "textual_model": 'M-CLIP/XLM-Roberta-Large-Vit-L-14',
                "dimensions": 768,
                "type": "multilingual_clip",
            },

        "multilingual-clip/XLM-R Large Vit-B/16+":
            {
                "name": "multilingual-clip/XLM-R Large Vit-B/16+",
                "visual_model": "open_clip/ViT-B-16-plus-240/laion400m_e32",
                "textual_model": 'M-CLIP/XLM-Roberta-Large-Vit-B-16Plus',
                "dimensions": 640,
                "type": "multilingual_clip",
            },

        "multilingual-clip/XLM-Roberta-Large-Vit-B-32":
            {
                "name": "multilingual-clip/XLM-Roberta-Large-Vit-B-32",
                "visual_model": "openai/ViT-B/32",
                "textual_model": 'M-CLIP/XLM-Roberta-Large-Vit-B-32',
                "dimensions": 512,
                "type": "multilingual_clip",
            },

        "multilingual-clip/LABSE-Vit-L-14":
            {
                "name": "multilingual-clip/LABSE-Vit-L-14",
                "visual_model": "openai/ViT-L/14",
                "textual_model": 'M-CLIP/LABSE-Vit-L-14',
                "dimensions": 768,
                "type": "multilingual_clip",
            }
    }
    return MULTILINGUAL_CLIP_PROPERTIES