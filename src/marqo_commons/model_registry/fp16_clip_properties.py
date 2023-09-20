def _get_fp16_clip_properties() -> Dict:
    FP16_CLIP_MODEL_PROPERTIES = {
        "fp16/ViT-L/14": {
            "name": "fp16/ViT-L/14",
            "dimensions": 768,
            "type": "fp16_clip",
            "notes": "The faster version (fp16, load from `cuda`) of openai clip model"
        },
        'fp16/ViT-B/32':
            {"name": "fp16/ViT-B/32",
             "dimensions": 512,
             "notes": "The faster version (fp16, load from `cuda`) of openai clip model",
             "type": "fp16_clip",
             },
        'fp16/ViT-B/16':
            {"name": "fp16/ViT-B/16",
             "dimensions": 512,
             "notes": "The faster version (fp16, load from `cuda`) of openai clip model",
             "type": "fp16_clip",
             },
    }

    return FP16_CLIP_MODEL_PROPERTIES
