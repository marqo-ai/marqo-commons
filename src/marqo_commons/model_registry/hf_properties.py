from typing import Dict


def _get_hf_properties() -> Dict:
    HF_MODEL_PROPERTIES = {
            "hf/all-MiniLM-L6-v1":
                {"name": "sentence-transformers/all-MiniLM-L6-v1",
                "dimensions": 384,
                "tokens":128,
                "type":"hf",
                "notes": ""},
            "hf/all-MiniLM-L6-v2":
                {"name": "sentence-transformers/all-MiniLM-L6-v2",
                "dimensions": 384,
                "tokens":256,
                "type":"hf",
                "notes": ""},
            "hf/all-mpnet-base-v1":
                {"name": "sentence-transformers/all-mpnet-base-v1",
                "dimensions": 768,
                "tokens":128,
                "type":"hf",
                "notes": ""},
            "hf/all-mpnet-base-v2":
                {"name": "sentence-transformers/all-mpnet-base-v2",
                "dimensions": 768,
                "tokens":128,
                "type":"hf",
                "notes": ""},

            "hf/all_datasets_v3_MiniLM-L12":
                {"name": "flax-sentence-embeddings/all_datasets_v3_MiniLM-L12",
                "dimensions": 384,
                "tokens":128,
                "type":"hf",
                "notes": ""},
            "hf/all_datasets_v3_MiniLM-L6":
                {"name": "flax-sentence-embeddings/all_datasets_v3_MiniLM-L6",
                "dimensions": 384,
                "tokens":128,
                "type":"hf",
                "notes": ""},
            "hf/all_datasets_v4_MiniLM-L12":
                {"name": "flax-sentence-embeddings/all_datasets_v4_MiniLM-L12",
                "dimensions": 384,
                "tokens":128,
                "type":"hf",
                "notes": ""},
            "hf/all_datasets_v4_MiniLM-L6":
                {"name": "flax-sentence-embeddings/all_datasets_v4_MiniLM-L6",
                "dimensions": 384,
                "tokens":128,
                "type":"hf",
                "notes": ""},

            "hf/all_datasets_v3_mpnet-base":
                {"name": "flax-sentence-embeddings/all_datasets_v3_mpnet-base",
                "dimensions": 768,
                "tokens":128,
                "type":"hf",
                "notes": ""},
            "hf/all_datasets_v4_mpnet-base":
                {"name": "flax-sentence-embeddings/all_datasets_v4_mpnet-base",
                "dimensions": 768,
                "tokens":128,
                "type":"hf",
                "notes": ""},

            "hf/e5-small":
                {"name": 'intfloat/e5-small',
                 "dimensions": 384,
                 "tokens": 192,
                 "type": "hf",
                 "model_size": 0.1342,
                 "notes": ""},
            "hf/e5-base":
                {"name": 'intfloat/e5-base',
                 "dimensions": 768,
                 "tokens": 192,
                 "type": "hf",
                 "model_size": 0.438,
                 "notes": ""},
            "hf/e5-large":
                {"name": 'intfloat/e5-large',
                 "dimensions": 1024,
                 "tokens": 192,
                 "type": "hf",
                 "model_size": 1.3,
                 "notes": ""},
            "hf/e5-large-unsupervised":
                {"name": 'intfloat/e5-large-unsupervised',
                 "dimensions": 1024,
                 "tokens": 128,
                 "type": "hf",
                 "model_size": 1.3,
                 "notes": ""},
            "hf/e5-base-unsupervised":
                {"name": 'intfloat/e5-base-unsupervised',
                 "dimensions": 768,
                 "tokens": 128,
                 "type": "hf",
                 "model_size": 0.438,
                 "notes": ""},
            "hf/e5-small-unsupervised":
                {"name": 'intfloat/e5-small-unsupervised',
                 "dimensions": 384,
                 "tokens": 128,
                 "type": "hf",
                 "model_size": 0.134,
                 "notes": ""},
            "hf/multilingual-e5-small":
                {"name": 'intfloat/multilingual-e5-small',
                 "dimensions": 384,
                 "tokens": 512,
                 "type": "hf",
                 "model_size": 0.471,
                 "notes": ""},
            "hf/multilingual-e5-base":
                {"name": 'intfloat/multilingual-e5-base',
                 "dimensions": 768,
                 "tokens": 512,
                 "type": "hf",
                 "model_size": 1.11,
                 "notes": ""},
            "hf/multilingual-e5-large":
                {"name": 'intfloat/multilingual-e5-large',
                 "dimensions": 1024,
                 "tokens": 512,
                 "type": "hf",
                 "model_size": 2.24,
                 "notes": ""},
            "hf/e5-small-v2":
                {"name": 'intfloat/e5-small-v2',
                 "dimensions": 384,
                 "tokens": 512,
                 "type": "hf",
                 "model_size": 0.134,
                 "notes": ""},
            "hf/e5-base-v2":
                {"name": 'intfloat/e5-base-v2',
                 "dimensions": 768,
                 "tokens": 512,
                 "type": "hf",
                 "model_size": 0.438,
                 "notes": ""},
            "hf/e5-large-v2":
                {"name": 'intfloat/e5-large-v2',
                 "dimensions": 1024,
                 "tokens": 512,
                 "type": "hf",
                 "model_size": 1.34,
                 "notes": ""},
    }
    return HF_MODEL_PROPERTIES