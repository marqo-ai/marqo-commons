from typing import Dict


def _get_sbert_properties() -> Dict:
    SBERT_MODEL_PROPERTIES = {
            "sentence-transformers/all-MiniLM-L6-v1":
                {"name": "sentence-transformers/all-MiniLM-L6-v1",
                "dimensions": 384,
                "tokens":128,
                "type":"sbert",
                "notes": ""},
            "sentence-transformers/all-MiniLM-L6-v2":
                {"name": "sentence-transformers/all-MiniLM-L6-v2",
                "dimensions": 384,
                "tokens":256,
                "type":"sbert",
                "notes": ""},
            "sentence-transformers/all-mpnet-base-v1":
                {"name": "sentence-transformers/all-mpnet-base-v1",
                "dimensions": 768,
                "tokens":128,
                "type":"sbert",
                "notes": ""},
            "sentence-transformers/all-mpnet-base-v2":
                {"name": "sentence-transformers/all-mpnet-base-v2",
                "dimensions": 768,
                "tokens":128,
                "type":"sbert",
                "notes": ""},
            'sentence-transformers/stsb-xlm-r-multilingual':
                {"name": 'sentence-transformers/stsb-xlm-r-multilingual',
                 "dimensions": 768,
                 "tokens": 128,
                 "type": "sbert",
                 "notes": ""},
            "flax-sentence-embeddings/all_datasets_v3_MiniLM-L12":
                {"name": "flax-sentence-embeddings/all_datasets_v3_MiniLM-L12",
                "dimensions": 384,
                "tokens":128,
                "type":"sbert",
                "notes": ""},
            "flax-sentence-embeddings/all_datasets_v3_MiniLM-L6":
                {"name": "flax-sentence-embeddings/all_datasets_v3_MiniLM-L6",
                "dimensions": 384,
                "tokens":128,
                "type":"sbert",
                "notes": ""},
            "flax-sentence-embeddings/all_datasets_v4_MiniLM-L12":
                {"name": "flax-sentence-embeddings/all_datasets_v4_MiniLM-L12",
                "dimensions": 384,
                "tokens":128,
                "type":"sbert",
                "notes": ""},
            "flax-sentence-embeddings/all_datasets_v4_MiniLM-L6":
                {"name": "flax-sentence-embeddings/all_datasets_v4_MiniLM-L6",
                "dimensions": 384,
                "tokens":128,
                "type":"sbert",
                "notes": ""},

            "flax-sentence-embeddings/all_datasets_v3_mpnet-base":
                {"name": "flax-sentence-embeddings/all_datasets_v3_mpnet-base",
                "dimensions": 768,
                "tokens":128,
                "type":"sbert",
                "notes": ""},
            "flax-sentence-embeddings/all_datasets_v4_mpnet-base":
                {"name": "flax-sentence-embeddings/all_datasets_v4_mpnet-base",
                "dimensions": 768,
                "tokens":128,
                "type":"sbert",
                "notes": ""},
    }
    return SBERT_MODEL_PROPERTIES