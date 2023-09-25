from typing import Dict


def _get_sbert_test_properties() -> Dict:
    TEST_MODEL_PROPERTIES = {
            "sentence-transformers/test":
                {"name": "sentence-transformers/all-MiniLM-L6-v1",
                "dimensions": 16,
                "tokens":128,
                "type":"test",
                "notes": ""},
            "test":
                {"name": "sentence-transformers/all-MiniLM-L6-v1",
                "dimensions": 16,
                "tokens":128,
                "type":"test",
                "notes": ""},
    }
    return TEST_MODEL_PROPERTIES