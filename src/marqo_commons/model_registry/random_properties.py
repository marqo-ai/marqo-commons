from typing import Dict


def _get_random_properties() -> Dict:
    RANDOM_MODEL_PROPERTIES = {
            "random":
                {"name": "random",
                "dimensions": 384,
                "tokens":128,
                "type":"random",
                "notes": ""},
            "random/large":
                {"name": "random/large",
                "dimensions": 768,
                "tokens":128,
                "type":"random",
                "notes": ""},
            "random/small":
                {"name": "random/small",
                "dimensions": 32,
                "tokens":128,
                "type":"random",
                "notes": ""},
            "random/medium":
                {"name": "random/medium",
                "dimensions": 128,
                "tokens":128,
                "type":"random",
                "notes": ""},

    }
    return RANDOM_MODEL_PROPERTIES
