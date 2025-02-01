MODEL_CONFIGS = {
    "emotion": {
        "task": "text-classification",
        "model": "SamLowe/roberta-base-go_emotions",
    },
    "education": {
        "task": "text-classification",
        "model": "HuggingFaceFW/fineweb-edu-classifier",
    }
}

MODEL_LIST = [model["model"] for model in MODEL_CONFIGS.values()]
