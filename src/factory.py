from transformers import pipeline
from src.constants import MODEL_CONFIGS

class ModelFactory:
    def __init__(self):
        self.models = {}
        self.model_config = MODEL_CONFIGS

    def get_model(self, model_name):
        if model_name not in self.models:
            if model_name in self.model_config:
                self.models[model_name] = pipeline(self.model_config[model_name]["task"], model=self.model_config[model_name]["model"], top_k=None)
            else:
                raise ValueError(f"Unknown model name: {model_name}")
        return self.models[model_name]
