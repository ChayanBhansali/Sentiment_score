from transformers import pipeline

class ModelFactory:
    def __init__(self):
        self.models = {}

    def get_model(self, model_name):
        if model_name not in self.models:
            if model_name == "emotion":
                self.models[model_name] = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
            elif model_name == "education":
                self.models[model_name] = pipeline("text-classification", model="HuggingFaceFW/fineweb-edu-classi", top_k=None)
            else:
                raise ValueError(f"Unknown model name: {model_name}")
        return self.models[model_name]
