from transformers import AutoModel, AutoTokenizer
from src.constants import MODEL_LIST

def download_models():
    for model_name in MODEL_LIST:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

if __name__ == "__main__":
    download_models()