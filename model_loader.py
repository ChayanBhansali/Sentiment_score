from transformers import pipeline

def load_emotion_model():
    return pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

def load_education_model():
    return pipeline("text-classification", model="HuggingFaceFW/fineweb-edu-classi", top_k=None)

# Initialize models
emotion_model = load_emotion_model()
education_model = load_education_model()