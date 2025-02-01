# main.py
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import sessionmaker, Session
from src.factory import ModelFactory
from src.db import db_base, TextEntry, TextRequest, AnalysisResult, engine

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# FastAPI setup
app = FastAPI()
db_base.metadata.create_all(bind=engine)

@app.post("/analyze/")
def analyze_text(request: TextRequest, db: Session = Depends(lambda: SessionLocal())):
    try:
        text_entry = TextEntry(input_text=request.text)
        db.add(text_entry)
        db.commit()
        db.refresh(text_entry)

        model_factory = ModelFactory()
        emotion_model = model_factory.get_model("emotion")
        education_model = model_factory.get_model("education")
        emotion_scores = emotion_model(request.text)
        education_scores = education_model(request.text)

        analysis_result = AnalysisResult(
            text_id=text_entry.id,
            emotion_scores=emotion_scores,
            education_scores=education_scores
        )
        db.add(analysis_result)
        db.commit()
        db.refresh(analysis_result)

        return {
            "id": text_entry.id,
            "text": request.text,
            "emotion_scores": emotion_scores,
            "education_scores": education_scores,
            "timestamp": text_entry.timestamp
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/logs/")
def get_logs(db: Session = Depends(lambda: SessionLocal())):
    logs = db.query(TextEntry).join(AnalysisResult).all()
    response = []
    for log in logs:
        response.append({
            "id": log.id,
            "text": log.input_text,
            "emotion_scores": log.results[0].emotion_scores,
            "education_scores": log.results[0].education_scores,
            "timestamp": log.timestamp
        })
    return response
