# main.py
import os
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, JSON, ForeignKey, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session
from datetime import datetime
from contextlib import asynccontextmanager
from src.factory import ModelFactory

@asynccontextmanager
async def lifespan(app: FastAPI):
    db_url = os.getenv("DATABASE_URL", "sqlite:///./test.db")
    engine = create_engine(db_url, connect_args={"check_same_thread": False} if "sqlite" in db_url else {})
    app.state.session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    app.state.db_base = declarative_base()
    yield
    # Shutdown code
    app.state.session_local.close_all()

# FastAPI setup
app = FastAPI(lifespan=lifespan)

def get_db():
    db = app.state.session_local
    try:
        yield db
    finally:
        db.close()

# Models
class TextEntry(app.state.db_base):
    __tablename__ = "text_entries"
    id = Column(Integer, primary_key=True, index=True)
    input_text = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    results = relationship("AnalysisResult", back_populates="text_entry")

class AnalysisResult(app.state.db_base):
    __tablename__ = "analysis_results"
    id = Column(Integer, primary_key=True, index=True)
    text_id = Column(Integer, ForeignKey("text_entries.id"))
    emotion_scores = Column(JSON)
    education_scores = Column(JSON)
    text_entry = relationship("TextEntry", back_populates="results")

class TextRequest(BaseModel):
    text: str

@app.post("/analyze/")
def analyze_text(request: TextRequest, db: Session = Depends(get_db)):
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
def get_logs(db: Session = Depends(get_db)):
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
