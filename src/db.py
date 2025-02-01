from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, JSON, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime
from sqlalchemy import create_engine
import os

db_base = declarative_base()
db_url = os.getenv("DATABASE_URL", "sqlite:///./test.db")
engine = create_engine(db_url, connect_args={"check_same_thread": False} if "sqlite" in db_url else {})

# Models
class TextEntry(db_base):
    __tablename__ = "text_entries"
    id = Column(Integer, primary_key=True, index=True)
    input_text = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    results = relationship("AnalysisResult", back_populates="text_entry")

class AnalysisResult(db_base):
    __tablename__ = "analysis_results"
    id = Column(Integer, primary_key=True, index=True)
    text_id = Column(Integer, ForeignKey("text_entries.id"))
    emotion_scores = Column(JSON)
    education_scores = Column(JSON)
    text_entry = relationship("TextEntry", back_populates="results")

class TextRequest(BaseModel):
    text: str