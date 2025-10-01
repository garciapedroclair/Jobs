from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from .database import Base, engine, SessionLocal
from .models import Summary
from .summarizer import Summarizer

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Summarizer API")

class SummarizeRequest(BaseModel):
    reclamacao: str
    interacoes: list[str] = []

class SummarizeResponse(BaseModel):
    id: int
    reclamacao: str
    resumo: str

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/summarize", response_model=SummarizeResponse)
def summarize(data: SummarizeRequest, db: Session = Depends(get_db)):
    summarizer = Summarizer(data.reclamacao, data.interacoes)
    resumo = summarizer.sum_by_ollama()

    # salvar no banco
    summary = Summary(reclamacao=data.reclamacao, resumo=resumo)
    db.add(summary)
    db.commit()
    db.refresh(summary)

    return SummarizeResponse(id=summary.id, reclamacao=summary.reclamacao, resumo=summary.resumo)
