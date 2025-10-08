from fastapi import FastAPI
from pydantic import BaseModel
from summarizer import Summarizer
import json
import random
import google.generativeai as genai
from dotenv import load_dotenv
import os

app = FastAPI()

class SummarizeResponse(BaseModel):
    resumo: str
    reclamacao_anonimizada: str

@app.get("/")
def hello():
    # Lista os modelos disponíveis
    models = genai.list_models()
    return {"message": "Hello World", "modelos_disponiveis": [m["name"] for m in models]}

@app.get("/summarize/random/gemini", response_model=SummarizeResponse)
def summarize_random_gemini():
    with open("iterations.json", "r", encoding="utf-8") as f:
        data_list = json.load(f)

    instance = random.choice(data_list)
    
    summarizer = Summarizer(instance)
    resumo = summarizer.sum_by_llm_gemini()

    return SummarizeResponse(
        resumo=resumo,
        reclamacao_anonimizada=instance.get("reclamacao_anonimizada")
    )

@app.get("/summarize/random/ollama", response_model=SummarizeResponse)
def summarize_random_ollama():
    with open("iterations.json", "r", encoding="utf-8") as f:
        data_list = json.load(f)

    instance = random.choice(data_list)
    
    summarizer = Summarizer(instance)
    resumo = summarizer.sum_by_llm_ollama()

    return SummarizeResponse(
        resumo=resumo,
        reclamacao_anonimizada=instance.get("reclamacao_anonimizada")
    )
