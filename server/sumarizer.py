import requests
import os
from dotenv import load_dotenv
import google.generativeai as genai

class Summarizer:
    def __init__(self, reclamacao: str, interacoes: list[str]):
        self.reclamacao = reclamacao
        self.interacoes = interacoes or []

    def _build_prompt(self):
        return (
            "Resuma a reclamação e as interações em até 300 caracteres.\n"
            "Formato:\nResumo: <texto>\n\n"
            f"Reclamação: {self.reclamacao}\n\n"
            + "\n".join(self.interacoes)
        )

    def sum_by_ollama(self):
        ollama_host = "http://ollama:11434"
        payload = {
            "model": "llama3:instruct",
            "system": "Resuma em até 300 caracteres.",
            "prompt": self._build_prompt(),
            "stream": False,
        }
        try:
            resp = requests.post(f"{ollama_host}/api/generate", json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json().get("response", "")
        except Exception as e:
            return f"Error: {e}"

    def sum_by_gemini(self):
        load_dotenv()
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(self._build_prompt())
        return response.text
