import json
import requests
import google.generativeai as genai
import os
from dotenv import load_dotenv
import random
from time import sleep

class Summarizer:
    def __init__(self, data: dict):
        self.id = data.get("id_reclamacao")
        
        # texto principal anonimizado
        self.reclamacao = data.get("reclamacao_anonimizada")
        
        # interações anonimizadas
        self.interacoes = [
            str(i.get("mensagem_anonimizada"))  # Converte o valor para string
            for i in data.get("interacoes", [])
            if i.get("mensagem_anonimizada") is not None    ]

        self.interacoes_autor = [
            f'{i.get("autor")}: {str(i.get("mensagem_anonimizada"))}'
            for i in data.get("interacoes", [])
            if i.get("mensagem_anonimizada") is not None
        ]

    def get_reclamacao(self):
        """Retorna somente o texto da reclamação anonimizada"""
        return self.reclamacao
    
    def get_interacoes(self):
        """Retorna a lista de mensagens das interações anonimizadas"""
        return self.interacoes
    
    def get_interacoes_autor(self):
        """Retorna a lista de mensagens das interações anonimizadas com o autor"""
        return self.interacoes_autor
    
    def sum_by_llm_ollama(self):
        """Retorna o resumo gerado pelo LLM"""
        ollama_host = "http://ollama_sum:11434"  # definido dentro da função

        prompt_text = self.reclamacao + "\n\n" + "\n".join(self.interacoes_autor)
        payload = {
            "model": "lhama3:instruct", # llama3:instruct
            "system": """
                         Resuma a reclamação e as interações de forma clara e objetiva em um texto narrativo de até 300 caracteres.
                         Seguindo o seguinte formato:
                         Reclamação: <Resumo>
                      """,
            "prompt": prompt_text,
            "stream": False
        }

        try:
            response = requests.post(f"{ollama_host}/v1/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except Exception as e:
            return f"Error: {str(e)}"
    
    def sum_by_llm_gemini(self):
        
        # Substitua "SUA_CHAVE_DE_API_AQUI" pela sua chave real
        # genai.configure(api_key="SUA_CHAVE_DE_API_AQUI") 
        load_dotenv()
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

        model = genai.GenerativeModel('gemini-1.5-flash')

        prompt_with_instructions = """
        Resuma a reclamação e as interações de forma clara e objetiva em um texto narrativo de até 300 caracteres.
        Seguindo o seguinte formato:
        
        Resumo: <Resumo>

        --- Reclamação e Interações ---
        """

        prompt_with_instructions += self.reclamacao + "\n\n" + "\n".join(self.interacoes_autor)
        
        response = model.generate_content(prompt_with_instructions)

        # RETORNA o texto da resposta
        return response.text

if __name__ == "__main__":
    # Abrir o arquivo original
    with open("iterations.json", "r", encoding="utf-8") as f:
        data_list = json.load(f)

    # Lista para armazenar as novas instâncias
    result = []

    for i, instance in enumerate(data_list[:50]):
        print(f"Processando reclamação {i+1}/50...")
        
        summarizer = Summarizer(instance)  # passa o objeto inteiro
        resumo = summarizer.sum_by_llm_ollama()

        # sleep(10)  # para evitar limite de taxa
        reclamacao = instance.get("reclamacao_anonimizada", "")

        result.append({
            "reclamação": reclamacao,
            "resposta": resumo
        })

        # Salvar em um novo arquivo JSON
        with open("sum_iterations_gemini.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)