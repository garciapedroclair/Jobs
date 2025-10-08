import requests

ollama_host = "http://localhost:11434"

try:
    response = requests.get(f"{ollama_host}/api/models")
    response.raise_for_status()
    data = response.json()
    print("Modelos disponíveis no Ollama:")
    for model in data:
        print("-", model.get("name", model))
except Exception as e:
    print(f"Erro ao listar modelos: {e}")
