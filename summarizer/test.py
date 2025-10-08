import requests
import json

# 1. Definições da Requisição
url = "http://localhost:11434/api/generate"
modelo = "llama3:instruct"
prompt = "Explique o que é um Large Language Model em uma frase."

# O corpo da requisição (payload) em formato de dicionário Python
payload = {
    "model": modelo,
    "prompt": prompt,
    "stream": False # Define que a resposta deve ser enviada de uma só vez (não em pedaços/streaming)
}

# 2. Execução da Requisição
try:
    # A função requests.post() envia a requisição.
    # O argumento json=payload garante que o dicionário seja serializado
    # corretamente para JSON e que o header Content-Type seja configurado.
    response = requests.post(url, json=payload, timeout=600) # timeout é opcional

    # 3. Verificação e Processamento da Resposta
    # Lança um erro HTTP se o status code não for 200 (OK)
    response.raise_for_status()

    # Obtém o corpo da resposta como um dicionário Python
    data = response.json()

    # Extrai o texto gerado pelo modelo
    texto_gerado = data.get("response")
    
    # 4. Saída
    print(f"Status da Requisição: {response.status_code} (OK)")
    print("-" * 30)
    print(f"Modelo Usado: {data.get('model')}")
    print(f"Resposta do LLM:\n{texto_gerado}")
    print("-" * 30)
    print("Detalhes completos da resposta:")
    print(json.dumps(data, indent=2))
    

except requests.exceptions.RequestException as e:
    # Trata erros como falha de conexão, timeout, erros 4xx/5xx, etc.
    print(f"Ocorreu um erro ao conectar ou processar a requisição: {e}")
    print("Verifique se o container 'ollama' está rodando na porta 11434.")