import json
import requests
import google.generativeai as genai
import os
from dotenv import load_dotenv
import random
from time import sleep

# --- LangChain/Pydantic Imports ---
from pydantic import BaseModel, Field
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException # Para tratamento de erro de formato

# --- 1. Definição do Schema de Saída (Pydantic) ---
class ResumoReclamacao(BaseModel):
    """Estrutura de dados para o resumo de uma reclamação."""
    Contexto: str = Field(description="Uma descrição concisa e objetiva do cenário e do histórico da reclamação.")
    Problemática: str = Field(description="O cerne do problema, o que motivou a reclamação e a falha do serviço/produto.")
    Solução: str = Field(description="A resolução proposta ou o resultado final da interação, se houver.")

# --- 2. Configuração do LLM Ollama com Saída Estruturada ---

# Configurar o LLM Ollama
# NOTA: OLLAMA_BASE_URL é definido automaticamente como http://localhost:11434 se não for especificado
ollama_llm = ChatOllama(model="llama3:instruct", temperature=0.0) 

# Instruir o LLM a usar o Pydantic Schema para garantir a saída JSON
structured_llm = ollama_llm.with_structured_output(ResumoReclamacao)

# Criar o Prompt
ollama_prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente de resumo. Sua tarefa é analisar o texto do usuário e preencher o objeto JSON estritamente no formato solicitado. Seja descritivo e completo. NÃO adicione introduções ou texto fora do JSON."),
    ("user", "Analise a Reclamação e Interações abaixo e gere o resumo:\n\nReclamação:\n{reclamacao}\n\nInterações:\n{interacoes_autor}")
])

# Criar a Chain (o "pipeline" que junta Prompt -> LLM)
summarize_chain = ollama_prompt | structured_llm

RESUMO_SCHEMA = {
    "type": "object",
    "properties": {
        "Contexto": {
            "type": "string",
            "description": "Resuma o ocorrido a partir da mensagem recebida."
        },
        "Problemática": {
            "type": "string",
            "description": "O cerne do problema, o que motivou a reclamação e a falha do serviço/produto."
        },
        "Solução": {
            "type": "string",
            "description": "A resolução proposta ou o resultado final da interação."
        }
    },
    "required": ["Contexto", "Problemática", "Solução"]
}

# Define o formato exato de saída que você espera
class ResumoReclamacao(BaseModel):
    """Estrutura de dados para o resumo de uma reclamação."""
    Contexto: str = Field(description="Um resumo detalhado do cenário e do histórico da reclamação.")
    Problemática: str = Field(description="O cerne do problema, o que motivou a reclamação e a falha do serviço/produto.")
    Solução: str = Field(description="A resolução proposta ou o resultado final da interação, se houver.")

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

    def sum_by_llm_ollama(self):
        """Retorna o resumo gerado pelo LLM Ollama em formato JSON com texto descritivo."""
        
        # O Ollama usa a chave 'system' para prompts de sistema
        # AQUI É O AJUSTE CRÍTICO: ESPECIFICAMOS QUE OS VALORES DEVEM SER STRINGS COMPLETAS.
        system_prompt = (
            "Você é um assistente de resumo. Sua tarefa é analisar a reclamação e as interações e gerar um resumo "
            "claro e objetivo. Sua resposta DEVE ser um objeto JSON válido, contendo APENAS as chaves: "
            "'Contexto', 'Problemática' e 'Solução'. "
            "O VALOR de cada uma dessas chaves DEVE ser uma STRING de TEXTO descritivo e completo. "
            "NÃO use objetos aninhados, listas, booleanos ou qualquer outro formato. "
            "NÃO inclua nenhum texto, introdução ou explicação além do objeto JSON."
        )

        # O prompt principal que o modelo vai processar (permanece o mesmo)
        user_prompt = f"""
            Reclamação:
            {self.reclamacao}

            Interações:
            {"\n".join(self.interacoes_autor)}
        """
        
        ollama_host = "http://localhost:11434"
        model_name = "llama3:instruct" 

        # 1. Payload com a Chave Mágica: "format": "json" (mantido)
        payload = {
            "model": model_name,
            "system": system_prompt, 
            "prompt": user_prompt,
            "format": "json", # Parâmetro crítico para Saída Estruturada
            "options": {
                "num_predict": 300,
                "temperature": 0.2 # Baixa temperatura ajuda a aderir ao formato
            },
            "stream": False
        }

        try:
            # 2. Execução e Tratamento
            url = f"{ollama_host}/api/generate"
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            
            data = response.json()
            json_string_response = data.get("response", "").strip()

            if not json_string_response:
                return f"Error: Resposta do LLM vazia. JSON completo: {json.dumps(data, indent=2)}"

            # 3. Decodifica a string JSON retornada
            resumo_dict = json.loads(json_string_response)

            # 4. Retorna o dicionário
            # A expectativa é um retorno como: 
            # {"Contexto": "...", "Problemática": "...", "Solução": "..."}
            return resumo_dict 

        except json.JSONDecodeError:
            # Captura se o modelo retornou um JSON malformado
            return f"Error: Resposta do modelo não é um JSON válido. Output Bruto: {json_string_response}"
        
        except requests.exceptions.RequestException as e:
            # ... (Tratamento de erros HTTP/Conexão) ...
            if response is not None and response.content:
                try:
                    error_data = response.json()
                    return f"Error HTTP {response.status_code}: {error_data.get('error', str(e))}"
                except json.JSONDecodeError:
                    return f"Error HTTP {response.status_code}: Não foi possível decodificar o erro da API."
            
            return f"Error de Conexão/Timeout: {str(e)}. Verifique se o Ollama está rodando."

    def sum_by_llm_gemini(self):
        # ... (código de configuração do Gemini) ...
        load_dotenv()
        # Garante que o API Key é configurado antes de usar o modelo
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY não está configurada no seu ambiente.")

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

        model = genai.GenerativeModel('gemini-1.5-flash')

        prompt_with_instructions = """
        Resuma a reclamação e as interações de forma clara e objetiva em um texto narrativo de até 300 caracteres.
        Seu retorno DEVE ser apenas o texto do resumo, precedido pela palavra 'Resumo:'.
        
        Exemplo de formato:
        Resumo: O cliente solicitou o cancelamento do serviço, mas recebeu uma cobrança indevida e está aguardando o estorno há 10 dias.
        
        --- Reclamação e Interações ---
        """

        prompt_with_instructions += self.reclamacao + "\n\n" + "\n".join(self.interacoes_autor)
        
        try:
            response = model.generate_content(prompt_with_instructions)
            
            # Pega o texto completo retornado pelo Gemini
            texto_completo = response.text.strip() 

            # Processamento para remover o prefixo "Resumo:" se o modelo o incluiu
            prefixo = "Resumo:"
            if texto_completo.startswith(prefixo):
                # Remove o prefixo e quaisquer espaços em branco que o sigam
                return texto_completo[len(prefixo):].strip() 
            
            # Se o prefixo não foi incluído ou se a resposta for apenas o resumo
            return texto_completo

        except Exception as e:
            # Captura erros da API Gemini
            return f"Error Gemini: {str(e)}"
    
    def sum_by_llm_ollama_chat(self):
        """Retorna o resumo gerado pelo LLM Ollama usando a rota /api/chat e schema JSON."""
        
        ollama_host = "http://localhost:11434"
        model_name = "llama3:instruct" 

        # 1. Definindo o Prompt de Instrução no 'system' role
        system_prompt = (
            "Você é um assistente de resumo. Sua tarefa é analisar a reclamação e as interações e gerar um "
            "resumo estritamente no formato JSON definido pelo schema de saída. Preencha cada campo de forma "
            "descritiva e completa. NÃO inclua nenhum texto adicional, introdução ou explicação fora do JSON."
        )

        # 2. Montagem da Mensagem para a rota /api/chat
        user_content = f"""
            Analise o texto abaixo e gere o resumo no formato solicitado.

            --- Reclamação ---
            {self.reclamacao}

            --- Interações ---
            {"\n".join(self.interacoes_autor)}
        """

        # 3. Payload para /api/chat
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            # Passamos o schema JSON diretamente no formato
            "format": RESUMO_SCHEMA, 
            "options": {
                "num_predict": 300,
                "temperature": 0.0 # Temperatura mais baixa para máxima aderência ao schema
            },
            "stream": False
        }

        try:
            # 4. Execução da Requisição
            url = f"{ollama_host}/api/chat" # MUDA A ROTA
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            
            data = response.json()
            
            # O texto JSON gerado estará dentro de data['message']['content']
            json_string_response = data.get("message", {}).get("content", "").strip()

            if not json_string_response:
                return f"Error: Resposta do LLM vazia. JSON completo: {json.dumps(data, indent=2)}"

            # 5. Decodifica a string JSON retornada
            # O modelo é forçado pelo 'format' a retornar um JSON válido e estrito.
            resumo_dict = json.loads(json_string_response)

            # 6. Retorna o dicionário
            return resumo_dict 

        except json.JSONDecodeError:
            return f"Error: Resposta do modelo não é um JSON válido. Output Bruto: {json_string_response}"
        
        except requests.exceptions.RequestException as e:
            # ... (Tratamento de erros HTTP/Conexão) ...
            if response is not None and response.content:
                try:
                    error_data = response.json()
                    return f"Error HTTP {response.status_code}: {error_data.get('error', str(e))}"
                except json.JSONDecodeError:
                    return f"Error HTTP {response.status_code}: Não foi possível decodificar o erro da API."
            
            return f"Error de Conexão/Timeout: {str(e)}. Verifique se o Ollama está rodando."

    def sum_by_llm_ollama_langchain(self):
        """Retorna o resumo gerado pelo LLM Ollama usando LangChain (Saída Estruturada)."""
        try:
            # O .invoke() injeta os dados na chain configurada
            resumo_pydantic = summarize_chain.invoke({
                "reclamacao": self.reclamacao,
                "interacoes_autor": "\n".join(self.interacoes_autor)
            })
            
            # Converte o objeto Pydantic validado para um dicionário Python simples
            return resumo_pydantic.model_dump()
            
        except OutputParserException as e:
            # Captura erro de formatação se o LLM falhar miseravelmente (raro com structured_output)
            return {"Error": f"Falha na formatação do Pydantic: {str(e)}"}
        except Exception as e:
            # Captura erros de conexão, Ollama fora do ar, etc.
            return {"Error": f"Falha na execução do Ollama/LangChain: {str(e)}"}



if __name__ == "__main__":
    # Abrir o arquivo original
    with open("iterations.json", "r", encoding="utf-8") as f:
        data_list = json.load(f)

    # Lista para armazenar as novas instâncias
    result = []

    for i, instance in enumerate(data_list[:10]):
        print(f"Processando reclamação {i+1}/10...")
        
        summarizer = Summarizer(instance)  # passa o objeto inteiro
        resumo = summarizer.sum_by_llm_ollama_langchain()

        # sleep(10)  # para evitar limite de taxa
        reclamacao = instance.get("reclamacao_anonimizada", "")

        result.append({
            "reclamação": reclamacao,
            "resposta": resumo
        })

        # Salvar em um novo arquivo JSON
        with open("10_reclamacoes_resumidas_lhama3.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)