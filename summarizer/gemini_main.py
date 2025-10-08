import json
import requests
import google.generativeai as genai
import os
from dotenv import load_dotenv
from time import sleep

# --- LangChain/Pydantic Imports ---
from pydantic import BaseModel, Field
# CORREÇÃO CRÍTICA: Usar o pacote 'langchain-ollama'
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException 

# --- 1. Definição ÚNICA do Schema de Saída (Pydantic) ---
class ResumoReclamacao(BaseModel):
    """Estrutura de dados para o resumo de uma reclamação."""
    # Descrições detalhadas para guiar o LLM na produção de texto corrido.
    Descrição: str = Field(description="Um resumo detalhado e objetivo do cenário e do histórico da reclamação, incluindo informações cruciais como datas e produtos.")
    Problemática: str = Field(description="O cerne do problema, descrevendo o que motivou a reclamação e a falha específica do serviço ou produto.")
    Solução: str = Field(description="A resolução proposta ou o resultado final da interação, se houver, focando em como o problema foi ou deveria ser resolvido.")

# --- 2. Configuração do LLM Ollama com Saída Estruturada (Global) ---

# Configurar o LLM Ollama. Temperatura baixa garante maior adesão ao formato.
# ollama_llm = ChatOllama(model="llama3:instruct", temperature=0.0)
ollama_llm = ChatOllama(model="qwen2:7b-instruct", temperature=0.0) 

# Instruir o LLM a usar o Pydantic Schema para garantir a saída JSON
structured_llm = ollama_llm.with_structured_output(ResumoReclamacao)

# Criar o Prompt
ollama_prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente de resumo. Sua tarefa é analisar o texto do usuário e preencher o objeto JSON estritamente no formato solicitado. Seja descritivo e completo. NÃO adicione introduções ou texto fora do JSON."),
    ("user", "Analise a Reclamação e Interações abaixo e gere o resumo:\n\nReclamação:\n{reclamacao}\n\nInterações:\n{interacoes_autor}")
])

# Criar a Chain (o "pipeline" que junta Prompt -> LLM)
summarize_chain = ollama_prompt | structured_llm

# --- REMOÇÃO DE DEFINIÇÕES DUPLICADAS E INCONSISTENTES ---
# Removido RESUMO_SCHEMA (duplicado e desnecessário com Pydantic)
# Removida a segunda definição de ResumoReclamacao (duplicada)


class Summarizer:
    def __init__(self, data: dict):
        self.id = data.get("id_reclamacao")
        
        # texto principal anonimizado
        self.reclamacao = data.get("reclamacao_anonimizada")
        
        # interações formatadas para o prompt (Autor: Mensagem)
        self.interacoes_autor = [
            f'{i.get("autor")}: {str(i.get("mensagem_anonimizada"))}'
            for i in data.get("interacoes", [])
            if i.get("mensagem_anonimizada") is not None
        ]

    def sum_by_llm_ollama(self):
        """Retorna o resumo gerado pelo LLM Ollama usando a rota /api/generate com formato JSON."""
        
        system_prompt = (
            "Você é um assistente de resumo. Sua resposta DEVE ser um objeto JSON válido, contendo APENAS as chaves: "
            "'Descrição', 'Problemática' e 'Solução'. "
            "O VALOR de cada uma dessas chaves DEVE ser uma STRING de TEXTO descritivo e completo. "
            "NÃO inclua nenhum texto, introdução ou explicação além do objeto JSON."
        )

        user_prompt = f"""
            Reclamação:
            {self.reclamacao}

            Interações:
            {"\n".join(self.interacoes_autor)}
        """
        
        ollama_host = "http://localhost:11434"
        model_name = "llama3:instruct" 

        payload = {
            "model": model_name,
            "system": system_prompt, 
            "prompt": user_prompt,
            "format": "json", # Parâmetro crítico para Saída Estruturada
            "options": {
                "num_predict": 300,
                "temperature": 0.2
            },
            "stream": False
        }

        try:
            url = f"{ollama_host}/api/generate"
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            
            data = response.json()
            json_string_response = data.get("response", "").strip()

            if not json_string_response:
                return {"Error": f"Resposta do LLM vazia. JSON completo: {json.dumps(data, indent=2)}"}

            resumo_dict = json.loads(json_string_response)
            return resumo_dict 

        except json.JSONDecodeError:
            return {"Error": f"Resposta do modelo não é um JSON válido. Output Bruto: {json_string_response}"}
        
        except requests.exceptions.RequestException as e:
            # Tratamento de erros de requisição
            error_msg = str(e)
            if response is not None and response.content:
                try:
                    error_data = response.json()
                    error_msg = error_data.get('error', error_msg)
                except json.JSONDecodeError:
                    pass
            
            return {"Error": f"Falha de Conexão/HTTP com Ollama: {error_msg}. Verifique se está rodando."}

    def sum_by_llm_ollama_langchain(self):
        """Retorna o resumo gerado pelo LLM Ollama usando LangChain (Saída Estruturada)."""
        try:
            # O .invoke() injeta os dados na chain configurada (que usa o LLM global)
            resumo_pydantic = summarize_chain.invoke({
                "reclamacao": self.reclamacao,
                "interacoes_autor": "\n".join(self.interacoes_autor)
            })
            
            # Converte o objeto Pydantic validado para um dicionário Python simples
            return resumo_pydantic.model_dump()
            
        except OutputParserException as e:
            # Captura erro de formatação se o LLM falhar miseravelmente 
            return {"Error": f"Falha na formatação do Pydantic: {str(e)}"}
        except Exception as e:
            # Captura erros de conexão, Ollama fora do ar, etc.
            return {"Error": f"Falha na execução do Ollama/LangChain: {str(e)}"}

    
    def sum_by_llm_gemini(self):
        """Retorna o resumo gerado pelo LLM Gemini com formato JSON estrito."""
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return {"Error": "GOOGLE_API_KEY não está configurada."}

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')

        user_content = f"""
        Reclamação: {self.reclamacao}
        
        Interações: {"\n".join(self.interacoes_autor)}
        """
        
        # O Gemini usa response_mime_type para forçar a saída JSON
        prompt_system = "Analise o texto e gere um objeto JSON estrito com as chaves \"Descrição\", \"Problemática\" e \"Solução\". O valor de cada chave deve ser uma string de texto corrido. NÃO inclua nenhum texto ou formatação adicional fora do JSON."

        try:
            response = model.generate_content(
                [prompt_system, user_content],
                config={"response_mime_type": "application/json"}
            )

            # A resposta de JSON está em response.text
            return json.loads(response.text)

        except Exception as e:
            return {"Error": f"Falha na execução do Gemini: {str(e)}"}

# --- 4. Bloco de Execução Principal ---
if __name__ == "__main__":
    
    # Carregamento dos dados
    try:
        with open("iterations.json", "r", encoding="utf-8") as f:
            data_list = json.load(f)
    except FileNotFoundError:
        print("ERRO: Arquivo 'iterations.json' não encontrado. Crie um arquivo com seus dados.")
        exit()
    except json.JSONDecodeError:
        print("ERRO: Arquivo 'iterations.json' está malformado.")
        exit()

    result = []
    MAX_ITENS = 10 # Limite para testar
    
    print(f"Iniciando o processamento das primeiras {MAX_ITENS} reclamações com Ollama (LangChain)...")

    for i, instance in enumerate(data_list[:MAX_ITENS]):
        print(f"\n--- Processando {i+1}/{MAX_ITENS} (ID: {instance.get('id_reclamacao', 'N/A')}) ---")
        
        summarizer = Summarizer(instance)
        
        # CHAMADA ABRANGENTE: Usando a função LangChain/Pydantic
        resumo = summarizer.sum_by_llm_ollama_langchain() 

        # sleep(1) # Descomente se for rodar em produção lenta
        
        # Garante que o ID da reclamação está no output
        reclamacao = instance.get("reclamacao_anonimizada", "")

        result.append({
            "id_reclamacao": instance.get("id_reclamacao"),
            "reclamação_original": reclamacao,
            "resumo_ollama_langchain": resumo
        })

        # Salvar o progresso em um novo arquivo JSON a cada iteração
        with open("10_reclamacoes_resumidas_lc_ollama_qwen2_7b.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
    print(f"\nProcessamento concluído. Resultados salvos em '10_reclamacoes_resumidas_lc_ollama.json'.")