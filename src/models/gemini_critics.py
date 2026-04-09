import os
import sys
import json
import time
from dotenv import load_dotenv


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import Literal


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')


sys.path.append(BASE_DIR)
from src.utils import gerar_chave

load_dotenv(os.path.join(BASE_DIR, '.env'))

AspectosPermitidos = Literal["Roteiro", "Direção", "Fotografia", "Atuação", "Montagem", "Som", "Arte"]
NiveisPermitidos = Literal["Muito Ruim", "Ruim", "Neutro", "Bom", "Muito Bom"]

class AnaliseDetalhada(BaseModel):
    aspecto: AspectosPermitidos = Field(description="Apenas um dos aspectos listados.")
    ponto_chave: str = Field(description="O que foi observado nas resenhas")
    tipo: Literal["Elogio", "Crítica"] = Field(description="Exatamente 'Elogio' ou 'Crítica'")

class AnaliseCriticaOutput(BaseModel):
    analises: list[AnaliseDetalhada]
    consenso: str = Field(description="Resumo do consenso geral")

class MetricaGrafico(BaseModel):
    aspecto: AspectosPermitidos
    nivel: NiveisPermitidos
    tag_tendencia: str = Field(description="Deve ser exatamente Aspecto_Nivel (ex: Roteiro_Ruim)")

class TagsTendenciasOutput(BaseModel):
    metricas: list[MetricaGrafico]
    sentimento_geral_score: int = Field(description="Nota de 1 a 5")


parser_analise = JsonOutputParser(pydantic_object=AnaliseCriticaOutput)
parser_tags = JsonOutputParser(pydantic_object=TagsTendenciasOutput)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.1,
    max_output_tokens=5000
)

system_critico = """
## Persona: Você é um Crítico de Cinema e Pesquisador da Cinematografia Nacional.
Sua especialidade é a análise de recepção: você traduz o que o público leigo sente em métricas técnicas de cinema brasileiro.

DIRETRIZES DE EXTRAÇÃO:
1. TRADUÇÃO TÉCNICA: Se o público diz "o filme é lento", identifique se o problema é a 'Montagem' ou 'Roteiro'.
2. FOCO NACIONAL: Valorize aspectos da produção brasileira.
3. CATEGORIAS ESTRITAS: Classifique ÚNICA E EXCLUSIVAMENTE nestes aspectos: [Roteiro, Direção, Fotografia, Atuação, Montagem, Som, Arte].
4. FILTRO DE RUÍDO: Ignore "amei" ou "odiei". Busque o substantivo que gerou a emoção.
"""

human_critico = "Analise as resenhas do filme \"{titulo}\" ({ano}):\n{resenhas}\n\n{format_instructions}"

prompt_critico = ChatPromptTemplate.from_messages([
    ("system", system_critico),
    ("human", human_critico),
]).partial(format_instructions=parser_analise.get_format_instructions())

system_analista = """
## Persona: Analista de Dados Sênior especializado em Indústria Audiovisual.
Converta análises textuais em métricas.
1. CATEGORIAS: [Roteiro, Direção, Atuação, Fotografia, Montagem, Som, Arte].
2. NÍVEIS: [Muito Ruim, Ruim, Neutro, Bom, Muito Bom].
3. SCORE: Calcule o 'sentimento_geral_score' de 1 a 5.
"""

human_analista = "Gere os dados estruturados baseados nesta análise:\n{analise_anterior}\n\n{format_instructions}"

prompt_tags = ChatPromptTemplate.from_messages([
    ("system", system_analista),
    ("human", human_analista),
]).partial(format_instructions=parser_tags.get_format_instructions())

chain_analise = prompt_critico | llm | parser_analise
chain_tags = prompt_tags | llm | parser_tags

def executar_pipeline_llm(dados):
    analise_raw = chain_analise.invoke(dados)
    tags_raw = chain_tags.invoke({"analise_anterior": json.dumps(analise_raw)})
    return {**analise_raw, "metadados_grafico": tags_raw}


def processar_resenhas():
    input_path = os.path.join(RAW_DIR, 'reviews_final.json')
    output_path = os.path.join(PROCESSED_DIR, 'reviews_analisadas_gemini.json')
    
    if not os.path.exists(input_path):
        print(f" ERRO: Arquivo base não encontrado: {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        dados_filmes = json.load(f)

    resultados = []
    total_filmes = len(dados_filmes)
    print(f"\nIniciando pipeline Gemini para {total_filmes} filmes...")

    for i, filme in enumerate(dados_filmes):
        titulo = filme.get('title', 'Sem Título')
        ano = filme.get('release_year', '0000')
        resenhas = filme.get('reviews', [])
        qtd_resenhas = len(resenhas)
        
        filme['id_movie'] = gerar_chave(titulo, ano)
        
        print(f"[{i+1}/{total_filmes}] {titulo} ({ano}) - {qtd_resenhas} resenhas.")
        
        if qtd_resenhas <= 3:
            print("   -> Pulado: Resenhas insuficientes.")
            filme['analise_gemini'] = None
        else:
            print("   -> Processando na API Gemini...")
            try:
                texto_resenhas = "\n---\n".join(resenhas)
                resultado = executar_pipeline_llm({
                    "titulo": titulo,
                    "ano": ano,
                    "resenhas": texto_resenhas
                })
                filme['analise_gemini'] = resultado
                print("   ->  Sucesso.")
            except Exception as e:
                print(f"   ->  ERRO na API: {e}")
                filme['analise_gemini'] = {"erro": str(e)}
                
            time.sleep(10)
            
        filme.pop('reviews', None)
        resultados.append(filme)

    # Salva o arquivo final
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, ensure_ascii=False, indent=4)
        
    print(f"\n🚀 Extração via LLM concluída! Arquivo salvo em: {output_path}")

if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    processar_resenhas()