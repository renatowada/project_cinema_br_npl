import os
import sys
import pandas as pd
from tqdm import tqdm
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
from transformers import pipeline


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

sys.path.append(BASE_DIR)


load_dotenv(os.path.join(BASE_DIR, '.env'))

def extrair_temas():
    input_path = os.path.join(PROCESSED_DIR, 'dataset_synopsis_treated.parquet')
    output_path = os.path.join(PROCESSED_DIR, 'dataset_synopsis_with_themes.parquet')

    if not os.path.exists(input_path):
        print(f" ERRO: Base de sinopses não encontrada em: {input_path}")
        print(" Rode o script 'build_base_data.py' primeiro.")
        return

    print("Carregando base de sinopses...")
    df_synopsis = pd.read_parquet(input_path)

    coluna_texto = 'synopsis' 
    if coluna_texto not in df_synopsis.columns:
        print(" ERRO: Coluna 'synopsis' não encontrada no dataframe.")
        return

    # Verifica se o token do HuggingFace está disponível
    token_hf = os.getenv("HF_TOKEN")
    
    if not token_hf:
        print(" AVISO: HF_TOKEN não encontrado no arquivo .env!")
        print("O download pode falhar devido a limites de requisição (Erro 429).")

    print("Carregando modelo HuggingFace (Zero-Shot Classification)...")
    print("Isso pode demorar alguns minutos na primeira vez para baixar o modelo.")
    

    classifier = pipeline(
    "zero-shot-classification", 
    model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", # 'm' de multilingual, para lidar melhor com o português
                                                     # 'DeBERTa' modelo mais inteligente para entender o contexto com menos esforço computacional
                                                     # 'v3' versão otimizada para tarefas de classificação
                                                     # 'base' para um modelo mais leve e rápido
                                                     # 'mnli' especializado em inferência de linguagem natural, ideal para classificação de texto
                                                     # 'xnli' validado no xnli, que é o padrão ouro para testar se uma IA entende lógica em múltiplos idiomas
    token=token_hf
    )


    labels_tematicas = [
        "Interior, Sertão e Natureza", 
        "Violência Urbana e Tráfico", 
        "Drama Familiar e Luto", 
        "Política e Ditadura", 
        "Comédia de Costumes",
        "Fantasia e Folclore"
    ]

    print("\nIniciando extração de temas por IA...")
    
    temas_principais = []
    
    for texto in tqdm(df_synopsis[coluna_texto].fillna(''), desc="Processando Filmes"):
        if len(texto.strip()) < 10:
            temas_principais.append("Sem Sinopse")
            continue
            
        try:

            resultado = classifier(texto, labels_tematicas)

            melhor_tema = resultado['labels'][0]
            temas_principais.append(melhor_tema)
        except Exception as e:
            temas_principais.append("Erro de Extração")

    df_synopsis['tema_principal'] = temas_principais

    df_synopsis.to_parquet(output_path, index=False)
    print(f"\n Extração finalizada! Arquivo com temas salvo em: {output_path}")

if __name__ == "__main__":
    print("-"*50)
    print(" INICIANDO EXTRAÇÃO DE TEMAS NLP ")
    extrair_temas()
    print("\n FIM DA EXTRAÇÃO DE TEMAS. Rode o script 'gemini_critics.py' para análise crítica via LLM.")