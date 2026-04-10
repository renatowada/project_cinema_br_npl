import os
import json
import pandas as pd
import sys


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')


sys.path.append(BASE_DIR)
from src.utils import gerar_chave

# Processamento das bases de dados: Master e Sinopses
def processar_master():
    """Lê o master json, corrige tipagens, cria o id_movie e salva como parquet."""
    print("Processando base Master de Filmes...")
    input_path = os.path.join(RAW_DIR, 'movies_master_completo.json')
    
    if not os.path.exists(input_path):
        print(f"ERRO: Arquivo {input_path} não encontrado.")
        print("Mova o 'movies_master_completo.json' para a pasta data/raw/")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        dados = json.load(f)
        
    df_master = pd.DataFrame(dados)

    df_master['id_movie'] = df_master.apply(
        lambda row: gerar_chave(row.get('title'), row.get('release_year')), axis=1
    )

    df_master = df_master.drop_duplicates(subset=['id_movie'], keep='first').copy()
    print(f"Base limpa na origem: {len(df_master)} filmes únicos garantidos.")
  
    df_master['directed by'] = df_master['directed by'].apply(
        lambda x: ', '.join([str(i).strip() for i in x]) if isinstance(x, list) else str(x).strip()
    )
    
    df_master['award_category'] = df_master['award_category'].apply(
        lambda x: x if isinstance(x, list) else []
    )
    df_master['genre'] = df_master['genre'].apply(
        lambda x: x if isinstance(x, list) else []
    )


    # Salvando em Parquet tratado
    output_path = os.path.join(PROCESSED_DIR, 'dataset_master_treated.parquet')
    df_master.to_parquet(output_path, index=False)
    print(f" Base Master salva em: {output_path}")

# Processamento da base de sinopses
def processar_sinopses():
    """Lê o json de sinopses, cria o id_movie e salva como parquet."""
    print("\n Processando base de Sinopses...")
    input_path = os.path.join(RAW_DIR, 'synopsis.json')

    if not os.path.exists(input_path):
        print(f"ERRO: Arquivo {input_path} não encontrado.")
        print("Mova o 'synopsis.json' para a pasta data/raw/")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        dados = json.load(f)
        
    df_synopsis = pd.DataFrame(dados)

    df_synopsis['id_movie'] = df_synopsis.apply(
        lambda row: gerar_chave(row.get('title'), row.get('release_year')), axis=1
    )

    # Salvando em Parquet tratado
    output_path = os.path.join(PROCESSED_DIR, 'dataset_synopsis_treated.parquet')
    df_synopsis.to_parquet(output_path, index=False)
    print(f" Base de Sinopses salva em: {output_path}")


if __name__ == "__main__":
    print("-"*50)
    print(" INICIANDO CONSTRUÇÃO DAS BASES ")
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    processar_master()
    processar_sinopses()
    
    print("\n Finalizado. Rode o script 'nlp_theme_extractor.py' para extrair os temas das sinopses usando IA.")