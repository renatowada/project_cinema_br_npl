import os
import pandas as pd
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
FINAL_DIR = os.path.join(BASE_DIR, 'data', 'final')

def unificar_bases():
    print("Consolidação final do Dataset...")
    
    path_master = os.path.join(PROCESSED_DIR, 'dataset_master_treated.parquet')
    if not os.path.exists(path_master):
        print(" ERRO: Base Master não encontrada.")
        return
    df_final = pd.read_parquet(path_master)
    print(f" Master carregada: {len(df_final)} filmes.")

    path_synopsis = os.path.join(PROCESSED_DIR, 'dataset_synopsis_with_themes.parquet')
    if os.path.exists(path_synopsis):
        df_synopsis = pd.read_parquet(path_synopsis)
        df_final = df_final.merge(
            df_synopsis[['id_movie', 'synopsis', 'tema_principal']], 
            on='id_movie', 
            how='left'
        )
        print(" Temas de NLP anexados com sucesso.")

    path_critics = os.path.join(PROCESSED_DIR, 'reviews_analisadas_gemini.json')
    if os.path.exists(path_critics):
        with open(path_critics, 'r', encoding='utf-8') as f:
            critics_json = json.load(f)
        
        df_critics = pd.DataFrame(critics_json)
        df_critics = df_critics[['id_movie', 'analise_gemini']]
        
        df_final = df_final.merge(df_critics, on='id_movie', how='left')
        print(" -> Sentimentos e Tags do Gemini anexados com sucesso.")

    os.makedirs(FINAL_DIR, exist_ok=True)
    output_path = os.path.join(FINAL_DIR, 'dataset_gramado_completo.parquet')
    
    df_final.to_parquet(output_path, index=False)
    print(f"\n Base final consolidada e salva em: {output_path}")

if __name__ == "__main__":
    print("-"*50)
    print(" CONSTRUINDO DATASET FINAL")
    unificar_bases()