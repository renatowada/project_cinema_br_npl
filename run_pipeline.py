import subprocess
import sys
import os

# Ordem de execução dos scripts
PIPELINE = [
    "src/extract/build_base_data.py",
    "src/models/nlp_theme_extractor.py",
    "src/models/gemini_critics.py",
    "src/pipeline/build_master_parquet.py"
]

def rodar_esteira():
    print("-"*50)
    print(" INICIANDO PIPELINE DE DADOS - Projeto PLN ")

    for script in PIPELINE:
        script_path = os.path.normpath(script)
        
        print(f"\n Executando: {script_path}...")
        
        resultado = subprocess.run([sys.executable, script_path])
        
        if resultado.returncode != 0:
            print(f"\n ERRO: O script {script_path} falhou.")
            print("A esteira foi interrompida.")
            sys.exit(1)

    print("\n" + "-"*50)
    print("  PIPELINE FINALIZADA COM SUCESSO! ")

if __name__ == "__main__":
    rodar_esteira()