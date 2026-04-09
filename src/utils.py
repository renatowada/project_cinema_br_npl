import re
from unidecode import unidecode

def gerar_chave(titulo: str, ano: int | str) -> str:
    """
    Gera uma chave primária padronizada (id_movie) baseada no título e ano do filme.
    Utilizado para fazer o merge seguro entre diferentes bases de dados.
    
    Argumentos:
        titulo (str): O título original do filme.
        ano (int | str): O ano de lançamento.
        
    Retorna:
        str: A chave formatada (ex: 'o_auto_da_compadecida_2000')
    """

    if not titulo or str(titulo).lower() in ['nan', 'none', 'null']:
        titulo = "sem_titulo"
        
    # Limpeza do título
    titulo_limpo = unidecode(str(titulo)).lower().strip()
    titulo_limpo = re.sub(r'[^a-z0-9\s]', '', titulo_limpo)
    titulo_limpo = re.sub(r'\s+', '_', titulo_limpo).strip('_')
    
    # Tratamento do ano
    try:
        ano_str = str(int(ano))
    except (ValueError, TypeError):
        ano_str = "0000"
        
    return f"{titulo_limpo}_{ano_str}"