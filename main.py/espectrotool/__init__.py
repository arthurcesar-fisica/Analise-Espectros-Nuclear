# __init__.py
from .gerador_espectros import gera_espectro, mostra_espectro, salva_espectro  # Importa as funções do módulo

# Define o que será exportado quando alguém importar o pacote
__all__ = [
    'gera_espectro',
    'mostra_espectro',
    'salva_espectro'
]