# __init__.py
from .gerador_espectros import gera_espectro, mostra_espectro, salva_espectro
from .analise_picos import detectar_picos, ajustar_picos_global, ajustar_picos_individual, visualizar_ajuste

__all__ = [
    'gera_espectro',
    'mostra_espectro',
    'salva_espectro',
    'detectar_picos',
    'ajustar_picos_global',
    'ajustar_picos_individual',
    'visualizar_ajuste'
]