# main.py

"""
===============================================================================
EXEMPLO BÁSICO: Análise de Espectros Nucleares
===============================================================================
Este exemplo demonstra o uso da biblioteca a partir de dados sintéticos. Ele realiza os seguintes passos:

  1. Gerar espectro sintético
  2. Detectar e ajustar picos
  3. Validar métricas de qualidade
===============================================================================
"""

from espectrotool import gera_espectro, mostra_espectro, salva_espectro, analise_completa_espectro
from espectrotool.analise_metricas import QualityMetrics
from espectrotool.formato_picos import picos_info_para_array


# =============================================================================
# EXECUÇÃO
# =============================================================================

# 1. Gerar espectro sintético
eixo_energia, dados_espectro, picos_verdadeiros, nivel_ruido = gera_espectro()

salva_espectro(eixo_energia, dados_espectro, 'espectro_exemplo')
mostra_espectro(eixo_energia, dados_espectro)

# 2. Detectar picos e reconstruir sinal
resultados = analise_completa_espectro(eixo_energia, dados_espectro)


# 3. Validar métricas de qualidade
validator = QualityMetrics(
    tolerance_center=0.02,      # 2%
    tolerance_amplitude=0.10,   # 10%
    tolerance_sigma=0.15        # 15%
)

#Arrays com parâmetros dos picos
params_verdadeiros = picos_info_para_array(picos_verdadeiros)
params_detectados = picos_info_para_array(resultados['picos_info'])

# Validar análise
analise_metricas = validator.check_requirements(
    y_true=dados_espectro,
    y_pred=resultados['sinal_reconstruido'],
    true_params=params_verdadeiros,
    fitted_params=params_detectados,
    noise_level=nivel_ruido
)

# Relatório
validator.generate_report(analise_metricas)