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
# CONFIGURAÇÃO
# =============================================================================

# Picos a simular (listar em ordem crescente de 'centro')
PARAMETROS_PICOS = [
    {'amp': 400, 'centro': 250, 'sigma': 10},
    {'amp': 400, 'centro': 350, 'sigma': 10},
    {'amp': 200, 'centro': 450, 'sigma': 10}
]

# Parâmetros de detecção (ajuste se necessário)
PARAMETROS_DETECCAO = {
    'altura_minima': 100,
    'distancia_minima': 25,
    'proeminencia': 40,
    'largura_minima': 2,
    'suavizar': True
}

# Parâmetros de ajuste
PARAMETROS_AJUSTE = {
    'tipo_pico': 'gaussiana',
    'tipo_fundo': 'exponencial'
}


# =============================================================================
# EXECUÇÃO
# =============================================================================

# 1. Gerar espectro sintético
print("\n[1/3] Gerando espectro...")

#Gera o espectro e estima o nível de ruído
eixo_energia, dados_espectro, picos_verdadeiros, nivel_ruido = gera_espectro(PARAMETROS_PICOS=PARAMETROS_PICOS)

salva_espectro(eixo_energia, dados_espectro, 'espectro_exemplo')
mostra_espectro(eixo_energia, dados_espectro)

print(f"   ✓ Espectro: {len(eixo_energia)} canais, {len(PARAMETROS_PICOS)} picos")


# 2. Detectar picos e reconstruir sinal
print("\n[2/3] Analisando espectro...")

resultados = analise_completa_espectro(
    eixo_energia, 
    dados_espectro,
    parametros_deteccao=PARAMETROS_DETECCAO,
    parametros_ajuste=PARAMETROS_AJUSTE
)
print(f"   ✓ Detectados: {len(resultados['picos_info']['indices'])} picos")


# 3. Validar métricas de qualidade
print("\n[3/3] Validando métricas...")

validator = QualityMetrics(
    tolerance_center=0.02,      # 2%
    tolerance_amplitude=0.10,   # 10%
    tolerance_sigma=0.15        # 15%
)

#Arrays com parâmetros dos picos
params_verdadeiros = picos_info_para_array(picos_verdadeiros)
params_detectados = picos_info_para_array(resultados['picos_info'])

#Futura implementação: Emparelhar os Picos
"""
params_v_matched, params_d_matched = validator._match_peaks_by_center(
    params_verdadeiros, params_detectados
)
"""

# Validar
analise_metricas = validator.check_requirements(
    y_true=dados_espectro,
    y_pred=resultados['sinal_reconstruido'],
    true_params=params_verdadeiros,
    fitted_params=params_detectados,
    noise_level=nivel_ruido
)

# Relatório
validator.generate_report(analise_metricas)