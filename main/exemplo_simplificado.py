from espectrotool import (gera_espectro, mostra_espectro, salva_espectro,
                         analise_completa_espectro, detectar_picos, visualizar_deteccao,
                         obter_sinal_reconstruido, visualizar_ajuste, exportar_resultados)

from espectrotool.analise_metricas import quick_validation, QualityMetrics
from espectrotool.formato_picos import picos_info_para_array

import numpy as np

#Determinar parâmetros dos picos gerados
#OBS: Limitação!! Por enquanto, os picos tem que ser ordenados em ordem crescente de centro!
parametros_picos = [
            {'amp': 400, 'centro': 250, 'sigma': 10},
            {'amp': 200, 'centro': 550, 'sigma': 10}
]


# =============================================================================
# 0. Geração do espectro
# =============================================================================

print("=== GERANDO ESPECTRO ===")
eixo_energia, dados_espectro, picos_info, magnitude_ruido = gera_espectro(PARAMETROS_PICOS = parametros_picos)

salva_espectro(eixo_energia, dados_espectro, 'espectro_exemplo')

# Mostrar espectro original
mostra_espectro(eixo_energia, dados_espectro)


# =============================================================================
# 1. Detecção e reconstrução de picos
# =============================================================================

print("=== INICIANDO RECONSTRUÇÃO ===")
resultados = analise_completa_espectro(
    eixo_energia, 
    dados_espectro
)

# =============================================================================
# 2. Visualização dos resultados
# =============================================================================

validator = QualityMetrics()
results = validator.check_requirements(
    y_true=dados_espectro,
    y_pred=resultados["resultado_ajuste"]["y_ajustado"],
    true_params=picos_info_para_array(picos_info),
    fitted_params=picos_info_para_array(resultados["picos_info"]),
    noise_level= magnitude_ruido
)
validator.generate_report(results)

print("\n=== PROCESSO CONCLUÍDO ===")