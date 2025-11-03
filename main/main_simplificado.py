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

# Parâmetros customizados para detecção (opcional)
parametros_deteccao = {
    'altura_minima': 100,
    'distancia_minima': 25,
    'proeminencia': 40,
    'largura_minima': 2,
    'suavizar': True
}

# Parâmetros customizados para ajuste (opcional)
parametros_ajuste = {
    'tipo_pico': 'gaussiana',
    'tipo_fundo': 'exponencial'
}

# Executar análise completa
resultados = analise_completa_espectro(
    eixo_energia, 
    dados_espectro,
    parametros_deteccao=parametros_deteccao,
    parametros_ajuste=parametros_ajuste
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

'''
print(picos_info_para_array(picos_info))
print(picos_info_para_array(resultados["picos_info"]))

# Calcula resíduos
residuos = dados_espectro - resultados["resultado_ajuste"]["y_ajustado"]

visualizar_resultados_completo(
        results=results,
        residuos=residuos,
        eixo_energia=eixo_energia,
        salvar_dashboard="dashboard_validacao.png",
        salvar_residuos="analise_residuos_detalhada.png",
        mostrar=True
)
'''

print("\n=== PROCESSO CONCLUÍDO ===")