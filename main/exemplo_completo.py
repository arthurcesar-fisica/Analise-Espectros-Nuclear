from espectrotool import (gera_espectro, mostra_espectro, salva_espectro,
                         analise_completa_espectro, detectar_picos, visualizar_deteccao,
                         obter_sinal_reconstruido, visualizar_ajuste, exportar_resultados)

from espectrotool.analise_metricas import quick_validation, QualityMetrics
from espectrotool.formato_picos import picos_info_para_array

# Picos a simular (listar em ordem crescente de 'centro')
PARAMETROS_PICOS = [
    {'amp': 400, 'centro': 250, 'sigma': 10},
    {'amp': 400, 'centro': 350, 'sigma': 10},
    {'amp': 200, 'centro': 450, 'sigma': 10}
]

# =============================================================================
# 0. Geração do espectro
# =============================================================================

print("=== GERANDO ESPECTRO ===")
eixo_energia, dados_espectro, picos_info, magnitude_ruido = gera_espectro(PARAMETROS_PICOS = PARAMETROS_PICOS)

salva_espectro(eixo_energia, dados_espectro, 'espectro_exemplo')

# Mostrar espectro original
mostra_espectro(eixo_energia, dados_espectro)


# =============================================================================
# 1. Detecção de picos
# =============================================================================

# Parâmetros customizados para detecção (opcional)
PARAMETROS_DETECCAO = {
    'altura_minima': 100,
    'distancia_minima': 5,
    'proeminencia': 40,
    'largura_minima': 2,
    'suavizar': True
}

print("\n1. DETECÇÃO DE PICOS")
picos_info = detectar_picos(dados_espectro, **PARAMETROS_DETECCAO)

if len(picos_info['indices']) == 0:
    print("Nenhum pico detectado. Análise interrompida.")

# Visualiza detecção
visualizar_deteccao(eixo_energia, dados_espectro, picos_info, 
                    salvar_grafico="deteccao_picos.png")


# =============================================================================
# 2. Reconstrução do espectro
# =============================================================================

# Parâmetros customizados para ajuste (opcional)
PARAMETROS_AJUSTE = {
    'tipo_pico': 'gaussiana',
    'tipo_fundo': 'exponencial'
}

print("\n2. RECONSTRUÇÃO DO ESPECTRO")
sinal_reconstruido, resultado_ajuste = obter_sinal_reconstruido(
    eixo_energia=eixo_energia,
    espectro=dados_espectro,
    picos_info=picos_info,
    **PARAMETROS_AJUSTE
)


# =============================================================================
# 3. Visualização dos resultados
# =============================================================================

print("\n3. VISUALIZAÇÃO DOS RESULTADOS")
visualizar_ajuste(eixo_energia, dados_espectro, picos_info, resultado_ajuste,
                    mostrar_componentes=True, 
                    salvar_grafico="reconstrucao_espectro.png")


# =============================================================================
# 4. Exportação dos resultados
# =============================================================================

print("\n4. EXPORTAÇÃO DOS RESULTADOS")
exportar_resultados(eixo_energia, dados_espectro, picos_info, resultado_ajuste)

print("ANÁLISE CONCLUÍDA COM SUCESSO!")

# =============================================================================
# 5. Validação com métricas
# =============================================================================

print("\n5. EXPORTAÇÃO DOS RESULTADOS")

resultados = {
    'picos_info': picos_info,
    'resultado_ajuste': resultado_ajuste,
    'sinal_reconstruido': sinal_reconstruido 
}

validator = QualityMetrics(
    tolerance_center=0.02,      # 2%
    tolerance_amplitude=0.10,   # 10%
    tolerance_sigma=0.15        # 15%
)

#Arrays com parâmetros dos picos
params_verdadeiros = picos_info_para_array(picos_info)
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
    noise_level=magnitude_ruido
)

# Relatório
validator.generate_report(analise_metricas)

print("\n=== PROCESSO CONCLUÍDO ===")