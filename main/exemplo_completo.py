from espectrotool import (gera_espectro, mostra_espectro, salva_espectro,
                         analise_completa_espectro, detectar_picos, visualizar_deteccao,
                         obter_sinal_reconstruido, visualizar_ajuste, exportar_resultados)

from espectrotool.analise_metricas import quick_validation, QualityMetrics
from espectrotool.formato_picos import picos_info_para_array

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
# 1. Detecção de picos
# =============================================================================

# Parâmetros customizados para detecção (opcional)
parametros_deteccao = {
    'altura_minima': 100,
    'distancia_minima': 25,  # Reduzido para detectar picos próximos
    'proeminencia': 40,
    'largura_minima': 2,
    'suavizar': True
}

print("\n1. DETECÇÃO DE PICOS")
picos_info = detectar_picos(dados_espectro, **parametros_deteccao)

if len(picos_info['indices']) == 0:
    print("Nenhum pico detectado. Análise interrompida.")

# Visualiza detecção
visualizar_deteccao(eixo_energia, dados_espectro, picos_info, 
                    salvar_grafico="deteccao_picos.png")


# =============================================================================
# 2. Reconstrução do espectro
# =============================================================================

# Parâmetros customizados para ajuste (opcional)
parametros_ajuste = {
    'tipo_pico': 'gaussiana',
    'tipo_fundo': 'exponencial'
}

print("\n2. RECONSTRUÇÃO DO ESPECTRO")
sinal_reconstruido, resultado_ajuste = obter_sinal_reconstruido(
    eixo_energia=eixo_energia,
    espectro=dados_espectro,
    picos_info=picos_info,
    **parametros_ajuste
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

resultados = {
    'picos_info': picos_info,
    'resultado_ajuste': resultado_ajuste,
    'sinal_reconstruido': sinal_reconstruido 
}

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