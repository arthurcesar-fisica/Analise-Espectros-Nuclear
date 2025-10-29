# main.py
from espectrotool import (gera_espectro, mostra_espectro, salva_espectro,
                         analise_completa_espectro)

from espectrotool.analise_metricas import quick_validation, QualityMetrics

from espectrotool.formato_picos import picos_info_para_array

#Determinar parâmetros dos picos gerados
parametros_picos = [
            {'amp': 400, 'centro': 250, 'sigma': 10},
            {'amp': 200, 'centro': 550, 'sigma': 10}
]

# Gerar espectro de exemplo
print("=== GERANDO ESPECTRO ===")
eixo_energia, dados_espectro, picos_info = gera_espectro(PARAMETROS_PICOS = parametros_picos)

print(picos_info)

salva_espectro(eixo_energia, dados_espectro, 'espectro_exemplo')

# Mostrar espectro original
mostra_espectro(eixo_energia, dados_espectro)

# Parâmetros customizados para detecção (opcional)
parametros_deteccao = {
    'altura_minima': 100,
    'distancia_minima': 25,  # Reduzido para detectar picos próximos
    'proeminencia': 40,
    'largura_minima': 2,
    'suavizar': True
}

# Parâmetros customizados para ajuste (opcional)
parametros_ajuste = {
    'tipo_pico': 'gaussiana',
    'tipo_fundo': 'exponencial',
    'tratar_picos_proximos': True  # Ativa o tratamento especial para picos próximos
}

# Executar análise completa
resultados = analise_completa_espectro(
    eixo_energia, 
    dados_espectro,
    parametros_deteccao=parametros_deteccao,
    parametros_ajuste=parametros_ajuste
)


# 5. Valida com métricas
validator = QualityMetrics()
results = validator.check_requirements(
    y_true=dados_espectro,
    y_pred=resultados["resultado_ajuste"]["y_ajustado"],
    true_params=picos_info_para_array(picos_info),
    fitted_params=picos_info_para_array(resultados["picos_info"]),
    noise_level= 10  # Estima ruído
)
validator.generate_report(results)

print("\n=== PROCESSO CONCLUÍDO ===")