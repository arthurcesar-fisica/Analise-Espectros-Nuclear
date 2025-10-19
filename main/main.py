# main.py
from espectrotool import (gera_espectro, mostra_espectro, salva_espectro,
                         analise_completa_espectro)

# Gerar espectro de exemplo
print("=== GERANDO ESPECTRO ===")
eixo_energia, dados_espectro = gera_espectro()

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

print("\n=== PROCESSO CONCLUÍDO ===")