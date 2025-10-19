# main.py
from espectrotool import (gera_espectro, mostra_espectro, salva_espectro,
                         detectar_picos, ajustar_picos_global, 
                         ajustar_picos_individual, visualizar_ajuste)

# Gerar espectro de exemplo
print("=== GERANDO ESPECTRO ===")
NCanais = 1000
FundoAmp = 500
FundoDecai = 0.05
ParametrosPicos = [
    {'amp': 300, 'centro': 250, 'sigma': 10},
    {'amp': 200, 'centro': 600, 'sigma': 15},
    {'amp': 150, 'centro': 750, 'sigma': 8}
]

# Gerar espectro recebendo ambos eixo_energia e dados
eixo_energia, dados_espectro = gera_espectro(N_CANAIS=NCanais, FUNDO_AMP=FundoAmp, FUNDO_DECAI=FundoDecai, PARAMETROS_PICOS=ParametrosPicos)

# Mostrar o espectro
mostra_espectro(eixo_energia, dados_espectro)

# Salvar o espectro
salva_espectro(eixo_energia, dados_espectro, nome_arquivo="espectro_personalizado")

# Detectar picos automaticamente
print("\n=== DETECTANDO PICOS ===")
picos_info = detectar_picos(dados_espectro, altura_minima=100, distancia_minima=50, 
                           proeminencia=50, largura_minima=5)


# Tentar ajuste global primeiro
print("\n=== TENTANDO AJUSTE GLOBAL ===")
resultado_global = ajustar_picos_global(eixo_energia, dados_espectro, picos_info, 
                                      tipo_pico='gaussiana', grau_fundo=1)


# Fallback para ajuste individual se o global falhar
if not resultado_global['sucesso']:
    print("\n=== FALLBACK PARA AJUSTE INDIVIDUAL ===")
    resultado_ajuste = ajustar_picos_individual(eixo_energia, dados_espectro, picos_info,
                                              tipo_pico='gaussiana')
else:
    resultado_ajuste = resultado_global

# Visualizar resultados
print("\n=== VISUALIZANDO RESULTADOS ===")
visualizar_ajuste(eixo_energia, dados_espectro, picos_info, resultado_ajuste,
                 salvar_grafico="analise_espectro.png")

# Salvar dados
salva_espectro(eixo_energia, dados_espectro)

print("\n=== ANÁLISE CONCLUÍDA ===")