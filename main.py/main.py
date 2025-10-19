# main.py
from espectrotool import gera_espectro, mostra_espectro, salva_espectro

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