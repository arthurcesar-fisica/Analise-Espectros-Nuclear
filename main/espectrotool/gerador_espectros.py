# gerador_espectros.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def fundo_exponencial(eixo_energia, amp, decai):
    fundo_puro = amp * np.exp(-eixo_energia * decai)
    fundo_com_ruido = np.random.poisson(fundo_puro).astype(np.float64)
    return fundo_com_ruido

def gerar_pico_gaussiano(eixo_energia, amp, centro, sigma):
    return amp * np.exp(-((eixo_energia - centro)**2) / (2 * sigma**2))

def simular_espectro(eixo_energia, params_fundo, lista_picos):
    fundo = fundo_exponencial(eixo_energia, **params_fundo)
    espectro_final = fundo.astype(np.float64)

    for params_pico in lista_picos:
        pico = gerar_pico_gaussiano(eixo_energia, **params_pico)
        espectro_final = espectro_final + pico
    
    return espectro_final

def gera_espectro(N_CANAIS=1000,
                  FUNDO_AMP=500,
                  FUNDO_DECAI=0.05,
                  PARAMETROS_PICOS=None):
    
    if PARAMETROS_PICOS is None:
        NUM_PICOS = np.random.randint(2, 6)  # Número aleatório de picos entre 2 e 5
        PARAMETROS_PICOS = []
        for _ in range(NUM_PICOS):
            amp = np.random.uniform(100, 500)  # Amplitude entre 100 e 500
            centro = np.random.uniform(50, 950)  # Centro entre 50 e 950
            sigma = np.random.uniform(5, 20)  # Sigma entre 5 e 20
            PARAMETROS_PICOS.append({'amp': amp, 'centro': centro, 'sigma': sigma})

    EIXO_ENERGIA = np.linspace(0, 1000, N_CANAIS)
    np.random.seed(42)

    espectro_simulado = simular_espectro(
        EIXO_ENERGIA,
        {'amp': FUNDO_AMP, 'decai': FUNDO_DECAI},
        PARAMETROS_PICOS
    )

    return EIXO_ENERGIA, espectro_simulado

def mostra_espectro(eixo_energia, espectro_simulado):
    plt.figure(figsize=(12, 7))
    plt.plot(eixo_energia, espectro_simulado, drawstyle='steps-mid')
    plt.title('Espectro Simulado com Fundo e Múltiplos Picos')
    plt.xlabel('Canal de Energia')
    plt.ylabel('Contagens')
    plt.grid(True)
    plt.show()

def salva_espectro(eixo_energia, espectro_simulado, nome_arquivo="espectro_simulado"):
    # Criar a pasta 'dados_simulados' se não existir
    pasta_dados = "dados_simulados"
    os.makedirs(pasta_dados, exist_ok=True)

    # Caminho completo para os arquivos
    caminho_excel = os.path.join(pasta_dados, f"{nome_arquivo}.xlsx")
    caminho_json = os.path.join(pasta_dados, f"{nome_arquivo}.json")

    # Salvar os arquivos
    df_espectro = pd.DataFrame({
        'Energia': eixo_energia,
        'Contagens': espectro_simulado
    })

    df_espectro.to_excel(caminho_excel, index=False)
    df_espectro.to_json(caminho_json, orient='records')