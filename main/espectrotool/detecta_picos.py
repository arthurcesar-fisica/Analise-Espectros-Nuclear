# detecta_picos.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths, savgol_filter
import os

def detectar_picos(espectro, altura_minima=None, distancia_minima=20, 
                  proeminencia=None, largura_minima=3, relacao_sinal_ruido=2.0,
                  suavizar=True, tamanho_janela=11, ordem_polinomio=3):
    """
    Detecta picos automaticamente no espectro usando análise de proeminência e largura.
    
    Parâmetros:
    -----------
    espectro : array
        Dados do espectro (contagens)
    suavizar : bool
        Se True, aplica filtro Savitzky-Golay para suavizar o espectro antes da detecção
    tamanho_janela : int
        Tamanho da janela para suavização (deve ser ímpar)
    ordem_polinomio : int
        Ordem do polinômio para suavização
    """
    
    # Aplica suavização se solicitado
    if suavizar and len(espectro) > tamanho_janela:
        espectro_suavizado = savgol_filter(espectro, tamanho_janela, ordem_polinomio)
    else:
        espectro_suavizado = espectro.copy()
    
    # Estima parâmetros automáticos se não fornecidos
    if altura_minima is None:
        altura_minima = np.median(espectro_suavizado) * 1.5
    
    if proeminencia is None:
        proeminencia = np.std(espectro_suavizado) * relacao_sinal_ruido
    
    # Detecta picos no espectro suavizado
    picos, propriedades = find_peaks(
        espectro_suavizado, 
        height=altura_minima,
        distance=distancia_minima,
        prominence=proeminencia,
        width=largura_minima
    )
    
    # Calcula larguras dos picos no espectro original (não suavizado)
    larguras_resultados = peak_widths(espectro, picos, rel_height=0.5)
    
    # Organiza informações dos picos
    picos_info = {
        'n_picos': len(picos),
        'indices': picos,
        'centros': picos,
        'alturas': propriedades['peak_heights'],
        'sigmas': larguras_resultados[0] / (2 * np.sqrt(2 * np.log(2))),
        'proeminencias': propriedades['prominences'],
        'larguras': larguras_resultados[0],
        'posicoes_larguras': larguras_resultados[2:4],
        'limites_esquerda': propriedades['left_bases'],
        'limites_direita': propriedades['right_bases'],
        'espectro_suavizado': espectro_suavizado if suavizar else None
    }
    
    print(f"Detectados {len(picos)} picos:")
    for i, (pico, altura, prominencia_pico, largura) in enumerate(zip(
        picos_info['indices'], 
        picos_info['alturas'], 
        picos_info['proeminencias'], 
        picos_info['larguras']
    )):
        print(f"  Pico {i+1}: Canal {pico}, Altura {altura:.1f}, "
              f"Proeminência {prominencia_pico:.1f}, Largura {largura:.1f} canais")
    
    return picos_info