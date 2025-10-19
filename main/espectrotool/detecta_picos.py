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
        'indices': picos,
        'alturas': propriedades['peak_heights'],
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

def visualizar_deteccao(eixo_energia, espectro, picos_info, salvar_grafico=None):
    """
    Visualiza a detecção de picos.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Espectro original
    ax.plot(eixo_energia, espectro, 'k-', alpha=0.7, linewidth=1, 
            label='Espectro Original', drawstyle='steps-mid')
    
    # Espectro suavizado (se aplicado)
    if picos_info['espectro_suavizado'] is not None:
        ax.plot(eixo_energia, picos_info['espectro_suavizado'], 'b-', alpha=0.5, 
                linewidth=1, label='Espectro Suavizado')
    
    # Picos detectados
    ax.plot(eixo_energia[picos_info['indices']], espectro[picos_info['indices']], 
            'ro', markersize=8, label=f'Picos Detectados ({len(picos_info["indices"])})')
    
    # Linhas verticais nos picos
    for i, pico in enumerate(picos_info['indices']):
        ax.axvline(x=eixo_energia[pico], color='r', linestyle='--', alpha=0.5)
        ax.text(eixo_energia[pico], espectro[pico] * 1.05, f'{i+1}', 
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Energia (Canal)')
    ax.set_ylabel('Contagens')
    ax.set_title('Detecção Automática de Picos')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if salvar_grafico:
        pasta_analises = "analises_espectros"
        os.makedirs(pasta_analises, exist_ok=True)
        caminho_completo = os.path.join(pasta_analises, salvar_grafico)
        plt.savefig(caminho_completo, dpi=300, bbox_inches='tight')
        print(f"Gráfico de detecção salvo como {caminho_completo}")
    
    plt.show()
    
    return fig, ax