# deteccao_picos_ml.py

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from carrega_dados import carrega_espectro

def extrai_features(sinal, indice, janela=10):
    # Analisa cada ponto com relação à vizinhança
    inicio = max(0, indice - janela)
    fim = min(len(sinal), indice + janela + 1)
    regiao = sinal[inicio:fim]
    
    return [
        sinal[indice],
        np.mean(regiao),
        np.std(regiao),
        sinal[indice] / (np.mean(regiao) + 1e-10),
        np.max(regiao) - np.min(regiao)
    ]

def treina_detector_ruido(n_samples=10000, mean=10, std=10):
    # Gera dados de ruído puro
    ruido = np.random.normal(mean, std, size=n_samples)
    
    # Análise de vizinhança
    X_ruido = []
    for i in range(100, n_samples-100):  # Evita bordas
        X_ruido.append(extrai_features(ruido, i, janela=10))
    
    X = np.array(X_ruido)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # IsolationForest detecta anomalias (outliers) no sinal, ou seja, detecta o porções diferentes de ruído (região de interesse para fit gaussiano)
    modelo = IsolationForest(
        contamination=0.1,
        random_state=42,
        n_estimators=100
    )
    modelo.fit(X_scaled)
    
    print(f"✓ Modelo treinado com {len(X)} amostras de ruído puro")
    print(f"  Parâmetros do ruído: μ={mean}, σ={std}")
    
    return modelo, scaler

def detecta_regioes_nao_ruido(energia, contagens, modelo, scaler, threshold=-0.5):
    # Análise de vizinhança
    X = []
    indices_validos = []
    for i in range(0, len(contagens)-10):
        X.append(extrai_features(contagens, i, janela=10))
        indices_validos.append(i)
    
    X = np.array(X)
    indices_validos = np.array(indices_validos)
    
    # Score: valores negativos = outliers = NÃO é ruído = possível pico
    X_scaled = scaler.transform(X)
    scores = modelo.score_samples(X_scaled)
    
    outliers = indices_validos[scores < threshold]
    
    print(f"✓ Detectados {len(outliers)} pontos que NÃO são ruído")
    
    return outliers, scores

def gaussiana(x, amp, mu, sigma):
    return amp * np.exp(-((x - mu)**2) / (2 * sigma**2))

def fit_gaussianas(energia, contagens, outliers, janela_fit=20, threshold_amplitude=50):
    # Agrupa outliers próximos em regiões
    if len(outliers) == 0:
        return []
    
    regioes = []
    regiao_atual = [outliers[0]]
    
    for i in range(1, len(outliers)):
        if outliers[i] - outliers[i-1] <= 3:  # Pontos próximos (mesma região)
            regiao_atual.append(outliers[i])
        else:
            if len(regiao_atual) >= 5:  # Região válida (mínimo 5 pontos)
                regioes.append(regiao_atual)
            regiao_atual = [outliers[i]]
    
    if len(regiao_atual) >= 5:
        regioes.append(regiao_atual)
    
    print(f"✓ Identificadas {len(regioes)} regiões de picos")
    
    # Fit gaussiano em cada região
    resultados = []
    picos_rejeitados = 0
    
    for idx, regiao in enumerate(regioes):
        centro_idx = regiao[len(regiao)//2]
        inicio = max(0, centro_idx - janela_fit)
        fim = min(len(energia), centro_idx + janela_fit)
        
        x_fit = energia[inicio:fim]
        y_fit = contagens[inicio:fim]
        
        # Estimativa inicial
        amp_inicial = np.max(y_fit)
        mu_inicial = x_fit[np.argmax(y_fit)]
        sigma_inicial = 5.0
        
        try:
            # Fit da gaussiana
            popt, pcov = curve_fit(
                gaussiana, x_fit, y_fit,
                p0=[amp_inicial, mu_inicial, sigma_inicial],
                bounds=([0, x_fit[0], 0.1], [np.inf, x_fit[-1], 50]),
                maxfev=5000
            )
            
            amp, mu, sigma = popt
            erro_mu = np.sqrt(pcov[1, 1])
            erro_sigma = np.sqrt(pcov[2, 2])
            
            # FILTRO: Verifica se amplitude está acima do threshold
            if amp < threshold_amplitude:
                print(f"  ✗ Pico {idx+1} rejeitado: amplitude {amp:.2f} < threshold {threshold_amplitude}")
                picos_rejeitados += 1
                continue
            
            resultados.append({
                'pico_id': len(resultados) + 1,
                'amplitude': amp,
                'media': mu,
                'desvio_padrao': sigma,
                'erro_media': erro_mu,
                'erro_sigma': erro_sigma,
                'x_fit': x_fit,
                'y_fit': y_fit,
                'y_gaussiana': gaussiana(x_fit, amp, mu, sigma)
            })
            
            print(f"  ✓ Pico {len(resultados)}: μ = {mu:.2f} ± {erro_mu:.2f}, σ = {sigma:.2f} ± {erro_sigma:.2f}, amp = {amp:.2f}")
            
        except Exception as e:
            print(f"  ✗ Erro no fit do pico {idx+1}: {e}")
    
    print(f"✓ {len(resultados)} picos aceitos, {picos_rejeitados} rejeitados pelo threshold de amplitude")
    
    return resultados

# Carrega dados usando a função do carrega_dados.py
caminho_arquivo = input('Caminho do arquivo (.xlsx ou .json): ')
energia, contagens = carrega_espectro(caminho_arquivo)

# Adiciona mais ruído
contagens = contagens + np.random.normal(10, 10, size=len(contagens))

# 1. Treina modelo apenas com ruído
modelo, scaler = treina_detector_ruido(n_samples=10000, mean=10, std=10)

# 2. Detecta regiões que NÃO são ruído
outliers, scores = detecta_regioes_nao_ruido(energia, contagens, modelo, scaler, threshold=-0.5)

# 3. Fit de gaussianas nos picos (com threshold de amplitude mínima)
THRESHOLD_AMPLITUDE = int(input('Limite mínimo para gaussiana: '))
resultados_fit = fit_gaussianas(energia, contagens, outliers, janela_fit=25, threshold_amplitude=THRESHOLD_AMPLITUDE)

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Detecção de outliers
ax1 = axes[0]
ax1.plot(energia, contagens, 'b-', alpha=0.5, label='Espectro + Ruído')
ax1.plot(energia[outliers], contagens[outliers], 'ro', markersize=3, 
         alpha=0.5, label=f'{len(outliers)} pontos outliers')
ax1.set_xlabel('Energia')
ax1.set_ylabel('Contagens')
ax1.set_title('Detecção de regiões que NÃO são ruído (Isolation Forest)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Fits gaussianos
ax2 = axes[1]
ax2.plot(energia, contagens, 'b-', alpha=0.5, label='Espectro + Ruído')

cores = plt.cm.rainbow(np.linspace(0, 1, len(resultados_fit)))
for i, resultado in enumerate(resultados_fit):
    ax2.plot(resultado['x_fit'], resultado['y_gaussiana'], '--', 
             color=cores[i], linewidth=2,
             label=f"Pico {resultado['pico_id']}: μ={resultado['media']:.1f}, σ={resultado['desvio_padrao']:.1f}")

ax2.set_xlabel('Energia')
ax2.set_ylabel('Contagens')
ax2.set_title(f'Fit gaussiano dos {len(resultados_fit)} picos detectados')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Tabela de resultados
print(f"{'Resumo dos picos detectados':^5}")
print(f"{'ID':<5} {'Amplitude':<12} {'Média (μ)':<15} {'Desvio (σ)':<15}")
for r in resultados_fit:
    print(f"{r['pico_id']:<5} {r['amplitude']:<12.2f} "
          f"{r['media']:.2f} ± {r['erro_media']:.2f}    "
          f"{r['desvio_padrao']:.2f} ± {r['erro_sigma']:.2f}")