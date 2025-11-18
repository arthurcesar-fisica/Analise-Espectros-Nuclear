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

def fit_gaussianas(df, outliers, coluna_x='Energia', coluna_y='Contagens',
                   janela_fit=30, threshold_amplitude=50, distancia_minima=10,
                   tolerancia_mu=15):
    
    energia = df[coluna_x].values
    contagens = df[coluna_y].values
    
    if len(outliers) == 0:
        return []
    
    # Agrupa outliers próximos em regiões
    regioes = []
    regiao_atual = [outliers[0]]
    
    for i in range(1, len(outliers)):
        if outliers[i] - outliers[i-1] <= 3:
            regiao_atual.append(outliers[i])
        else:
            if len(regiao_atual) >= 5:
                regioes.append(regiao_atual)
            regiao_atual = [outliers[i]]
    
    if len(regiao_atual) >= 5:
        regioes.append(regiao_atual)
    
    print(f'✓ Identificadas {len(regioes)} regiões de picos')
    
    resultados_temporarios = []
    
    for regiao in regioes:
        # Define janela estendida para buscar múltiplos picos
        inicio = max(0, regiao[0] - janela_fit)
        fim = min(len(energia), regiao[-1] + janela_fit)
        
        indices_regiao = list(range(inicio, fim))
        x_regiao = energia[inicio:fim]
        y_regiao = contagens[inicio:fim]
        
        # Encontra todos os máximos locais na região
        picos_locais = []
        for i in range(1, len(y_regiao)-1):
            # É máximo local se for maior que os vizinhos
            if y_regiao[i] > y_regiao[i-1] and y_regiao[i] > y_regiao[i+1]:
                # Filtra por amplitude mínima
                if y_regiao[i] > threshold_amplitude:
                    picos_locais.append(i)
        
        # Remove picos muito próximos (mantém o maior)
        picos_filtrados = []
        picos_usados = set()
        
        for pico_idx in picos_locais:
            if pico_idx in picos_usados:
                continue
            
            # Verifica se há picos próximos
            grupo = [pico_idx]
            for outro_idx in picos_locais:
                if outro_idx != pico_idx and abs(outro_idx - pico_idx) < distancia_minima:
                    grupo.append(outro_idx)
            
            # Escolhe o maior do grupo
            idx_max = max(grupo, key=lambda idx: y_regiao[idx])
            picos_filtrados.append(idx_max)
            picos_usados.update(grupo)
        
        print(f'  Região: {len(picos_locais)} máximos locais → {len(picos_filtrados)} após filtro de distância')
        
        # Ajusta gaussiana para cada pico detectado
        for pico_idx in picos_filtrados:
            centro_idx = inicio + pico_idx
            inicio_fit = max(0, centro_idx - janela_fit)
            fim_fit = min(len(energia), centro_idx + janela_fit)
            
            x_fit = energia[inicio_fit:fim_fit]
            y_fit = contagens[inicio_fit:fim_fit]
            
            # Estimativas iniciais
            amp_inicial = contagens[centro_idx]
            mu_inicial = energia[centro_idx]
            sigma_inicial = 5.0
            
            try:
                popt, pcov = curve_fit(
                    gaussiana, x_fit, y_fit,
                    p0=[amp_inicial, mu_inicial, sigma_inicial],
                    bounds=([0, x_fit[0], 0.1], [np.inf, x_fit[-1], 50]),
                    maxfev=5000
                )
                
                amp, mu, sigma = popt
                erro_mu = np.sqrt(pcov[1, 1])
                erro_sigma = np.sqrt(pcov[2, 2])
                
                resultados_temporarios.append({
                    'amplitude': amp,
                    'media': mu,
                    'desvio_padrao': sigma,
                    'erro_media': erro_mu,
                    'erro_sigma': erro_sigma,
                    'x_fit': x_fit,
                    'y_fit': y_fit,
                    'y_gaussiana': gaussiana(x_fit, amp, mu, sigma)
                })
                
            except Exception as e:
                print(f'  ✗ Erro no fit: {e}')
    
    # Remove duplicatas baseadas em média e desvio padrão próximos
    resultados_finais = []
    
    for candidato in resultados_temporarios:
        eh_duplicata = False

        for existente in resultados_finais:
            # Verifica se média e desvio são próximos
            diff_mu = abs(candidato['media'] - existente['media'])
            
            if diff_mu < tolerancia_mu:
                eh_duplicata = True
                # Mantém o de maior amplitude
                if candidato['amplitude'] > existente['amplitude']:
                    # Substitui o existente pelo candidato
                    idx = resultados_finais.index(existente)
                    resultados_finais[idx] = candidato
                    print(f"  ↻ Substituído: μ={existente['media'] :.2f} por μ={candidato['media']:.2f} (maior amplitude)")
                else:
                    print(f"  ✗ Duplicata descartada: μ={candidato['media']:.2f} (menor amplitude)")
                break
        
        if not eh_duplicata:
            resultados_finais.append(candidato)
    
    # Adiciona IDs sequenciais aos resultados finais
    for i, resultado in enumerate(resultados_finais):
        resultado['pico_id'] = i + 1
        print(f"  ✓ Pico {resultado['pico_id']}: μ = {resultado['media']:.2f} ± {resultado['erro_media']:.2f}, "
              f"  = {resultado['desvio_padrao']:.2f} ± {resultado['erro_sigma']:.2f}, amp = {resultado['amplitude']:.2f}")
    
    print(f"✓ Total: {len(resultados_temporarios)} ajustes → {len(resultados_finais)} picos únicos após remoção de duplicatas")
    return resultados_finais
                       
def deteccao_picos_ml(caminho_pai, lista_caminhos_arquivos, THRESHOLD_AMPLITUDE=50, salvar_tabela=True, arquivo_saida='resultados_picos.xlsx'):
    for caminho in lista_caminhos_arquivos:        
        # Carrega dados
        df = pd.read_excel(f'{caminho_pai}{caminho}')
    
        # Adiciona mais ruído
        df['Contagens'] = df['Contagens'] + np.random.normal(10, 10, size=len(df))
    
        # 1. Treina modelo apenas com ruído
        modelo, scaler = treina_detector_ruido(n_samples=10000, mean=10, std=10)
    
        # 2. Detecta regiões que NÃO são ruído
        # CORREÇÃO: passar energia e contagens separadamente
        energia = df['Energia'].values
        contagens = df['Contagens'].values
        outliers, _ = detecta_regioes_nao_ruido(energia, contagens, modelo, scaler, threshold=-0.5)
    
        # 3. Fit de gaussianas nos picos (com threshold de amplitude mínima)
        resultados_fit = fit_gaussianas(df, outliers, threshold_amplitude=THRESHOLD_AMPLITUDE)
    
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
        # Plot 1: Detecção de outliers
        ax1 = axes[0]
        ax1.plot(energia, contagens, 'b-', alpha=0.5, label='Espectro + Ruído')
        ax1.plot(energia[outliers], contagens[outliers], 'ro', markersize=3, 
                alpha=0.5, label=f'{len(outliers)} pontos outliers')
        ax1.set_xlabel('Energia')
        ax1.set_ylabel('Contagens')
        ax1.set_title(f'Detecção de Regiões que NÃO são Ruído - {caminho}')
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
        ax2.set_title(f'Fit Gaussiano dos {len(resultados_fit)} Picos Detectados - {caminho}')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
    
        plt.tight_layout()
        plt.show()
    
        # Tabela de resultados
        print(f"\n{'RESUMO DOS PICOS DETECTADOS':^60}")
        print(f"{'ID':<5} {'Amplitude':<12} {'Média (μ)':<20} {'Desvio (σ)':<20}")
        print("-" * 60)
        for r in resultados_fit:
            print(f"{r['pico_id']:<5} {r['amplitude']:<12.2f} "
                f"{r['media']:.2f} ± {r['erro_media']:.2f}        "
                f"{r['desvio_padrao']:.2f} ± {r['erro_sigma']:.2f}")
        
        if salvar_tabela:
            dados_tabela = []
            for r in resultados_fit:
                dados_tabela.append({
                    'Arquivo': caminho,
                    'ID': r['pico_id'], 
                    'Amplitude': r['amplitude'], 
                    'Média (μ)': r['media'],
                    'Erro Média': r['erro_media'], 
                    'Desvio (σ)': r['desvio_padrao'], 
                    'Erro Desvio': r['erro_sigma']
                })
            
            df_resultados = pd.DataFrame(dados_tabela)
            
            # Salvar em modo append se já existir
            try:
                # Tenta ler o arquivo existente
                df_existente = pd.read_excel(arquivo_saida)
                df_final = pd.concat([df_existente, df_resultados], ignore_index=True)
                df_final.to_excel(arquivo_saida, index=False, sheet_name='Picos Detectados')
                print(f"\n✓ Resultados adicionados à tabela: {arquivo_saida}")
            except FileNotFoundError:
                # Se não existe, cria novo
                df_resultados.to_excel(arquivo_saida, index=False, sheet_name='Picos Detectados')
                print(f"\n✓ Tabela criada: {arquivo_saida}")


# Exemplo de uso
deteccao_picos_ml(
    'C:/Users/z004wshk/OneDrive - Siemens Energy/Documents/BI Project/Computação Científica/dados_simulados/', 
    ['espectro_1_poucos_picos.xlsx', 'espectro_2_muitos_picos.xlsx', 'espectro_3_picos_largos.xlsx', 'espectro_4_picos_estreitos.xlsx'], 
    THRESHOLD_AMPLITUDE=100, 
    salvar_tabela=True,
    arquivo_saida='resultados_picos_todos.xlsx'
)
