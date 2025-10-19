# analise_picos.py

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
from scipy.special import wofz
import warnings

# =============================================================================
# FUNÇÕES DE MODELO PARA FUNDO E PICOS
# =============================================================================

def fundo_exponencial(x, amp, decaimento):
    """Função de fundo exponencial para ajuste."""
    return amp * np.exp(-x * decaimento)

def gaussiana(x, amp, centro, sigma):
    """Função gaussiana para ajuste de picos."""
    return amp * np.exp(-(x - centro)**2 / (2 * sigma**2))

def lorentziana(x, amp, centro, gamma):
    """Função lorentziana para ajuste de picos."""
    return amp * (gamma**2 / ((x - centro)**2 + gamma**2))

def voigt(x, amp, centro, sigma, gamma):
    """Função Voigt (convolução Gaussiana-Lorentziana)."""
    z = (x - centro + 1j*gamma) / (sigma * np.sqrt(2))
    return amp * np.real(wofz(z)) / (sigma * np.sqrt(2*np.pi))

# =============================================================================
# DETECÇÃO AUTOMÁTICA DE PICOS
# =============================================================================

def detectar_picos(espectro, altura_minima=None, distancia_minima=20, 
                  proeminencia=None, largura_minima=5, relacao_sinal_ruido=2.0):
    """
    Detecta picos automaticamente no espectro usando análise de proeminência e largura.
    """
    # Estima parâmetros automáticos se não fornecidos
    if altura_minima is None:
        altura_minima = np.median(espectro) * 1.5
    
    if proeminencia is None:
        proeminencia = np.std(espectro) * relacao_sinal_ruido
    
    # Detecta picos
    picos, propriedades = find_peaks(
        espectro, 
        height=altura_minima,
        distance=distancia_minima,
        prominence=proeminencia,
        width=largura_minima
    )
    
    # Calcula larguras dos picos
    larguras_resultados = peak_widths(espectro, picos, rel_height=0.5)
    
    # Organiza informações dos picos
    picos_info = {
        'indices': picos,
        'alturas': propriedades['peak_heights'],
        'proeminencias': propriedades['prominences'],
        'larguras': larguras_resultados[0],
        'posicoes_larguras': larguras_resultados[2:4],
        'limites_esquerda': propriedades['left_bases'],
        'limites_direita': propriedades['right_bases']
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

# =============================================================================
# AJUSTE NÃO-LINEAR GLOBAL COM FUNDO EXPONENCIAL
# =============================================================================

def ajustar_picos_global(eixo_energia, espectro, picos_info, 
                        tipo_pico='gaussiana', tipo_fundo='exponencial', metodo='lm'):
    """
    Ajuste não-linear global de todos os picos simultaneamente.
    """
    n_picos = len(picos_info['indices'])
    
    # Define função do pico
    if tipo_pico == 'gaussiana':
        funcao_pico = gaussiana
        parametros_por_pico = 3
    elif tipo_pico == 'lorentziana':
        funcao_pico = lorentziana
        parametros_por_pico = 3
    elif tipo_pico == 'voigt':
        funcao_pico = voigt
        parametros_por_pico = 4
    else:
        raise ValueError("Tipo de pico deve ser 'gaussiana', 'lorentziana' ou 'voigt'")
    
    # Define função do fundo
    if tipo_fundo == 'exponencial':
        funcao_fundo = fundo_exponencial
        parametros_fundo = 2
    else:
        raise ValueError("Tipo de fundo deve ser 'exponencial'")
    
    # Função modelo global
    def modelo_global(x, amp_fundo, decaimento_fundo, *params_picos):
        # Calcula fundo
        y = funcao_fundo(x, amp_fundo, decaimento_fundo)
        
        # Adiciona cada pico
        for i in range(n_picos):
            inicio = i * parametros_por_pico
            fim = inicio + parametros_por_pico
            params_pico = params_picos[inicio:fim]
            y += funcao_pico(x, *params_pico)
        
        return y
    
    # Chutes iniciais para o fundo exponencial
    # Estimativa robusta para fundo exponencial
    try:
        # Usa os primeiros e últimos pontos para estimar fundo
        x_data = eixo_energia
        y_data = espectro
        
        # Estimativa inicial para fundo exponencial
        amp_estimada = np.max(espectro) * 0.8
        decaimento_estimado = 0.05  # Valor padrão baseado na simulação
        
        # Ajuste rápido só do fundo para melhor estimativa
        try:
            mask_picos = np.ones_like(espectro, dtype=bool)
            for idx in picos_info['indices']:
                largura = picos_info['larguras'][np.where(picos_info['indices'] == idx)[0][0]]
                inicio = max(0, int(idx - largura * 2))
                fim = min(len(espectro), int(idx + largura * 2))
                mask_picos[inicio:fim] = False
            
            if np.sum(mask_picos) > 10:  # Precisa ter pontos suficientes
                popt_fundo, _ = curve_fit(fundo_exponencial, x_data[mask_picos], y_data[mask_picos],
                                        p0=[amp_estimada, decaimento_estimado], maxfev=1000)
                amp_estimada, decaimento_estimado = popt_fundo
        except:
            pass  # Mantém as estimativas iniciais se o ajuste falhar
            
    except:
        amp_estimada = np.median(espectro) * 2
        decaimento_estimado = 0.05
    
    p0 = [amp_estimada, decaimento_estimado]
    
    # Chutes para cada pico
    for i in range(n_picos):
        centro_estimado = eixo_energia[picos_info['indices'][i]]
        altura_estimada = picos_info['alturas'][i] - fundo_exponencial(centro_estimado, amp_estimada, decaimento_estimado)
        largura_estimada = picos_info['larguras'][i] * (eixo_energia[1] - eixo_energia[0])
        
        # Garante que a altura estimada seja positiva
        altura_estimada = max(altura_estimada, picos_info['alturas'][i] * 0.1)
        
        if tipo_pico in ['gaussiana', 'lorentziana']:
            p0.extend([altura_estimada, centro_estimado, largura_estimada])
        elif tipo_pico == 'voigt':
            p0.extend([altura_estimada, centro_estimado, largura_estimada/2, largura_estimada/2])
    
    try:
        # Ajuste global
        popt, pcov = curve_fit(modelo_global, eixo_energia, espectro, p0=p0, method=metodo, maxfev=5000)
        
        # Calcula R²
        y_pred = modelo_global(eixo_energia, *popt)
        ss_res = np.sum((espectro - y_pred)**2)
        ss_tot = np.sum((espectro - np.mean(espectro))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Separa parâmetros do fundo e dos picos
        parametros_fundo = popt[:parametros_fundo]
        parametros_picos = popt[parametros_fundo:]
        
        # Organiza resultados
        resultado = {
            'sucesso': True,
            'parametros': popt,
            'parametros_fundo': parametros_fundo,
            'parametros_picos': parametros_picos,
            'covariancia': pcov,
            'r_quadrado': r_squared,
            'y_ajustado': y_pred,
            'tipo_ajuste': 'global',
            'n_picos': n_picos,
            'tipo_pico': tipo_pico,
            'tipo_fundo': tipo_fundo
        }
        
        print(f"Ajuste global bem-sucedido! R² = {r_squared:.4f}")
        
    except Exception as e:
        print(f"Ajuste global falhou: {e}")
        resultado = {
            'sucesso': False,
            'erro': str(e),
            'tipo_ajuste': 'global'
        }
    
    return resultado

# =============================================================================
# AJUSTE INDIVIDUAL COM FUNDO EXPONENCIAL (FALLBACK)
# =============================================================================

def ajustar_picos_individual(eixo_energia, espectro, picos_info, 
                           tipo_pico='gaussiana', tipo_fundo='exponencial', janela_relativa=0.15):
    """
    Ajuste individual de cada pico (fallback quando o global falha).
    """
    resultados_individuais = []
    y_ajustado_total = np.zeros_like(espectro)
    
    # Primeiro ajusta o fundo global exponencial
    try:
        # Estimativa inicial para fundo
        amp_estimada = np.max(espectro) * 0.8
        decaimento_estimado = 0.05
        
        # Ajuste do fundo global, evitando regiões de picos
        mask_fundo = np.ones_like(espectro, dtype=bool)
        for idx in picos_info['indices']:
            largura = picos_info['larguras'][np.where(picos_info['indices'] == idx)[0][0]]
            inicio = max(0, int(idx - largura * 3))
            fim = min(len(espectro), int(idx + largura * 3))
            mask_fundo[inicio:fim] = False
        
        if np.sum(mask_fundo) > 10:
            popt_fundo, _ = curve_fit(fundo_exponencial, eixo_energia[mask_fundo], espectro[mask_fundo],
                                    p0=[amp_estimada, decaimento_estimado], maxfev=1000)
            fundo_global = fundo_exponencial(eixo_energia, *popt_fundo)
        else:
            fundo_global = fundo_exponencial(eixo_energia, amp_estimada, decaimento_estimado)
    except:
        # Fallback: fundo constante
        fundo_global = np.full_like(espectro, np.median(espectro))
    
    y_sem_fundo = espectro - fundo_global
    y_ajustado_total += fundo_global
    
    # Ajusta cada pico individualmente
    for i, pico_idx in enumerate(picos_info['indices']):
        centro_estimado = eixo_energia[pico_idx]
        
        # Define janela ao redor do pico
        largura_pico = picos_info['larguras'][i]
        largura_janela = int(max(largura_pico * 4, len(eixo_energia) * janela_relativa))
        inicio = max(0, pico_idx - largura_janela)
        fim = min(len(eixo_energia), pico_idx + largura_janela)
        
        x_local = eixo_energia[inicio:fim]
        y_local = y_sem_fundo[inicio:fim]
        
        # Chutes iniciais para o pico
        altura_estimada = picos_info['alturas'][i] - fundo_global[pico_idx]
        largura_estimada = largura_pico * (eixo_energia[1] - eixo_energia[0])
        
        # Garante valores positivos e razoáveis
        altura_estimada = max(altura_estimada, picos_info['alturas'][i] * 0.1)
        largura_estimada = max(largura_estimada, (eixo_energia[1] - eixo_energia[0]) * 2)
        
        if tipo_pico == 'gaussiana':
            funcao_ajuste = gaussiana
            p0 = [altura_estimada, centro_estimado, largura_estimada]
            bounds = ([0, centro_estimado*0.9, largura_estimada*0.1], 
                     [altura_estimada*10, centro_estimado*1.1, largura_estimada*10])
        elif tipo_pico == 'lorentziana':
            funcao_ajuste = lorentziana
            p0 = [altura_estimada, centro_estimado, largura_estimada]
            bounds = ([0, centro_estimado*0.9, largura_estimada*0.1], 
                     [altura_estimada*10, centro_estimado*1.1, largura_estimada*10])
        elif tipo_pico == 'voigt':
            funcao_ajuste = voigt
            p0 = [altura_estimada, centro_estimado, largura_estimada/2, largura_estimada/2]
            bounds = ([0, centro_estimado*0.9, largura_estimada*0.05, largura_estimada*0.05], 
                     [altura_estimada*10, centro_estimado*1.1, largura_estimada*5, largura_estimada*5])
        
        try:
            popt, pcov = curve_fit(funcao_ajuste, x_local, y_local, p0=p0, 
                                  bounds=bounds, maxfev=2000)
            y_pico_ajustado = funcao_ajuste(eixo_energia, *popt)
            y_ajustado_total += y_pico_ajustado
            
            resultados_individuais.append({
                'sucesso': True,
                'parametros': popt,
                'covariancia': pcov,
                'indice_pico': i,
                'centro': popt[1],
                'altura': popt[0],
                'largura': popt[2] if len(popt) > 2 else popt[2] + popt[3]
            })
            
            print(f"Pico {i+1} ajustado individualmente: centro={popt[1]:.1f}, altura={popt[0]:.1f}")
            
        except Exception as e:
            print(f"Ajuste individual do pico {i+1} falhou: {e}")
            resultados_individuais.append({
                'sucesso': False,
                'erro': str(e),
                'indice_pico': i
            })
    
    # Calcula R² geral
    ss_res = np.sum((espectro - y_ajustado_total)**2)
    ss_tot = np.sum((espectro - np.mean(espectro))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    resultado = {
        'sucesso': len([r for r in resultados_individuais if r['sucesso']]) > 0,
        'resultados_individuais': resultados_individuais,
        'r_quadrado': r_squared,
        'y_ajustado': y_ajustado_total,
        'fundo_ajustado': fundo_global,
        'tipo_ajuste': 'individual',
        'n_picos_ajustados': sum(1 for r in resultados_individuais if r['sucesso']),
        'tipo_pico': tipo_pico,
        'tipo_fundo': tipo_fundo
    }
    
    print(f"Ajuste individual completo. {resultado['n_picos_ajustados']}/{len(picos_info['indices'])} "
          f"picos ajustados. R² = {r_squared:.4f}")
    
    return resultado

# =============================================================================
# VISUALIZAÇÃO MELHORADA
# =============================================================================

def visualizar_ajuste(eixo_energia, espectro, picos_info, resultado_ajuste, 
                     mostrar_componentes=True, salvar_grafico=None):
    """
    Visualiza o resultado do ajuste com componentes individuais.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Gráfico superior: dados e ajuste completo
    ax1.plot(eixo_energia, espectro, 'k-', alpha=0.7, linewidth=1, 
             label='Dados Originais', drawstyle='steps-mid')
    
    if resultado_ajuste['sucesso']:
        ax1.plot(eixo_energia, resultado_ajuste['y_ajustado'], 'r-', linewidth=2, 
                label=f"Ajuste ({resultado_ajuste['tipo_ajuste']}, R² = {resultado_ajuste['r_quadrado']:.4f})")
        
        # Mostra componentes individuais se solicitado
        if mostrar_componentes and resultado_ajuste['tipo_ajuste'] == 'individual':
            # Fundo
            ax1.plot(eixo_energia, resultado_ajuste['fundo_ajustado'], 'g--', 
                    alpha=0.7, label='Fundo Ajustado')
            
            # Picos individuais
            for i, resultado_pico in enumerate(resultado_ajuste['resultados_individuais']):
                if resultado_pico['sucesso']:
                    if resultado_ajuste['tipo_pico'] == 'gaussiana':
                        y_pico = gaussiana(eixo_energia, *resultado_pico['parametros'])
                    elif resultado_ajuste['tipo_pico'] == 'lorentziana':
                        y_pico = lorentziana(eixo_energia, *resultado_pico['parametros'])
                    elif resultado_ajuste['tipo_pico'] == 'voigt':
                        y_pico = voigt(eixo_energia, *resultado_pico['parametros'])
                    
                    ax1.plot(eixo_energia, y_pico + resultado_ajuste['fundo_ajustado'], 
                            '--', alpha=0.5, label=f'Pico {i+1}')
    
    # Picos detectados
    ax1.plot(eixo_energia[picos_info['indices']], espectro[picos_info['indices']], 
            'ro', markersize=8, label='Picos Detectados', zorder=5)
    
    ax1.set_xlabel('Energia (Canal)')
    ax1.set_ylabel('Contagens')
    ax1.set_title('Análise de Espectro - Detecção e Ajuste de Picos')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico inferior: resíduos
    if resultado_ajuste['sucesso']:
        residuos = espectro - resultado_ajuste['y_ajustado']
        ax2.plot(eixo_energia, residuos, 'k-', linewidth=1, drawstyle='steps-mid')
        ax2.axhline(y=0, color='r', linestyle='-', alpha=0.7)
        ax2.set_xlabel('Energia (Canal)')
        ax2.set_ylabel('Resíduos')
        ax2.set_title('Resíduos do Ajuste')
        ax2.grid(True, alpha=0.3)
        
        # Estatísticas dos resíduos
        rms_residuos = np.sqrt(np.mean(residuos**2))
        ax2.text(0.02, 0.98, f'RMS Resíduos: {rms_residuos:.2f}', 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if salvar_grafico:
        # Criar a pasta 'analises_espectros' se não existir
        pasta_analises = "analises_espectros"
        os.makedirs(pasta_analises, exist_ok=True)

        # Caminho completo para salvar o gráfico
        caminho_completo = os.path.join(pasta_analises, salvar_grafico)
        plt.savefig(caminho_completo, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo como {caminho_completo}")
    plt.show()
    
    return fig, (ax1, ax2)