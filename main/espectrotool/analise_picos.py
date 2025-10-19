# analise_picos.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
from lmfit import Model, Parameters
import warnings

# =============================================================================
# DETECÇÃO AUTOMÁTICA DE PICOS
# =============================================================================

def detectar_picos(espectro, altura_minima=None, distancia_minima=20, 
                  proeminencia=None, largura_minima=5, relacao_sinal_ruido=2.0):
    """
    Detecta picos automaticamente no espectro usando análise de proeminência e largura.
    
    Parâmetros:
    -----------
    espectro : array
        Dados do espectro (contagens)
    altura_minima : float, opcional
        Altura mínima absoluta para detecção de picos
    distancia_minima : int
        Distância mínima entre picos (em canais)
    proeminencia : float, opcional
        Proeminência mínima do pico. Se None, calculada automaticamente
    largura_minima : int
        Largura mínima do pico (em canais)
    relacao_sinal_ruido : float
        Mínima relação sinal/ruído para considerar um pico
    
    Retorna:
    --------
    picos_info : dict
        Informações sobre os picos detectados
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
    for i, (pico, altura, prominencia, largura) in enumerate(zip(
        picos_info['indices'], 
        picos_info['alturas'], 
        picos_info['proeminencias'], 
        picos_info['larguras']
    )):
        print(f"  Pico {i+1}: Canal {pico}, Altura {altura:.1f}, "
              f"Proeminência {prominencia:.1f}, Largura {largura:.1f} canais")
    
    return picos_info

# =============================================================================
# FUNÇÕES DE MODELO PARA AJUSTE
# =============================================================================

def gaussiana(x, amp, centro, sigma):
    """Função gaussiana para ajuste de picos."""
    return amp * np.exp(-(x - centro)**2 / (2 * sigma**2))

def lorentziana(x, amp, centro, gamma):
    """Função lorentziana para ajuste de picos."""
    return amp * (gamma**2 / ((x - centro)**2 + gamma**2))

def voigt(x, amp, centro, sigma, gamma):
    """Função Voigt (convolução Gaussiana-Lorentziana)."""
    from scipy.special import wofz
    z = (x - centro + 1j*gamma) / (sigma * np.sqrt(2))
    return amp * np.real(wofz(z)) / (sigma * np.sqrt(2*np.pi))

def fundo_polinomial(x, *coefs):
    """Função de fundo polinomial."""
    return np.polyval(coefs, x)

# =============================================================================
# AJUSTE NÃO-LINEAR GLOBAL
# =============================================================================

def ajustar_picos_global(eixo_energia, espectro, picos_info, 
                        tipo_pico='gaussiana', grau_fundo=1, metodo='lm'):
    """
    Ajuste não-linear global de todos os picos simultaneamente.
    
    Parâmetros:
    -----------
    eixo_energia : array
        Eixo de energias/canais
    espectro : array
        Dados do espectro
    picos_info : dict
        Informações dos picos detectados
    tipo_pico : str
        Tipo de função para ajuste ('gaussiana', 'lorentziana', 'voigt')
    grau_fundo : int
        Grau do polinômio para o fundo
    metodo : str
        Método de otimização
    
    Retorna:
    --------
    resultado_ajuste : dict
        Resultados do ajuste
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
    
    # Função modelo global
    def modelo_global(x, *params):
        # Separa parâmetros do fundo e dos picos
        coefs_fundo = params[:grau_fundo+1]
        params_picos = params[grau_fundo+1:]
        
        # Calcula fundo
        y = fundo_polinomial(x, *coefs_fundo)
        
        # Adiciona cada pico
        for i in range(n_picos):
            inicio = i * parametros_por_pico
            fim = inicio + parametros_por_pico
            params_pico = params_picos[inicio:fim]
            y += funcao_pico(x, *params_pico)
        
        return y
    
    # Chutes iniciais
    p0 = []
    
    # Chutes para fundo (linear por padrão)
    p0.extend([0.1] * (grau_fundo + 1))
    
    # Chutes para cada pico
    for i in range(n_picos):
        centro_estimado = eixo_energia[picos_info['indices'][i]]
        altura_estimada = picos_info['alturas'][i]
        largura_estimada = picos_info['larguras'][i] * (eixo_energia[1] - eixo_energia[0])
        
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
        r_squared = 1 - (ss_res / ss_tot)
        
        # Organiza resultados
        resultado = {
            'sucesso': True,
            'parametros': popt,
            'covariancia': pcov,
            'r_quadrado': r_squared,
            'y_ajustado': y_pred,
            'tipo_ajuste': 'global',
            'n_picos': n_picos,
            'tipo_pico': tipo_pico
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
# AJUSTE INDIVIDUAL (FALLBACK)
# =============================================================================

def ajustar_picos_individual(eixo_energia, espectro, picos_info, 
                           tipo_pico='gaussiana', grau_fundo=1, janela_relativa=0.1):
    """
    Ajuste individual de cada pico (fallback quando o global falha).
    
    Parâmetros:
    -----------
    eixo_energia : array
        Eixo de energias/canais
    espectro : array
        Dados do espectro
    picos_info : dict
        Informações dos picos detectados
    tipo_pico : str
        Tipo de função para ajuste
    grau_fundo : int
        Grau do polinômio para o fundo local
    janela_relativa : float
        Fração do espectro para análise de cada pico
    
    Retorna:
    --------
    resultado_ajuste : dict
        Resultados do ajuste
    """
    
    resultados_individuais = []
    y_ajustado_total = np.zeros_like(espectro)
    
    # Primeiro ajusta o fundo global
    try:
        coefs_fundo = np.polyfit(eixo_energia, espectro, grau_fundo)
        fundo_global = np.polyval(coefs_fundo, eixo_energia)
    except:
        fundo_global = np.full_like(espectro, np.median(espectro))
    
    y_sem_fundo = espectro - fundo_global
    y_ajustado_total += fundo_global
    
    # Ajusta cada pico individualmente
    for i, pico_idx in enumerate(picos_info['indices']):
        centro_estimado = eixo_energia[pico_idx]
        
        # Define janela ao redor do pico
        largura_janela = int(len(eixo_energia) * janela_relativa)
        inicio = max(0, pico_idx - largura_janela)
        fim = min(len(eixo_energia), pico_idx + largura_janela)
        
        x_local = eixo_energia[inicio:fim]
        y_local = y_sem_fundo[inicio:fim]
        
        # Chutes iniciais
        altura_estimada = picos_info['alturas'][i] - fundo_global[pico_idx]
        largura_estimada = picos_info['larguras'][i] * (eixo_energia[1] - eixo_energia[0])
        
        if tipo_pico == 'gaussiana':
            funcao_ajuste = gaussiana
            p0 = [altura_estimada, centro_estimado, largura_estimada]
        elif tipo_pico == 'lorentziana':
            funcao_ajuste = lorentziana
            p0 = [altura_estimada, centro_estimado, largura_estimada]
        elif tipo_pico == 'voigt':
            funcao_ajuste = voigt
            p0 = [altura_estimada, centro_estimado, largura_estimada/2, largura_estimada/2]
        
        try:
            popt, pcov = curve_fit(funcao_ajuste, x_local, y_local, p0=p0, maxfev=1000)
            y_pico_ajustado = funcao_ajuste(eixo_energia, *popt)
            y_ajustado_total += y_pico_ajustado
            
            resultados_individuais.append({
                'sucesso': True,
                'parametros': popt,
                'covariancia': pcov,
                'indice_pico': i
            })
            
            print(f"Pico {i+1} ajustado individualmente com sucesso")
            
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
        'sucesso': len(resultados_individuais) > 0,
        'resultados_individuais': resultados_individuais,
        'r_quadrado': r_squared,
        'y_ajustado': y_ajustado_total,
        'tipo_ajuste': 'individual',
        'n_picos_ajustados': sum(1 for r in resultados_individuais if r['sucesso']),
        'tipo_pico': tipo_pico
    }
    
    print(f"Ajuste individual completo. {resultado['n_picos_ajustados']}/{len(picos_info['indices'])} "
          f"picos ajustados. R² = {r_squared:.4f}")
    
    return resultado

# =============================================================================
# VISUALIZAÇÃO
# =============================================================================

def visualizar_ajuste(eixo_energia, espectro, picos_info, resultado_ajuste, 
                     mostrar_components=True, salvar_grafico=None):
    """
    Visualiza o resultado do ajuste.
    
    Parâmetros:
    -----------
    eixo_energia : array
        Eixo de energias
    espectro : array
        Dados originais
    picos_info : dict
        Informações dos picos
    resultado_ajuste : dict
        Resultado do ajuste
    mostrar_components : bool
        Se True, mostra componentes individuais
    salvar_grafico : str, opcional
        Caminho para salvar o gráfico
    """
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Dados originais
    ax.plot(eixo_energia, espectro, 'k-', alpha=0.7, linewidth=1, label='Dados Originais', drawstyle='steps-mid')
    
    # Ajuste
    if resultado_ajuste['sucesso']:
        ax.plot(eixo_energia, resultado_ajuste['y_ajustado'], 'r-', linewidth=2, 
               label=f"Ajuste ({resultado_ajuste['tipo_ajuste']}, R² = {resultado_ajuste['r_quadrado']:.4f})")
    
    # Picos detectados
    ax.plot(eixo_energia[picos_info['indices']], espectro[picos_info['indices']], 
           'ro', markersize=8, label='Picos Detectados')
    
    ax.set_xlabel('Energia (Canal)')
    ax.set_ylabel('Contagens')
    ax.set_title('Análise de Espectro - Detecção e Ajuste de Picos')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if salvar_grafico:
        plt.savefig(salvar_grafico, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo como {salvar_grafico}")
    
    plt.show()
    
    return fig, ax