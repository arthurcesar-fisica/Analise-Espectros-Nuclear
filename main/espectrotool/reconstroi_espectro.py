# reconstroi_espectro.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import wofz
import os
import pandas as pd
import json

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
# FUNÇÕES AUXILIARES PARA PICOS PRÓXIMOS
# =============================================================================

def _separar_picos_proximos(picos_info, eixo_energia, distancia_minima_canais=30):
    """
    Identifica grupos de picos que estão muito próximos para ajuste conjunto.
    """
    indices = picos_info['indices']
    if len(indices) < 2:
        return [indices]
    
    # Ordena os índices
    indices_ordenados = sorted(indices)
    
    grupos = []
    grupo_atual = [indices_ordenados[0]]
    
    for i in range(1, len(indices_ordenados)):
        distancia = indices_ordenados[i] - indices_ordenados[i-1]
        if distancia <= distancia_minima_canais:
            grupo_atual.append(indices_ordenados[i])
        else:
            grupos.append(grupo_atual)
            grupo_atual = [indices_ordenados[i]]
    
    grupos.append(grupo_atual)
    
    print(f"Identificados {len(grupos)} grupo(s) de picos:")
    for i, grupo in enumerate(grupos):
        print(f"  Grupo {i+1}: {len(grupo)} pico(s) nos canais {grupo}")
    
    return grupos

def _ajustar_grupo_picos(eixo_energia, espectro, picos_info, indices_grupo, 
                         tipo_pico='gaussiana', tipo_fundo='exponencial'):
    """
    Ajusta um grupo de picos próximos simultaneamente.
    """
    n_picos = len(indices_grupo)
    
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
    elif tipo_fundo == 'nenhum':
        funcao_fundo = None
        parametros_fundo = 0
    else:
        raise ValueError("Tipo de fundo deve ser 'exponencial' ou 'nenhum'")
    
    # Função modelo para o grupo
    if funcao_fundo is not None:
        def modelo_grupo(x, amp_fundo, decaimento_fundo, *params_picos):
            y = funcao_fundo(x, amp_fundo, decaimento_fundo)
            
            for i in range(n_picos):
                inicio = i * parametros_por_pico
                fim = inicio + parametros_por_pico
                params_pico = params_picos[inicio:fim]
                y += funcao_pico(x, *params_pico)
            
            return y
    else:
        def modelo_grupo(x, *params_picos):
            y = 0.0
            for i in range(n_picos):
                inicio = i * parametros_por_pico
                fim = inicio + parametros_por_pico
                params_pico = params_picos[inicio:fim]
                y += funcao_pico(x, *params_pico)
            return y
    
    # Encontra os índices no picos_info
    indices_no_info = []
    for idx in indices_grupo:
        pos = np.where(picos_info['indices'] == idx)[0]
        if len(pos) > 0:
            indices_no_info.append(pos[0])
    
    # Chutes iniciais
    p0 = []
    if funcao_fundo is not None:
        amp_estimada = np.max(espectro) * 0.8
        decaimento_estimado = 0.05
        p0 = [amp_estimada, decaimento_estimado]
    
    # Chutes para cada pico no grupo
    for idx_info in indices_no_info:
        centro_estimado = eixo_energia[picos_info['indices'][idx_info]]
        altura_estimada = picos_info['alturas'][idx_info]
        largura_estimada = picos_info['larguras'][idx_info] * (eixo_energia[1] - eixo_energia[0])
        
        altura_estimada = max(altura_estimada, picos_info['alturas'][idx_info] * 0.1)
        
        if tipo_pico in ['gaussiana', 'lorentziana']:
            p0.extend([altura_estimada, centro_estimado, largura_estimada])
        elif tipo_pico == 'voigt':
            p0.extend([altura_estimada, centro_estimado, largura_estimada/2, largura_estimada/2])
    
    # Define região de interesse para o ajuste do grupo
    if len(indices_grupo) > 0:
        indices_grupo_array = np.array(indices_grupo)
        inicio_roi = max(0, min(indices_grupo) - 50)
        fim_roi = min(len(eixo_energia), max(indices_grupo) + 50)
        
        mask_roi = (eixo_energia >= eixo_energia[inicio_roi]) & (eixo_energia <= eixo_energia[fim_roi-1])
        x_roi = eixo_energia[mask_roi]
        y_roi = espectro[mask_roi]
    else:
        # Se não há picos, usa todo o espectro
        x_roi = eixo_energia
        y_roi = espectro
    
    try:
        # Ajuste do grupo
        popt, pcov = curve_fit(modelo_grupo, x_roi, y_roi, p0=p0, maxfev=5000)
        
        # Separa parâmetros
        if funcao_fundo is not None:
            parametros_fundo = popt[:parametros_fundo]
            parametros_picos = popt[parametros_fundo:]
        else:
            parametros_fundo = None
            parametros_picos = popt
        
        # Calcula o ajuste completo
        y_pred = modelo_grupo(eixo_energia, *popt)
        
        resultado_grupo = {
            'sucesso': True,
            'parametros': popt,
            'parametros_fundo_local': parametros_fundo,
            'parametros_picos': parametros_picos,
            'y_ajustado': y_pred,
            'indices_picos': indices_grupo,
            'indices_no_info': indices_no_info
        }
        
    except Exception as e:
        print(f"Ajuste do grupo falhou: {e}")
        resultado_grupo = {
            'sucesso': False,
            'erro': str(e),
            'indices_picos': indices_grupo
        }
    
    return resultado_grupo

# =============================================================================
# AJUSTE NÃO-LINEAR GLOBAL MELHORADO
# =============================================================================

def ajustar_picos_global(eixo_energia, espectro, picos_info, 
                        tipo_pico='gaussiana', tipo_fundo='exponencial', 
                        metodo='lm', tratar_picos_proximos=True):
    """
    Ajuste não-linear global melhorado para lidar com picos próximos.
    """
    if len(picos_info['indices']) == 0:
        print("Nenhum pico detectado para ajuste.")
        return {'sucesso': False, 'erro': 'Nenhum pico detectado'}
    
    # Se há picos próximos e devemos tratá-los em grupos
    if tratar_picos_proximos and len(picos_info['indices']) > 1:
        grupos_picos = _separar_picos_proximos(picos_info, eixo_energia)
        
        # Se todos os picos estão em um único grupo, faz ajuste global normal
        if len(grupos_picos) == 1:
            print("Todos os picos estão suficientemente separados, usando ajuste global único.")
            return _ajustar_global_unico(eixo_energia, espectro, picos_info, tipo_pico, tipo_fundo, metodo)
        else:
            print("Picos próximos detectados, usando ajuste por grupos.")
            return _ajustar_por_grupos(eixo_energia, espectro, picos_info, grupos_picos, tipo_pico, tipo_fundo)
    else:
        return _ajustar_global_unico(eixo_energia, espectro, picos_info, tipo_pico, tipo_fundo, metodo)

def _ajustar_global_unico(eixo_energia, espectro, picos_info, tipo_pico, tipo_fundo, metodo):
    """Ajuste global único para todos os picos."""
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
    
    # Define função do fundo
    if tipo_fundo == 'exponencial':
        funcao_fundo = fundo_exponencial
        parametros_fundo = 2
    
    # Função modelo global
    def modelo_global(x, amp_fundo, decaimento_fundo, *params_picos):
        y = funcao_fundo(x, amp_fundo, decaimento_fundo)
        
        for i in range(n_picos):
            inicio = i * parametros_por_pico
            fim = inicio + parametros_por_pico
            params_pico = params_picos[inicio:fim]
            y += funcao_pico(x, *params_pico)
        
        return y
    
    # Chutes iniciais
    amp_estimada = np.max(espectro) * 0.8
    decaimento_estimado = 0.05
    
    p0 = [amp_estimada, decaimento_estimado]
    
    for i in range(n_picos):
        centro_estimado = eixo_energia[picos_info['indices'][i]]
        altura_estimada = picos_info['alturas'][i]
        largura_estimada = picos_info['larguras'][i] * (eixo_energia[1] - eixo_energia[0])
        
        altura_estimada = max(altura_estimada, picos_info['alturas'][i] * 0.1)
        
        if tipo_pico in ['gaussiana', 'lorentziana']:
            p0.extend([altura_estimada, centro_estimado, largura_estimada])
        elif tipo_pico == 'voigt':
            p0.extend([altura_estimada, centro_estimado, largura_estimada/2, largura_estimada/2])
    
    try:
        popt, pcov = curve_fit(modelo_global, eixo_energia, espectro, p0=p0, method=metodo, maxfev=5000)
        
        y_pred = modelo_global(eixo_energia, *popt)
        ss_res = np.sum((espectro - y_pred)**2)
        ss_tot = np.sum((espectro - np.mean(espectro))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        parametros_fundo = popt[:parametros_fundo]
        parametros_picos = popt[parametros_fundo:]
        
        resultado = {
            'sucesso': True,
            'parametros': popt,
            'parametros_fundo': parametros_fundo,
            'parametros_picos': parametros_picos,
            'covariancia': pcov,
            'r_quadrado': r_squared,
            'y_ajustado': y_pred,
            'tipo_ajuste': 'global_unico',
            'n_picos': n_picos,
            'tipo_pico': tipo_pico,
            'tipo_fundo': tipo_fundo
        }
        
        print(f"Ajuste global único bem-sucedido! R² = {r_squared:.4f}")
        
    except Exception as e:
        print(f"Ajuste global único falhou: {e}")
        resultado = {
            'sucesso': False,
            'erro': str(e),
            'tipo_ajuste': 'global_unico'
        }
    
    return resultado

def _ajustar_por_grupos(eixo_energia, espectro, picos_info, grupos_picos, tipo_pico, tipo_fundo):
    """Ajuste por grupos para picos próximos."""
    y_ajustado_total = np.zeros_like(espectro)
    resultados_grupos = []
    
    # Primeiro ajusta o fundo global
    try:
        mask_fundo = np.ones_like(espectro, dtype=bool)
        for idx in picos_info['indices']:
            pos = np.where(picos_info['indices'] == idx)[0]
            if len(pos) > 0:
                largura = picos_info['larguras'][pos[0]]
                inicio = max(0, int(idx - largura * 3))
                fim = min(len(espectro), int(idx + largura * 3))
                mask_fundo[inicio:fim] = False
        
        if np.sum(mask_fundo) > 10:
            amp_estimada = np.max(espectro) * 0.8
            decaimento_estimado = 0.05
            popt_fundo, _ = curve_fit(fundo_exponencial, eixo_energia[mask_fundo], espectro[mask_fundo],
                                    p0=[amp_estimada, decaimento_estimado], maxfev=1000)
            fundo_global = fundo_exponencial(eixo_energia, *popt_fundo)
        else:
            fundo_global = fundo_exponencial(eixo_energia, np.max(espectro)*0.8, 0.05)
    except Exception as e:
        print(f"Ajuste do fundo global falhou: {e}")
        fundo_global = np.full_like(espectro, np.median(espectro))
    
    y_ajustado_total += fundo_global
    y_sem_fundo = espectro - fundo_global
    
    # Ajusta cada grupo de picos
    for i, grupo in enumerate(grupos_picos):
        print(f"Ajustando grupo {i+1} com {len(grupo)} picos...")
        
        # Cria um picos_info temporário para o grupo
        indices_no_info = []
        alturas_grupo = []
        larguras_grupo = []
        
        for idx in grupo:
            pos = np.where(picos_info['indices'] == idx)[0]
            if len(pos) > 0:
                indices_no_info.append(pos[0])
                alturas_grupo.append(picos_info['alturas'][pos[0]])
                larguras_grupo.append(picos_info['larguras'][pos[0]])
        
        picos_info_grupo = {
            'indices': grupo,
            'alturas': alturas_grupo,
            'larguras': larguras_grupo
        }
        
        # Para ajuste por grupos, não usamos fundo adicional (já removemos o fundo global)
        resultado_grupo = _ajustar_grupo_picos(eixo_energia, y_sem_fundo, picos_info_grupo, 
                                              grupo, tipo_pico, 'nenhum')
        
        if resultado_grupo['sucesso']:
            # Extrai apenas a contribuição dos picos (sem fundo)
            params_picos_grupo = resultado_grupo['parametros_picos']
            n_picos_grupo = len(grupo)
            
            if tipo_pico == 'gaussiana':
                for j in range(n_picos_grupo):
                    inicio = j * 3
                    if inicio + 3 <= len(params_picos_grupo):
                        amp, centro, sigma = params_picos_grupo[inicio:inicio+3]
                        y_pico = gaussiana(eixo_energia, amp, centro, sigma)
                        y_ajustado_total += y_pico
            elif tipo_pico == 'lorentziana':
                for j in range(n_picos_grupo):
                    inicio = j * 3
                    if inicio + 3 <= len(params_picos_grupo):
                        amp, centro, gamma = params_picos_grupo[inicio:inicio+3]
                        y_pico = lorentziana(eixo_energia, amp, centro, gamma)
                        y_ajustado_total += y_pico
            elif tipo_pico == 'voigt':
                for j in range(n_picos_grupo):
                    inicio = j * 4
                    if inicio + 4 <= len(params_picos_grupo):
                        amp, centro, sigma, gamma = params_picos_grupo[inicio:inicio+4]
                        y_pico = voigt(eixo_energia, amp, centro, sigma, gamma)
                        y_ajustado_total += y_pico
            
            resultados_grupos.append(resultado_grupo)
        else:
            print(f"Grupo {i+1} não foi ajustado com sucesso.")
    
    # Calcula R²
    ss_res = np.sum((espectro - y_ajustado_total)**2)
    ss_tot = np.sum((espectro - np.mean(espectro))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    resultado = {
        'sucesso': len(resultados_grupos) > 0,
        'resultados_grupos': resultados_grupos,
        'r_quadrado': r_squared,
        'y_ajustado': y_ajustado_total,
        'fundo_ajustado': fundo_global,
        'tipo_ajuste': 'global_por_grupos',
        'n_picos_ajustados': sum(len(grupo) for grupo in grupos_picos),
        'tipo_pico': tipo_pico,
        'tipo_fundo': tipo_fundo
    }
    
    print(f"Ajuste por grupos completo. R² = {r_squared:.4f}")
    
    return resultado

# =============================================================================
# AJUSTE INDIVIDUAL (FALLBACK)
# =============================================================================

def ajustar_picos_individual(eixo_energia, espectro, picos_info, 
                           tipo_pico='gaussiana', tipo_fundo='exponencial', 
                           janela_relativa=0.15):
    """
    Ajuste individual de cada pico (fallback quando o global falha).
    """
    # Implementação similar à anterior, mantida por compatibilidade
    from .analise_picos import ajustar_picos_individual as ajuste_individual_legado
    return ajuste_individual_legado(eixo_energia, espectro, picos_info, 
                                  tipo_pico, tipo_fundo, janela_relativa)

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
        
        # Mostra componentes
        if mostrar_componentes:
            # Fundo
            if 'fundo_ajustado' in resultado_ajuste:
                ax1.plot(eixo_energia, resultado_ajuste['fundo_ajustado'], 'g--', 
                        alpha=0.7, label='Fundo Ajustado')
            
            # Picos individuais para ajuste por grupos
            if resultado_ajuste['tipo_ajuste'] == 'global_por_grupos' and 'resultados_grupos' in resultado_ajuste:
                for i, resultado_grupo in enumerate(resultado_ajuste['resultados_grupos']):
                    if resultado_grupo['sucesso']:
                        n_picos_grupo = len(resultado_grupo['indices_picos'])
                        params_picos = resultado_grupo['parametros_picos']
                        
                        for j in range(n_picos_grupo):
                            if resultado_ajuste['tipo_pico'] == 'gaussiana':
                                inicio = j * 3
                                amp, centro, sigma = params_picos[inicio:inicio+3]
                                y_pico = gaussiana(eixo_energia, amp, centro, sigma)
                            elif resultado_ajuste['tipo_pico'] == 'lorentziana':
                                inicio = j * 3
                                amp, centro, gamma = params_picos[inicio:inicio+3]
                                y_pico = lorentziana(eixo_energia, amp, centro, gamma)
                            elif resultado_ajuste['tipo_pico'] == 'voigt':
                                inicio = j * 4
                                amp, centro, sigma, gamma = params_picos[inicio:inicio+4]
                                y_pico = voigt(eixo_energia, amp, centro, sigma, gamma)
                            
                            y_pico_total = y_pico + resultado_ajuste.get('fundo_ajustado', 0)
                            ax1.plot(eixo_energia, y_pico_total, '--', alpha=0.5, 
                                    label=f'Pico G{i+1}-{j+1}')
    
    # Picos detectados
    ax1.plot(eixo_energia[picos_info['indices']], espectro[picos_info['indices']], 
            'ro', markersize=8, label='Picos Detectados', zorder=5)
    
    ax1.set_xlabel('Energia (Canal)')
    ax1.set_ylabel('Contagens')
    ax1.set_title('Reconstrução do Espectro - Ajuste de Picos')
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
        pasta_analises = "analises_espectros"
        os.makedirs(pasta_analises, exist_ok=True)
        caminho_completo = os.path.join(pasta_analises, salvar_grafico)
        plt.savefig(caminho_completo, dpi=300, bbox_inches='tight')
        print(f"Gráfico de reconstrução salvo como {caminho_completo}")
    
    plt.show()
    
    return fig, (ax1, ax2)

# =============================================================================
# IMPLEMENTAÇÃO COMPLETA DO AJUSTE INDIVIDUAL
# =============================================================================

def _ajustar_pico_individual_completo(eixo_energia, espectro, picos_info, 
                                    tipo_pico='gaussiana', tipo_fundo='exponencial', 
                                    janela_relativa=0.15):
    """
    Implementação completa do ajuste individual (fallback).
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
            largura_idx = picos_info['larguras'][np.where(picos_info['indices'] == idx)[0][0]]
            inicio = max(0, int(idx - largura_idx * 3))
            fim = min(len(espectro), int(idx + largura_idx * 3))
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
        
        # Define janela ao redor do pico baseada na largura detectada
        largura_pico = picos_info['larguras'][i]
        largura_janela = int(max(largura_pico * 6, len(eixo_energia) * janela_relativa))
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
            
            # Calcula área do pico (aproximada)
            if tipo_pico == 'gaussiana':
                area = popt[0] * popt[2] * np.sqrt(2 * np.pi)
            elif tipo_pico == 'lorentziana':
                area = np.pi * popt[0] * popt[2] / 2
            elif tipo_pico == 'voigt':
                # Aproximação para área Voigt
                area = popt[0] * np.sqrt(2 * np.pi) * popt[2]
            
            resultados_individuais.append({
                'sucesso': True,
                'parametros': popt,
                'covariancia': pcov,
                'indice_pico': i,
                'centro': popt[1],
                'altura': popt[0],
                'largura': popt[2] if len(popt) > 2 else popt[2] + popt[3],
                'area': area,
                'canal_pico': pico_idx
            })
            
            print(f"Pico {i+1} ajustado individualmente: centro={popt[1]:.1f}, "
                  f"altura={popt[0]:.1f}, área={area:.1f}")
            
        except Exception as e:
            print(f"Ajuste individual do pico {i+1} falhou: {e}")
            resultados_individuais.append({
                'sucesso': False,
                'erro': str(e),
                'indice_pico': i,
                'canal_pico': pico_idx
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

# Atualiza a função principal para usar a implementação completa
def ajustar_picos_individual(eixo_energia, espectro, picos_info, 
                           tipo_pico='gaussiana', tipo_fundo='exponencial', 
                           janela_relativa=0.15):
    """
    Ajuste individual de cada pico (fallback quando o global falha).
    """
    return _ajustar_pico_individual_completo(eixo_energia, espectro, picos_info,
                                           tipo_pico, tipo_fundo, janela_relativa)

# =============================================================================
# FUNÇÃO PARA EXPORTAR RESULTADOS
# =============================================================================

def exportar_resultados(eixo_energia, espectro, picos_info, resultado_ajuste, 
                       nome_arquivo="resultados_analise"):
    """
    Exporta os resultados da análise para arquivos Excel e JSON.
    """
    import pandas as pd
    import json
    
    pasta_analises = "analises_espectros"
    os.makedirs(pasta_analises, exist_ok=True)
    
    # DataFrame com dados originais e ajustados
    dados_completos = {
        'Energia': eixo_energia,
        'Contagens_Originais': espectro
    }
    
    if resultado_ajuste['sucesso']:
        dados_completos['Contagens_Ajustadas'] = resultado_ajuste['y_ajustado']
        dados_completos['Residuos'] = espectro - resultado_ajuste['y_ajustado']
    
    df_completo = pd.DataFrame(dados_completos)
    
    # DataFrame com informações dos picos
    dados_picos = []
    for i, (indice, altura, proeminencia, largura) in enumerate(zip(
        picos_info['indices'],
        picos_info['alturas'], 
        picos_info['proeminencias'],
        picos_info['larguras']
    )):
        info_pico = {
            'Pico_ID': i + 1,
            'Canal': indice,
            'Energia': eixo_energia[indice],
            'Altura_Detectada': altura,
            'Proeminencia': proeminencia,
            'Largura_Detectada': largura
        }
        
        # Adiciona informações do ajuste se disponível
        if resultado_ajuste['sucesso'] and 'resultados_individuais' in resultado_ajuste:
            resultados_pico = [r for r in resultado_ajuste['resultados_individuais'] 
                             if r.get('canal_pico') == indice and r['sucesso']]
            if resultados_pico:
                rp = resultados_pico[0]
                info_pico.update({
                    'Centro_Ajustado': rp['centro'],
                    'Altura_Ajustada': rp['altura'],
                    'Largura_Ajustada': rp['largura'],
                    'Area': rp.get('area', np.nan)
                })
        
        dados_picos.append(info_pico)
    
    df_picos = pd.DataFrame(dados_picos)
    
    # Salva os arquivos
    caminho_excel = os.path.join(pasta_analises, f"{nome_arquivo}.xlsx")
    with pd.ExcelWriter(caminho_excel) as writer:
        df_completo.to_excel(writer, sheet_name='Dados_Completos', index=False)
        df_picos.to_excel(writer, sheet_name='Picos_Detectados', index=False)
    
    # Salva JSON com metadados
    metadados = {
        'parametros_ajuste': {
            'tipo_ajuste': resultado_ajuste.get('tipo_ajuste', 'N/A'),
            'tipo_pico': resultado_ajuste.get('tipo_pico', 'N/A'),
            'tipo_fundo': resultado_ajuste.get('tipo_fundo', 'N/A'),
            'r_quadrado': resultado_ajuste.get('r_quadrado', 0),
            'n_picos_detectados': len(picos_info['indices']),
            'n_picos_ajustados': resultado_ajuste.get('n_picos_ajustados', 0)
        },
        'estatisticas': {
            'rms_residuos': float(np.sqrt(np.mean((espectro - resultado_ajuste.get('y_ajustado', espectro))**2))) 
            if resultado_ajuste.get('sucesso', False) else 0
        }
    }
    
    caminho_json = os.path.join(pasta_analises, f"{nome_arquivo}.json")
    with open(caminho_json, 'w', encoding='utf-8') as f:
        json.dump(metadados, f, indent=2, ensure_ascii=False)
    
    print(f"Resultados exportados:")
    print(f"  - Dados completos: {caminho_excel} (aba 'Dados_Completos')")
    print(f"  - Informações dos picos: {caminho_excel} (aba 'Picos_Detectados')")
    print(f"  - Metadados: {caminho_json}")

# =============================================================================
# FUNÇÃO PRINCIPAL DE ANÁLISE COMPLETA
# =============================================================================

# =============================================================================
# FUNÇÃO PRINCIPAL DE ANÁLISE COMPLETA (EDITADA)
# =============================================================================

def analise_completa_espectro(eixo_energia, espectro, 
                             parametros_deteccao=None,
                             parametros_ajuste=None):
    """
    Executa uma análise completa do espectro: detecção + reconstrução + exportação.
    
    Parâmetros:
    -----------
    eixo_energia : array
        Eixo de energias/canais
    espectro : array
        Dados do espectro
    parametros_deteccao : dict, opcional
        Parâmetros para detecção de picos
    parametros_ajuste : dict, opcional
        Parâmetros para ajuste/reconstrução
    
    Retorna:
    --------
    dict
        Dicionário com informações dos picos, resultado do ajuste e sinal reconstruído
    """
    
    # Parâmetros padrão
    if parametros_deteccao is None:
        parametros_deteccao = {
            'altura_minima': None,
            'distancia_minima': 30,
            'proeminencia': None,
            'largura_minima': 3,
            'suavizar': True
        }
    
    if parametros_ajuste is None:
        parametros_ajuste = {
            'tipo_pico': 'gaussiana',
            'tipo_fundo': 'exponencial',
            'tratar_picos_proximos': True
        }
    
    print("=" * 60)
    print("ANÁLISE COMPLETA DO ESPECTRO")
    print("=" * 60)
    
    # 1. Detecção de picos
    print("\n1. DETECÇÃO DE PICOS")
    from .detecta_picos import detectar_picos, visualizar_deteccao
    picos_info = detectar_picos(espectro, **parametros_deteccao)
    
    if len(picos_info['indices']) == 0:
        print("Nenhum pico detectado. Análise interrompida.")
        return None
    
    # Visualiza detecção
    visualizar_deteccao(eixo_energia, espectro, picos_info, 
                       salvar_grafico="deteccao_picos.png")
    
    # 2. Reconstrução do espectro 
    print("\n2. RECONSTRUÇÃO DO ESPECTRO")
    sinal_reconstruido = obter_sinal_reconstruido(
        eixo_energia=eixo_energia,
        espectro=espectro,
        picos_info=picos_info,
        **parametros_ajuste
    )
    
    # Cria um objeto resultado_ajuste compatível para as funções seguintes
    resultado_ajuste = {
        'sucesso': True,
        'y_ajustado': sinal_reconstruido,
        'r_quadrado': 1 - (np.sum((espectro - sinal_reconstruido)**2) / np.sum((espectro - np.mean(espectro))**2)) 
                      if np.sum((espectro - np.mean(espectro))**2) != 0 else 0,
        'tipo_ajuste': 'otimizado',
        'n_picos_ajustados': len(picos_info['indices']),
        'tipo_pico': parametros_ajuste.get('tipo_pico', 'gaussiana'),
        'tipo_fundo': parametros_ajuste.get('tipo_fundo', 'exponencial')
    }
    
    # 3. Visualização dos resultados
    print("\n3. VISUALIZAÇÃO DOS RESULTADOS")
    visualizar_ajuste(eixo_energia, espectro, picos_info, resultado_ajuste,
                     mostrar_componentes=True, 
                     salvar_grafico="reconstrucao_espectro.png")
    
    # 4. Exportação dos resultados
    print("\n4. EXPORTAÇÃO DOS RESULTADOS")
    exportar_resultados(eixo_energia, espectro, picos_info, resultado_ajuste)
    
    print("\n" + "=" * 60)
    print("ANÁLISE CONCLUÍDA COM SUCESSO!")
    print("=" * 60)
    
    return {
        'picos_info': picos_info,
        'resultado_ajuste': resultado_ajuste,
        'sinal_reconstruido': sinal_reconstruido  # Inclui o sinal reconstruído no retorno
    }

# =============================================================================
# FUNÇÃO PARA OBTER SINAL RECONSTRUÍDO
# =============================================================================
def obter_sinal_reconstruido(eixo_energia, espectro, picos_info, 
                            tipo_pico='gaussiana', tipo_fundo='exponencial',
                            tratar_picos_proximos=True):
    """
    Retorna apenas o array do sinal reconstruído a partir do ajuste dos picos.
    
    Parâmetros:
    -----------
    eixo_energia : array
        Eixo de energias/canais
    espectro : array
        Dados do espectro original
    picos_info : dict
        Informações dos picos detectados (indices, alturas, larguras, proeminencias)
    tipo_pico : str, opcional
        Tipo de função para ajuste dos picos ('gaussiana', 'lorentziana', 'voigt')
    tipo_fundo : str, opcional
        Tipo de função para ajuste do fundo ('exponencial')
    tratar_picos_proximos : bool, opcional
        Se True, trata picos próximos em grupos separados
    
    Retorna:
    --------
    array
        Sinal reconstruído com o mesmo shape do espectro original
    """
    
    # Primeiro tenta o ajuste global
    resultado_global = ajustar_picos_global(eixo_energia, espectro, picos_info,
                                          tipo_pico=tipo_pico,
                                          tipo_fundo=tipo_fundo,
                                          tratar_picos_proximos=tratar_picos_proximos)
    
    # Se o ajuste global foi bem-sucedido, retorna o sinal reconstruído
    if resultado_global['sucesso']:
        return resultado_global['y_ajustado']
    
    # Se o ajuste global falhou, tenta o ajuste individual como fallback
    print("Ajuste global falhou, usando ajuste individual como fallback...")
    resultado_individual = ajustar_picos_individual(eixo_energia, espectro, picos_info,
                                                  tipo_pico=tipo_pico,
                                                  tipo_fundo=tipo_fundo)
    
    if resultado_individual['sucesso']:
        return resultado_individual['y_ajustado']
    else:
        print("Erro: Não foi possível reconstruir o sinal.")
        # Retorna array de zeros como fallback final
        return np.zeros_like(espectro)