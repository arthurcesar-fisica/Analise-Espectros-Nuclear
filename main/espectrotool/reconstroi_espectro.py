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
# VISUALIZAÇÃO
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

def ajuste_pico_individual(eixo_energia, espectro, picos_info, 
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
    from .detecta_picos import detectar_picos
    from .visualizacao_dados import visualizar_deteccao
    picos_info = detectar_picos(espectro, **parametros_deteccao)
    
    if len(picos_info['indices']) == 0:
        print("Nenhum pico detectado. Análise interrompida.")
        return None
    
    # Visualiza detecção
    visualizar_deteccao(eixo_energia, espectro, picos_info, 
                       salvar_grafico="deteccao_picos.png")
    
    # 2. Reconstrução do espectro 
    print("\n2. RECONSTRUÇÃO DO ESPECTRO")
    sinal_reconstruido, resultado_ajuste = obter_sinal_reconstruido(
        eixo_energia=eixo_energia,
        espectro=espectro,
        picos_info=picos_info,
        **parametros_ajuste
    )
    
    # 3. Visualização dos resultados
    print("\n3. VISUALIZAÇÃO DOS RESULTADOS")
    visualizar_ajuste(eixo_energia, espectro, picos_info, resultado_ajuste,
                     mostrar_componentes=True, 
                     salvar_grafico="reconstrucao_espectro.png")
    
    # 4. Exportação dos resultados
    print("\n4. EXPORTAÇÃO DOS RESULTADOS")
    exportar_resultados(eixo_energia, espectro, picos_info, resultado_ajuste)
    
    print("ANÁLISE CONCLUÍDA COM SUCESSO!")
    
    return {
        'picos_info': picos_info,
        'resultado_ajuste': resultado_ajuste,
        'sinal_reconstruido': sinal_reconstruido  # Inclui o sinal reconstruído no retorno
    }

# =============================================================================
# FUNÇÃO PARA OBTER SINAL RECONSTRUÍDO
# =============================================================================
def obter_sinal_reconstruido(eixo_energia, espectro, picos_info, 
                            tipo_pico='gaussiana', tipo_fundo='exponencial'):
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
    
    Retorna:
    --------
    array
        Sinal reconstruído com o mesmo shape do espectro original
    """
    
    resultado_individual = ajuste_pico_individual(eixo_energia, espectro, picos_info,
                                                  tipo_pico=tipo_pico,
                                                  tipo_fundo=tipo_fundo)

    if resultado_individual['sucesso']:
        yAjustado = resultado_individual['y_ajustado']
    else:
        print("Erro: Não foi possível reconstruir o sinal.")
        yAjustado = np.zeros_like(espectro)
    
    resultado_ajuste = {
        'sucesso': True,
        'y_ajustado': yAjustado,
        'r_quadrado': 1 - (np.sum((espectro - yAjustado)**2) / np.sum((espectro - np.mean(espectro))**2))
                          if np.sum((espectro - np.mean(espectro))**2) != 0 else 0,
        'tipo_ajuste': 'otimizado',
        'n_picos_ajustados': len(picos_info['indices']),
        'tipo_pico': tipo_pico,
        'tipo_fundo': tipo_fundo
    }

    return yAjustado, resultado_ajuste