# visualizacao_dados.py

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Optional, Tuple

def configurar_estilos_graficos():
    """
    Configura estilos padrão para os gráficos.
    """
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

def mostra_espectro(eixo_energia: np.ndarray, espectro: np.ndarray, 
                   titulo: str = 'Espectro', salvar_grafico: Optional[str] = None):
    """
    Exibe um espectro simples.
    
    Args:
        eixo_energia: Array com os valores do eixo de energia/canais
        espectro: Array com os dados do espectro
        titulo: Título do gráfico
        salvar_grafico: Caminho para salvar o gráfico (opcional)
    """
    configurar_estilos_graficos()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(eixo_energia, espectro, 'b-', linewidth=1, drawstyle='steps-mid')
    ax.set_xlabel('Canal de Energia')
    ax.set_ylabel('Contagens')
    ax.set_title(titulo)
    ax.grid(True, alpha=0.3)
    
    if salvar_grafico:
        pasta_analises = "analises_espectros"
        os.makedirs(pasta_analises, exist_ok=True)
        caminho_completo = os.path.join(pasta_analises, salvar_grafico)
        plt.savefig(caminho_completo, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo como {caminho_completo}")
    
    plt.show()
    return fig, ax

def visualizar_deteccao(eixo_energia: np.ndarray, espectro: np.ndarray, 
                       picos_info: Dict, salvar_grafico: Optional[str] = None):
    """
    Visualiza a detecção de picos.
    
    Args:
        eixo_energia: Array com os valores do eixo de energia/canais
        espectro: Array com os dados do espectro
        picos_info: Dicionário com informações dos picos detectados
        salvar_grafico: Caminho para salvar o gráfico (opcional)
    """
    configurar_estilos_graficos()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Espectro original
    ax.plot(eixo_energia, espectro, 'k-', alpha=0.7, linewidth=1, 
            label='Espectro Original', drawstyle='steps-mid')
    
    # Espectro suavizado (se aplicado)
    if picos_info.get('espectro_suavizado') is not None:
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

def visualizar_ajuste(eixo_energia: np.ndarray, espectro: np.ndarray, 
                     picos_info: Dict, resultado_ajuste: Dict,
                     mostrar_componentes: bool = True, 
                     salvar_grafico: Optional[str] = None):
    """
    Visualiza o resultado do ajuste com componentes individuais.
    
    Args:
        eixo_energia: Array com os valores do eixo de energia/canais
        espectro: Array com os dados do espectro
        picos_info: Dicionário com informações dos picos detectados
        resultado_ajuste: Dicionário com resultados do ajuste
        mostrar_componentes: Se True, mostra componentes individuais
        salvar_grafico: Caminho para salvar o gráfico (opcional)
    """
    configurar_estilos_graficos()
    
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

def visualizar_comparacao_multipla(espectros: List[Tuple[np.ndarray, np.ndarray, str]],
                                 titulo: str = "Comparação de Múltiplos Espectros",
                                 salvar_grafico: Optional[str] = None):
    """
    Visualiza múltiplos espectros para comparação.
    
    Args:
        espectros: Lista de tuplas (eixo_energia, espectro, label)
        titulo: Título do gráfico
        salvar_grafico: Caminho para salvar o gráfico (opcional)
    """
    configurar_estilos_graficos()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    cores = plt.cm.tab10(np.linspace(0, 1, len(espectros)))
    
    for i, (eixo, dados, label) in enumerate(espectros):
        ax.plot(eixo, dados, color=cores[i], linewidth=1.5, 
                label=label, drawstyle='steps-mid', alpha=0.8)
    
    ax.set_xlabel('Energia (Canal)')
    ax.set_ylabel('Contagens')
    ax.set_title(titulo)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if salvar_grafico:
        pasta_analises = "analises_espectros"
        os.makedirs(pasta_analises, exist_ok=True)
        caminho_completo = os.path.join(pasta_analises, salvar_grafico)
        plt.savefig(caminho_completo, dpi=300, bbox_inches='tight')
        print(f"Gráfico de comparação salvo como {caminho_completo}")
    
    plt.show()
    return fig, ax

def visualizar_metricas_validacao(metricas: Dict, salvar_grafico: Optional[str] = None):
    """
    Visualiza métricas de validação em gráficos de barras.
    
    Args:
        metricas: Dicionário com métricas de validação
        salvar_grafico: Caminho para salvar o gráfico (opcional)
    """
    configurar_estilos_graficos()
    
    if 'parameter_validation' not in metricas or not metricas['parameter_validation']:
        print("Nenhuma métrica de validação de parâmetros disponível para visualização.")
        return None, None
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    pv = metricas['parameter_validation']
    
    # 1. Erros relativos por parâmetro
    erros_amplitude = []
    erros_centro = []
    erros_sigma = []
    
    for peak_data in pv.values():
        erros_amplitude.append(peak_data['amplitude']['rel_error_percent'])
        erros_centro.append(peak_data['centro']['rel_error_percent'])
        erros_sigma.append(peak_data['sigma']['rel_error_percent'])
    
    x_pos = np.arange(len(erros_amplitude))
    width = 0.25
    
    axes[0].bar(x_pos - width, erros_amplitude, width, label='Amplitude', alpha=0.7)
    axes[0].bar(x_pos, erros_centro, width, label='Centro', alpha=0.7)
    axes[0].bar(x_pos + width, erros_sigma, width, label='Sigma', alpha=0.7)
    
    axes[0].axhline(y=10, color='r', linestyle='--', alpha=0.7, label='Tol. Amplitude (10%)')
    axes[0].axhline(y=2, color='g', linestyle='--', alpha=0.7, label='Tol. Centro (2%)')
    axes[0].axhline(y=15, color='b', linestyle='--', alpha=0.7, label='Tol. Sigma (15%)')
    
    axes[0].set_xlabel('Pico')
    axes[0].set_ylabel('Erro Relativo (%)')
    axes[0].set_title('Erros Relativos dos Parâmetros por Pico')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Comparação valores verdadeiros vs ajustados
    for i, (peak_name, peak_data) in enumerate(pv.items()):
        centros_verdadeiros = [peak_data['centro']['true']]
        centros_ajustados = [peak_data['centro']['fitted']]
        
        axes[1].plot(centros_verdadeiros, centros_ajustados, 'o', markersize=8, 
                    label=f'Pico {i+1}', alpha=0.7)
    
    # Linha de referência y=x
    min_val = min([ax.get_xlim()[0] for ax in [axes[1]]] + [ax.get_ylim()[0] for ax in [axes[1]]])
    max_val = max([ax.get_xlim()[1] for ax in [axes[1]]] + [ax.get_ylim()[1] for ax in [axes[1]]])
    axes[1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y = x')
    
    axes[1].set_xlabel('Centro Verdadeiro')
    axes[1].set_ylabel('Centro Ajustado')
    axes[1].set_title('Comparação: Centro Verdadeiro vs Ajustado')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Distribuição dos resíduos (se disponível)
    if 'residual_analysis' in metricas:
        ra = metricas['residual_analysis']
        residuos_simulados = np.random.normal(ra['bias'], ra['std'], 1000)
        
        axes[2].hist(residuos_simulados, bins=30, density=True, alpha=0.7, 
                    color='skyblue', edgecolor='black')
        
        # Adiciona curva normal teórica
        x = np.linspace(residuos_simulados.min(), residuos_simulados.max(), 100)
        from scipy.stats import norm
        y = norm.pdf(x, ra['bias'], ra['std'])
        axes[2].plot(x, y, 'r-', linewidth=2, label=f'Normal (μ={ra["bias"]:.2f}, σ={ra["std"]:.2f})')
        
        axes[2].axvline(x=ra['bias'], color='r', linestyle='--', alpha=0.7)
        axes[2].axvline(x=ra['bias'] - ra['std'], color='g', linestyle='--', alpha=0.5, label='±1σ')
        axes[2].axvline(x=ra['bias'] + ra['std'], color='g', linestyle='--', alpha=0.5)
        
        axes[2].set_xlabel('Resíduos')
        axes[2].set_ylabel('Densidade')
        axes[2].set_title('Distribuição dos Resíduos')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    # 4. Status dos requisitos
    if 'requirements_check' in metricas:
        req = metricas['requirements_check']
        
        requisitos = []
        status = []
        cores = []
        
        for key, value in req.items():
            if key in ['parameters_within_tolerance', 'rmse_comparable_to_noise', 
                      'bias_near_zero', 'residuals_normal_distribution'] and value is not None:
                requisitos.append(key.replace('_', ' ').title())
                status.append(1 if value else 0)
                cores.append('green' if value else 'red')
        
        if requisitos:
            bars = axes[3].bar(requisitos, status, color=cores, alpha=0.7)
            axes[3].set_ylabel('Atendido (1 = Sim, 0 = Não)')
            axes[3].set_title('Status dos Requisitos de Qualidade')
            axes[3].tick_params(axis='x', rotation=45)
            
            # Adiciona valores nas barras
            for bar, s in zip(bars, status):
                height = bar.get_height()
                axes[3].text(bar.get_x() + bar.get_width()/2., height/2,
                            f"{'✓' if s == 1 else '✗'}", ha='center', va='center',
                            fontweight='bold', fontsize=12,
                            color='white' if s == 1 else 'black')
    
    plt.tight_layout()
    
    if salvar_grafico:
        pasta_analises = "analises_espectros"
        os.makedirs(pasta_analises, exist_ok=True)
        caminho_completo = os.path.join(pasta_analises, salvar_grafico)
        plt.savefig(caminho_completo, dpi=300, bbox_inches='tight')
        print(f"Gráfico de métricas salvo como {caminho_completo}")
    
    plt.show()
    return fig, axes

# Funções auxiliares necessárias para visualizar_ajuste
def gaussiana(x, amp, centro, sigma):
    """Função gaussiana para visualização."""
    return amp * np.exp(-(x - centro)**2 / (2 * sigma**2))

def lorentziana(x, amp, centro, gamma):
    """Função lorentziana para visualização."""
    return amp * (gamma**2 / ((x - centro)**2 + gamma**2))

def voigt(x, amp, centro, sigma, gamma):
    """Função Voigt para visualização."""
    from scipy.special import wofz
    z = (x - centro + 1j*gamma) / (sigma * np.sqrt(2))
    return amp * np.real(wofz(z)) / (sigma * np.sqrt(2*np.pi))