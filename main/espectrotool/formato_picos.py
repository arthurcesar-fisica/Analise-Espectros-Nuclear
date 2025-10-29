# formato.py
"""
Funções utilitárias para converter entre formatos de parâmetros de picos.

Este módulo resolve a incompatibilidade entre:
- Formato 1: Lista de dicionários (gerador_espectros.py)
- Formato 2: Dicionário com arrays (detecta_picos.py)
- Formato 3: Array 2D NumPy (analise_metricas.py)
"""

import numpy as np
from typing import List, Dict, Tuple


# =============================================================================
# FORMATO PADRÃO RECOMENDADO: Dicionário com Arrays NumPy
# =============================================================================

def criar_picos_info_padrao(amplitudes, centros, sigmas, **kwargs):
    """
    Cria estrutura de picos_info no formato padrão do projeto.
    
    Este é o formato RECOMENDADO para usar em todo o projeto.
    
    Args:
        amplitudes: Array ou lista com amplitudes dos picos
        centros: Array ou lista com posições centrais dos picos
        sigmas: Array ou lista com larguras (sigma) dos picos
        **kwargs: Campos opcionais (proeminencias, limites_esquerda, etc.)
    
    Returns:
        dict: Dicionário padronizado com informações dos picos
        
    Exemplo:
        >>> picos_info = criar_picos_info_padrao(
        ...     amplitudes=[400, 300, 200],
        ...     centros=[250, 500, 750],
        ...     sigmas=[10, 8, 12]
        ... )
    """
    # Converte para arrays NumPy
    amplitudes = np.asarray(amplitudes, dtype=float)
    centros = np.asarray(centros, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    
    # Valida que todos têm o mesmo tamanho
    n_picos = len(amplitudes)
    if not (len(centros) == n_picos and len(sigmas) == n_picos):
        raise ValueError(f"Todos os arrays devem ter o mesmo tamanho. "
                        f"amplitudes: {len(amplitudes)}, centros: {len(centros)}, "
                        f"sigmas: {len(sigmas)}")
    
    # Cria dicionário padrão
    picos_info = {
        'n_picos': n_picos,
        'indices': centros,  # Para compatibilidade com código existente
        'alturas': amplitudes,
        'centros': centros,  # Nome mais descritivo
        'sigmas': sigmas,
        'larguras': sigmas,  # Alias para compatibilidade
    }
    
    # Adiciona campos opcionais
    for key, value in kwargs.items():
        if value is not None:
            picos_info[key] = np.asarray(value, dtype=float)
    
    return picos_info


# =============================================================================
# CONVERSORES: Lista de Dicionários ↔ picos_info
# =============================================================================

def lista_dict_para_picos_info(lista_picos: List[Dict]) -> Dict:
    """
    Converte lista de dicionários (formato gerador_espectros) para picos_info.
    
    Args:
        lista_picos: Lista no formato [{'amp': 400, 'centro': 250, 'sigma': 10}, ...]
        
    Returns:
        dict: picos_info no formato padrão
        
    Exemplo:
        >>> lista = [
        ...     {'amp': 400, 'centro': 250, 'sigma': 10},
        ...     {'amp': 300, 'centro': 500, 'sigma': 8}
        ... ]
        >>> picos_info = lista_dict_para_picos_info(lista)
    """
    if not lista_picos:
        return criar_picos_info_padrao([], [], [])
    
    # Extrai valores, permitindo diferentes nomes de chaves
    amplitudes = []
    centros = []
    sigmas = []
    
    for pico in lista_picos:
        # Amplitude: tenta 'amp', 'amplitude', 'altura'
        amp = pico.get('amp') or pico.get('amplitude') or pico.get('altura')
        if amp is None:
            raise KeyError(f"Pico sem amplitude: {pico}")
        amplitudes.append(amp)
        
        # Centro: tenta 'centro', 'center', 'mu', 'posicao'
        centro = pico.get('centro') or pico.get('center') or pico.get('mu') or pico.get('posicao')
        if centro is None:
            raise KeyError(f"Pico sem centro: {pico}")
        centros.append(centro)
        
        # Sigma: tenta 'sigma', 'largura', 'width'
        sigma = pico.get('sigma') or pico.get('largura') or pico.get('width')
        if sigma is None:
            raise KeyError(f"Pico sem sigma: {pico}")
        sigmas.append(sigma)
    
    return criar_picos_info_padrao(amplitudes, centros, sigmas)


def picos_info_para_lista_dict(picos_info: Dict) -> List[Dict]:
    """
    Converte picos_info para lista de dicionários.
    
    Args:
        picos_info: Dicionário no formato padrão
        
    Returns:
        list: Lista de dicionários [{'amp': ..., 'centro': ..., 'sigma': ...}, ...]
        
    Exemplo:
        >>> lista = picos_info_para_lista_dict(picos_info)
        >>> print(lista[0])
        {'amp': 400.0, 'centro': 250.0, 'sigma': 10.0}
    """
    n_picos = picos_info['n_picos']
    lista_picos = []
    
    for i in range(n_picos):
        pico_dict = {
            'amp': float(picos_info['alturas'][i]),
            'centro': float(picos_info['centros'][i]),
            'sigma': float(picos_info['sigmas'][i])
        }
        lista_picos.append(pico_dict)
    
    return lista_picos


# =============================================================================
# CONVERSORES: picos_info ↔ Array 2D NumPy (para analise_metricas.py)
# =============================================================================

def picos_info_para_array(picos_info: Dict, formato='amp_centro_sigma') -> np.ndarray:
    """
    Converte picos_info para array 2D NumPy (para usar em analise_metricas.py).
    
    Args:
        picos_info: Dicionário no formato padrão
        formato: Ordem das colunas no array
                'amp_centro_sigma' → [[amp, centro, sigma], ...]
                'centro_amp_sigma' → [[centro, amp, sigma], ...]
        
    Returns:
        np.ndarray: Array 2D de shape (n_picos, 3)
        
    Exemplo:
        >>> array_params = picos_info_para_array(picos_info)
        >>> print(array_params.shape)
        (3, 3)  # 3 picos, 3 parâmetros cada
    """
    n_picos = picos_info['n_picos']
    
    if formato == 'amp_centro_sigma':
        # Formato: [amplitude, centro, sigma]
        array = np.column_stack([
            picos_info['alturas'],
            picos_info['centros'],
            picos_info['sigmas']
        ])
    elif formato == 'centro_amp_sigma':
        # Formato alternativo: [centro, amplitude, sigma]
        array = np.column_stack([
            picos_info['centros'],
            picos_info['alturas'],
            picos_info['sigmas']
        ])
    else:
        raise ValueError(f"Formato desconhecido: {formato}")
    
    return array


def array_para_picos_info(array: np.ndarray, formato='amp_centro_sigma') -> Dict:
    """
    Converte array 2D NumPy para picos_info.
    
    Args:
        array: Array 2D de shape (n_picos, 3)
        formato: Ordem das colunas no array
        
    Returns:
        dict: picos_info no formato padrão
        
    Exemplo:
        >>> array = np.array([[400, 250, 10], [300, 500, 8]])
        >>> picos_info = array_para_picos_info(array)
    """
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"Array deve ter shape (n_picos, 3). Recebido: {array.shape}")
    
    if formato == 'amp_centro_sigma':
        amplitudes = array[:, 0]
        centros = array[:, 1]
        sigmas = array[:, 2]
    elif formato == 'centro_amp_sigma':
        centros = array[:, 0]
        amplitudes = array[:, 1]
        sigmas = array[:, 2]
    else:
        raise ValueError(f"Formato desconhecido: {formato}")
    
    return criar_picos_info_padrao(amplitudes, centros, sigmas)


# =============================================================================
# FUNÇÕES UTILITÁRIAS
# =============================================================================

def validar_picos_info(picos_info: Dict) -> Tuple[bool, str]:
    """
    Valida se picos_info está no formato correto.
    
    Returns:
        tuple: (valido, mensagem_erro)
    """
    campos_obrigatorios = ['n_picos', 'alturas', 'centros', 'sigmas']
    
    for campo in campos_obrigatorios:
        if campo not in picos_info:
            return False, f"Campo obrigatório ausente: {campo}"
    
    n = picos_info['n_picos']
    for campo in ['alturas', 'centros', 'sigmas']:
        if len(picos_info[campo]) != n:
            return False, f"Campo '{campo}' tem tamanho incorreto: {len(picos_info[campo])} != {n}"
    
    return True, "OK"


def mesclar_picos_info(picos_verdadeiros: Dict, picos_detectados: Dict) -> Dict:
    """
    Mescla informações de picos verdadeiros e detectados para comparação.
    
    Esta função é útil para análise de métricas quando você quer comparar
    picos conhecidos (simulados) com picos detectados/ajustados.
    
    Args:
        picos_verdadeiros: picos_info dos parâmetros verdadeiros
        picos_detectados: picos_info dos parâmetros detectados/ajustados
        
    Returns:
        dict: Dicionário mesclado com campos '_verdadeiro' e '_detectado'
    """
    resultado = {
        'n_picos_verdadeiros': picos_verdadeiros['n_picos'],
        'n_picos_detectados': picos_detectados['n_picos'],
        
        'alturas_verdadeiras': picos_verdadeiros['alturas'],
        'alturas_detectadas': picos_detectados['alturas'],
        
        'centros_verdadeiros': picos_verdadeiros['centros'],
        'centros_detectados': picos_detectados['centros'],
        
        'sigmas_verdadeiras': picos_verdadeiros['sigmas'],
        'sigmas_detectadas': picos_detectados['sigmas'],
    }
    
    return resultado


def imprimir_picos_info(picos_info: Dict, titulo="Informações dos Picos"):
    """
    Imprime picos_info de forma legível.
    """
    print(f"\n{'='*60}")
    print(f"{titulo}")
    print(f"{'='*60}")
    print(f"Número de picos: {picos_info['n_picos']}")
    print(f"\n{'Pico':<6} {'Amplitude':>12} {'Centro':>12} {'Sigma':>12}")
    print("-" * 60)
    
    for i in range(picos_info['n_picos']):
        print(f"{i+1:<6} {picos_info['alturas'][i]:>12.2f} "
              f"{picos_info['centros'][i]:>12.2f} {picos_info['sigmas'][i]:>12.2f}")
    
    print("="*60)