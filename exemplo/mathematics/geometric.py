# geometric.py
import numpy as np

def pitagoras(a, b, calc='hip'):
    '''
    Calcula o teorema de Pitágoras para encontrar a hipotenusa ou um cateto.
    '''
    if calc == 'hip':
        return np.sqrt((a**2) + (b**2))
    elif calc == 'cat':
        return np.sqrt((max(a, b)**2) - (min(a, b)**2))