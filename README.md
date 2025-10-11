# Simulação e Análise de Espectros de Energia Nuclear

## 📊 Descrição
Projeto de computação científica para simular espectros de detectores nucleares e analisar sinais espectrais. Implementa detecção clássica de picos e um módulo experimental para detecção via IA.

## 👥 Autores
- Matheus Novello (236511)
- João Victor Pomiglio de Oliveira (250391)  
- Arthur Cesar (245730)
- André de Moraes Salvi (231323)

## 🎯 Objetivos

### Objetivo Geral
Desenvolver sistema completo de simulação e análise de espectros de energia capaz de detectar automaticamente picos gaussianos em sinais ruidosos, realizar ajustes matemáticos precisos e avaliar quantitativamente a qualidade das reconstruções.

### Objetivos Específicos
- Implementar simulador de espectros com fundo, múltiplos picos e ruído controlado
- Desenvolver detecção automática de picos por análise de proeminência e largura
- Implementar ajuste não-linear global e individual (fallback)
- Calcular métricas de qualidade (MSE, RMSE, resíduos) e gerar visualizações comparativas
- Validar capacidade de recuperar parâmetros conhecidos dentro das incertezas estatísticas
- (Opcional) Implementar detecção de picos via algoritmos de machine learning

## 🛠️ Algoritmos e Estruturas de Dados

### Algoritmos Principais
- **find_peaks (SciPy)**: Detecção de picos por proeminência e largura
- **Filtro Savitzky-Golay**: Suavização preservando características
- **Levenberg-Marquardt (TRF)**: Otimização não-linear via curve_fit
- **Mersenne Twister**: Geração de números pseudo-aleatórios
- **Mínimos Quadrados**: Ajuste de parâmetros minimizando χ²
- **Random Forest** (Opcional): Modelo de IA para detecção de picos

### Estruturas de Dados
- **Arrays NumPy 1D**: Espectros, eixos, resíduos
- **Arrays NumPy 2D**: Parâmetros de picos, matriz de covariância
- **Lista de Dicionários**: Resultados do ajuste por pico
- **Tuplas**: Parâmetros de fundo exponencial
- **Dicionários**: Métricas de qualidade globais

## 📚 Bibliotecas

### Principais Dependências
```python
numpy >= 1.20        # Operações numéricas e arrays
scipy >= 1.7         # Algoritmos científicos (find_peaks, curve_fit)
matplotlib >= 3.3    # Visualização de dados
scikit-learn         # Algoritmos de machine learning (opcional)