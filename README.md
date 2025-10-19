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

## 📚 Como Instalar as Bibliotecas

Para garantir que todas as bibliotecas necessárias sejam instaladas corretamente, siga os passos abaixo:

1. **Certifique-se de ter o Python instalado**  
    Verifique se o Python está instalado em sua máquina. Recomendamos a versão 3.8 ou superior. Para verificar, execute o comando:
    ```bash
    python --version
    ```
    ou
    ```bash
    python3 --version
    ```

2. **Crie e ative um ambiente virtual (opcional, mas recomendado)**  
    Criar um ambiente virtual ajuda a isolar as dependências do projeto. Para criar e ativar um ambiente virtual, use os comandos abaixo:

    No Windows:
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

    No macOS/Linux:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Instale as dependências listadas no arquivo `requirements.txt`**  
    Certifique-se de estar no diretório onde o arquivo `requirements.txt` está localizado e execute o comando:
    ```bash
    pip install -r requirements.txt
    ```

4. **Verifique se as bibliotecas foram instaladas corretamente**  
    Após a instalação, você pode verificar se as bibliotecas foram instaladas executando:
    ```bash
    pip list
    ```

Agora, todas as dependências necessárias para o projeto estarão configuradas e prontas para uso.
