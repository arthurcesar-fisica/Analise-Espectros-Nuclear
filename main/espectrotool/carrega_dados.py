# carrega_dados.py

import pandas as pd
import numpy as np
import json
import os

def carrega_xlsx(caminho_arquivo):
    """
    Carrega dados de espectro de um arquivo Excel (.xlsx).
    """
    try:
        # Lê o arquivo Excel
        df = pd.read_excel(caminho_arquivo)
        
        # Verifica se as colunas necessárias existem
        colunas_necessarias = ['Energia', 'Contagens']
        colunas_disponiveis = df.columns.tolist()
        
        # Tenta encontrar colunas com nomes similares
        energia_col = None
        contagens_col = None
        
        for col in colunas_disponiveis:
            col_lower = str(col).lower()
            if any(x in col_lower for x in ['energia', 'channel', 'canal', 'energy', 'x']):
                energia_col = col
            elif any(x in col_lower for x in ['contagen', 'count', 'intensity', 'y', 'signal']):
                contagens_col = col
        
        # Se não encontrou automaticamente, usa as primeiras colunas
        if energia_col is None and len(df.columns) >= 2:
            energia_col = df.columns[0]
            print(f"Aviso: Usando primeira coluna '{energia_col}' para energia")
        
        if contagens_col is None and len(df.columns) >= 2:
            if energia_col == df.columns[0]:
                contagens_col = df.columns[1]
            else:
                contagens_col = df.columns[0]
            print(f"Aviso: Usando coluna '{contagens_col}' para contagens")
        
        if energia_col is None or contagens_col is None:
            raise ValueError(f"Não foi possível identificar as colunas de energia e contagens. Colunas disponíveis: {colunas_disponiveis}")
        
        # Extrai os dados
        eixo_energia = df[energia_col].values.astype(float)
        dados_espectro = df[contagens_col].values.astype(float)
        
        # Remove possíveis valores NaN
        mask = ~(np.isnan(eixo_energia) | np.isnan(dados_espectro))
        eixo_energia = eixo_energia[mask]
        dados_espectro = dados_espectro[mask]
        
        print(f"✓ Dados carregados do arquivo: {os.path.basename(caminho_arquivo)}")
        print(f"  Forma dos dados: {len(eixo_energia)} pontos")
        print(f"  Faixa de energia: {eixo_energia[0]:.1f} a {eixo_energia[-1]:.1f}")
        print(f"  Faixa de contagens: {dados_espectro.min():.1f} a {dados_espectro.max():.1f}")
        
        return eixo_energia, dados_espectro
        
    except Exception as e:
        print(f"✗ Erro ao carregar arquivo XLSX {caminho_arquivo}: {e}")
        raise

def carrega_json(caminho_arquivo):
    """
    Carrega dados de espectro de um arquivo JSON.
    """
    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            dados = json.load(f)
        
        # Verifica o formato do JSON
        if isinstance(dados, list):
            # Formato: lista de objetos [{"Energia": x, "Contagens": y}, ...]
            energias = []
            contagens = []
            for item in dados:
                # Procura por chaves de energia
                energia_val = None
                contagens_val = None
                
                for key, value in item.items():
                    key_lower = str(key).lower()
                    if any(x in key_lower for x in ['energia', 'channel', 'canal', 'energy', 'x']):
                        energia_val = float(value)
                    elif any(x in key_lower for x in ['contagen', 'count', 'intensity', 'y', 'signal']):
                        contagens_val = float(value)
                
                if energia_val is not None and contagens_val is not None:
                    energias.append(energia_val)
                    contagens.append(contagens_val)
            
            if not energias:
                raise ValueError("Não foi possível extrair dados de energia e contagens do JSON")
            
            eixo_energia = np.array(energias)
            dados_espectro = np.array(contagens)
            
        elif isinstance(dados, dict):
            # Formato: dicionário com arrays separados
            energia_key = None
            contagens_key = None
            
            for key in dados.keys():
                key_lower = str(key).lower()
                if any(x in key_lower for x in ['energia', 'channel', 'canal', 'energy', 'x']):
                    energia_key = key
                elif any(x in key_lower for x in ['contagen', 'count', 'intensity', 'y', 'signal']):
                    contagens_key = key
            
            if energia_key and contagens_key:
                eixo_energia = np.array(dados[energia_key], dtype=float)
                dados_espectro = np.array(dados[contagens_key], dtype=float)
            else:
                # Tenta usar as primeiras chaves se disponíveis
                keys = list(dados.keys())
                if len(keys) >= 2:
                    energia_key = keys[0]
                    contagens_key = keys[1]
                    eixo_energia = np.array(dados[energia_key], dtype=float)
                    dados_espectro = np.array(dados[contagens_key], dtype=float)
                    print(f"Aviso: Usando chaves '{energia_key}' e '{contagens_key}' para energia e contagens")
                else:
                    raise ValueError("Formato JSON não reconhecido - necessárias pelo menos 2 chaves")
        else:
            raise ValueError("Formato JSON não suportado")
        
        # Remove possíveis valores NaN
        mask = ~(np.isnan(eixo_energia) | np.isnan(dados_espectro))
        eixo_energia = eixo_energia[mask]
        dados_espectro = dados_espectro[mask]
        
        print(f"✓ Dados carregados do arquivo: {os.path.basename(caminho_arquivo)}")
        print(f"  Forma dos dados: {len(eixo_energia)} pontos")
        print(f"  Faixa de energia: {eixo_energia[0]:.1f} a {eixo_energia[-1]:.1f}")
        print(f"  Faixa de contagens: {dados_espectro.min():.1f} a {dados_espectro.max():.1f}")
        
        return eixo_energia, dados_espectro
        
    except Exception as e:
        print(f"✗ Erro ao carregar arquivo JSON {caminho_arquivo}: {e}")
        raise

def carrega_espectro(caminho_arquivo):
    """
    Carrega dados de espectro automaticamente baseado na extensão do arquivo.
    """
    if not os.path.exists(caminho_arquivo):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")
    
    extensao = os.path.splitext(caminho_arquivo)[1].lower()
    
    if extensao == '.xlsx':
        return carrega_xlsx(caminho_arquivo)
    elif extensao == '.json':
        return carrega_json(caminho_arquivo)
    else:
        raise ValueError(f"Formato de arquivo não suportado: {extensao}")

def lista_arquivos_dados(pasta="dados_simulados"):
    """
    Lista todos os arquivos de dados disponíveis na pasta especificada.
    """
    if not os.path.exists(pasta):
        print(f"Pasta '{pasta}' não encontrada")
        return []
    
    arquivos_suportados = []
    for arquivo in os.listdir(pasta):
        if arquivo.lower().endswith(('.xlsx', '.json')):
            caminho_completo = os.path.join(pasta, arquivo)
            arquivos_suportados.append(caminho_completo)
    
    if arquivos_suportados:
        print(f"✓ Encontrados {len(arquivos_suportados)} arquivos de dados em '{pasta}':")
        for arquivo in arquivos_suportados:
            print(f"  - {os.path.basename(arquivo)}")
    else:
        print(f"✗ Nenhum arquivo de dados encontrado em '{pasta}'")
    
    return arquivos_suportados

def carrega_multiplos_arquivos(pasta="dados_simulados"):
    """
    Carrega todos os arquivos de dados de uma pasta.
    
    Retorna:
    --------
    dict: Dicionário com {nome_arquivo: (eixo_energia, dados_espectro)}
    """
    arquivos = lista_arquivos_dados(pasta)
    dados_carregados = {}
    
    for arquivo in arquivos:
        try:
            nome = os.path.basename(arquivo)
            eixo, dados = carrega_espectro(arquivo)
            dados_carregados[nome] = (eixo, dados)
        except Exception as e:
            print(f"✗ Erro ao carregar {arquivo}: {e}")
    
    return dados_carregados