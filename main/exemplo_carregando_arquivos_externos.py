import os 
from espectrotool import (gera_espectro, mostra_espectro, salva_espectro,
                         analise_completa_espectro, carrega_espectro, 
                         lista_arquivos_dados)

from espectrotool.analise_metricas import QualityMetrics


# Opção 1: Gerar novo espectro
print("=== GERANDO NOVO ESPECTRO ===")
eixo_energia, dados_espectro, picos_info, magnitude_ruido = gera_espectro()
salva_espectro(eixo_energia, dados_espectro, 'espectro_exemplo')
mostra_espectro(eixo_energia, dados_espectro)

# Opção 2: Carregar espectro existente
print("\n=== CARREGANDO ESPECTRO EXISTENTE ===")
arquivos_disponiveis = lista_arquivos_dados("dados_simulados")

if arquivos_disponiveis:
    # Usa o primeiro arquivo disponível
    arquivo_escolhido = arquivos_disponiveis[0]
    print(f"Carregando: {arquivo_escolhido}")
    
    try:
        eixo_energia, dados_espectro = carrega_espectro(arquivo_escolhido)
        
        # Visualiza os dados carregados        
        mostra_espectro(eixo_energia, dados_espectro)
        
        # Parâmetros customizados para detecção
        parametros_deteccao = {
            'altura_minima': 100,
            'distancia_minima': 5,
            'proeminencia': 40,
            'largura_minima': 2,
            'suavizar': True
        }

        # Parâmetros customizados para ajuste
        parametros_ajuste = {
            'tipo_pico': 'gaussiana',
            'tipo_fundo': 'exponencial'
        }

        # Executar análise completa
        resultados = analise_completa_espectro(
            eixo_energia, 
            dados_espectro,
            parametros_deteccao=parametros_deteccao,
            parametros_ajuste=parametros_ajuste
        )

        #Relatório de Métricas 
        validator = QualityMetrics()
        results = validator.check_requirements(
            y_true=dados_espectro,
            y_pred=resultados["resultado_ajuste"]["y_ajustado"],
            noise_level= magnitude_ruido
        )
        validator.generate_report(results)
        
    except Exception as e:
        print(f"Erro ao processar arquivo: {e}")
else:
    print("Nenhum arquivo de dados encontrado. Um novo espectro foi gerado e salvo.")

print("\n=== PROCESSO CONCLUÍDO ===")