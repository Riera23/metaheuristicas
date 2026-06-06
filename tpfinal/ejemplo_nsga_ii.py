from nsga_ii import NSGAII
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def ejecutar_ejemplo():
    # 1. Configuración de activos (6 activos)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META']
    print(f"Descargando datos para: {tickers}...")
    
    # Descargar datos históricos (2 años para tener una ventana W considerable)
    datos = yf.download(tickers, period='2y', interval='1d', auto_adjust=True)['Close']
    
    # Calcular retornos diarios logarítmicos
    retornos_diarios = np.log(datos / datos.shift(1)).dropna()
    
    # Estimación de parámetros (Ri y Σij)
    retornos_esperados = retornos_diarios.mean().values
    matriz_covarianza = retornos_diarios.cov().values
    
    print("Datos procesados correctamente.\n")
    print("Parámetros del Problema:")
    print(f"Retornos esperados promedio: {retornos_esperados}")
    print("-" * 50)

    # 2. Configuración e instanciación de NSGA-II
    # Usamos parámetros estándar para asegurar convergencia en el ejemplo
    n_activos = len(tickers)
    optimizador = NSGAII(
        n_activos=n_activos,
        retornos_esperados=retornos_esperados,
        matriz_covarianza=matriz_covarianza,
        tam_poblacion=50,
        n_generaciones=100,
        prob_crossover=0.9,
        prob_mutacion=0.2,
        eta_c=20,
        eta_m=20
    )

    # 3. Ejecución del algoritmo
    print("Ejecutando NSGA-II...")
    pesos_pareto, objetivos_pareto = optimizador.ejecutar()
    print(f"Algoritmo finalizado. Se encontraron {len(pesos_pareto)} soluciones en el frente de Pareto.")

    # 4. Mostrar resultados
    # Transformar f2 (negativo) a retorno positivo para la visualización
    riesgos = objetivos_pareto[:, 0]
    retornos = -objetivos_pareto[:, 1]

    # Crear un DataFrame para visualizar las mejores soluciones (ej. las primeras 5)
    df_resultados = pd.DataFrame(pesos_pareto, columns=tickers)
    df_resultados['Riesgo (Var)'] = riesgos
    df_resultados['Retorno Esperado'] = retornos
    
    print("\nMuestra de soluciones en el Frente de Pareto:")
    print(df_resultados.head().to_string(index=False))

    # 5. Visualización del Frente de Pareto
    plt.figure(figsize=(10, 6))
    plt.scatter(riesgos, retornos, c='red', marker='o', label='Frontera de Pareto')
    
    # Graficar puntos de activos individuales para referencia
    for i, ticker in enumerate(tickers):
        var_individual = matriz_covarianza[i, i]
        ret_individual = retornos_esperados[i]
        plt.scatter(var_individual, ret_individual, marker='x', label=f'Solo {ticker}')

    plt.title('Frente de Pareto: Riesgo vs Retorno (Optimización de Portafolio)')
    plt.xlabel('Riesgo (Varianza f1)')
    plt.ylabel('Retorno Esperado (-f2)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

if __name__ == "__main__":
    ejecutar_ejemplo()
