import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


def calcular_hipervolumen_2d(frente_objetivos, punto_ref):
    """Calcula el área dominada por el frente respecto a un punto de referencia."""
    
    frente = np.array(frente_objetivos)
    frente = frente[np.argsort(frente[:, 0])] # Ordenar por f1 (riesgo)
    
    hv = 0.0
    ultimo_f2 = punto_ref[1]
    
    for p in frente:
        f1, f2 = p[0], p[1]
        if f1 >= punto_ref[0] or f2 >= punto_ref[1]: 
            continue
        hv += (punto_ref[0] - f1) * (ultimo_f2 - f2)
        ultimo_f2 = f2
    return hv


def exportar_carteras(mejor_pob_nsga, mejor_obj_nsga, mejores_pesos_abc, mejores_obj_abc, lambdas, carpeta_salida):
    """Estructura y guarda los CSV con los pesos y métricas financieras de cada cartera."""

    NOMBRES_ACTIVOS = ['COPEC', 'CTC-A', 'CAP', 'COLBUN', 'ENDESA', 'ENTEL']
    tasa_libre_riesgo = 0.0
    
    # NSGA-II
    df_nsga = pd.DataFrame(mejor_pob_nsga, columns=NOMBRES_ACTIVOS)
    df_nsga['Varianza'] = mejor_obj_nsga[:, 0]
    df_nsga['Retorno'] = -mejor_obj_nsga[:, 1]
    df_nsga['Desviacion'] = np.sqrt(df_nsga['Varianza'])
    df_nsga['Sharpe'] = (df_nsga['Retorno'] - tasa_libre_riesgo) / df_nsga['Desviacion']

    columnas_nsga = ['Retorno', 'Varianza', 'Desviacion', 'Sharpe'] + NOMBRES_ACTIVOS
    df_nsga[columnas_nsga].to_csv(os.path.join(carpeta_salida, 'detalles_carteras_nsga_ii.csv'), index=False)

    # ABC
    df_abc = pd.DataFrame(mejores_pesos_abc, columns=NOMBRES_ACTIVOS)
    df_abc['Lambda'] = lambdas
    df_abc['Varianza'] = mejores_obj_abc[:, 0]
    df_abc['Retorno'] = -mejores_obj_abc[:, 1]
    df_abc['Desviacion'] = np.sqrt(df_abc['Varianza'])
    df_abc['Sharpe'] = (df_abc['Retorno'] - tasa_libre_riesgo) / df_abc['Desviacion']

    columnas_abc = ['Lambda', 'Sharpe', 'Retorno', 'Varianza', 'Desviacion'] + NOMBRES_ACTIVOS
    df_abc[columnas_abc].to_csv(os.path.join(carpeta_salida, 'detalles_carteras_abc.csv'), index=False)


def exportar_hipervolumen(mejor_obj_nsga, mejores_obj_abc, punto_ref_global, carpeta_salida):
    """Calcula y exporta el hipervolumen global del mejor frente obtenido."""
    
    hv_final_nsga = calcular_hipervolumen_2d(mejor_obj_nsga, punto_ref_global)
    hv_final_abc = calcular_hipervolumen_2d(mejores_obj_abc, punto_ref_global)
    
    df_hv = pd.DataFrame({
        'Algoritmo': ['NSGA-II', 'ABC'],
        'Hipervolumen': [hv_final_nsga, hv_final_abc],
        'Punto_Ref_f1 (Riesgo)': [punto_ref_global[0], punto_ref_global[0]],
        'Punto_Ref_f2 (RetNeg)': [punto_ref_global[1], punto_ref_global[1]]
    })
    df_hv.to_csv(os.path.join(carpeta_salida, 'hipervolumen_robusto.csv'), index=False)


def exportar_test_wilcoxon(semillas, abc_fronts_por_semilla, pto_ref_temp, dist_hv_nsga, dist_sharpe_nsga, carpeta_salida):
    """Procesa las distribuciones y ejecuta el test estadístico de Wilcoxon.
    
    Para el Hipervolumen: Calculamos el área del frente completo de 
    NSGA-II vs. el área formada por los 11 puntos que generó ABC.
    

    Para el Índice de Sharpe: Buscamos el portafolio con el Sharpe máximo dentro del 
    frente de NSGA-II y lo comparamos contra el Sharpe máximo de los 11 portafolios de ABC.
    """

    dist_hv_abc, dist_sharpe_abc = [], []
    
    # Extraer métricas de ABC por cada semilla para emparejar con NSGA
    for semilla in semillas:
        frente_abc = np.array(abc_fronts_por_semilla[semilla])
        dist_hv_abc.append(calcular_hipervolumen_2d(frente_abc, pto_ref_temp))
        sharpes_abc = (-frente_abc[:, 1] - 0.0) / np.sqrt(frente_abc[:, 0])
        dist_sharpe_abc.append(np.max(sharpes_abc))

    # Test estadístico
    _, pval_hv = wilcoxon(dist_hv_nsga, dist_hv_abc)
    _, pval_sh = wilcoxon(dist_sharpe_nsga, dist_sharpe_abc)

    df_wilcoxon = pd.DataFrame({
        'Metrica': ['Hipervolumen', 'Sharpe_Maximo'],
        'P_Valor': [pval_hv, pval_sh],
        'Significativo_alpha_0.05': [pval_hv < 0.05, pval_sh < 0.05]
    })
    df_wilcoxon.to_csv(os.path.join(carpeta_salida, 'estadisticas_wilcoxon.csv'), index=False)


def generar_graficos(mejor_obj_nsga, mejores_obj_abc, carpeta_salida):
    """Genera y guarda los gráficos comparativos de frentes de Pareto."""
    
    # Gráfico 1: Frente de Pareto Comparativo
    plt.figure(figsize=(10, 6))
    riesgo_nsga_std = np.sqrt(mejor_obj_nsga[:, 0]) * 100
    retorno_nsga_pos = -mejor_obj_nsga[:, 1] * 100
    riesgo_abc_std = np.sqrt(mejores_obj_abc[:, 0]) * 100
    retorno_abc_pos = -mejores_obj_abc[:, 1] * 100

    plt.scatter(riesgo_nsga_std, retorno_nsga_pos, color='blue', alpha=0.6, label='NSGA-II (Mejor Frente)')
    plt.plot(riesgo_abc_std, retorno_abc_pos, color='red', marker='o', linestyle='--', label='ABC (Mejores Puntos)')
    plt.title('Comparación de Frentes: NSGA-II vs ABC (mejor de las semillas)')
    plt.xlabel('Riesgo (Desviación Estándar %)')
    plt.ylabel('Retorno Esperado (%)')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.savefig(os.path.join(carpeta_salida, 'grafico_comparativo_frentes.png'), dpi=300)
    plt.close()

    # Gráfico 2: NSGA-II puro
    plt.figure(figsize=(8, 5))
    plt.scatter(mejor_obj_nsga[:, 0], mejor_obj_nsga[:, 1], color='purple', alpha=0.7)
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    plt.title('NSGA-II: frente de pareto')
    plt.xlabel('f1: Varianza de la cartera (min)')
    plt.ylabel('f2: Retorno negativo (min)')
    plt.grid(alpha=0.5)
    plt.savefig(os.path.join(carpeta_salida, 'grafico_nsga_f1_vs_f2.png'), dpi=300)
    plt.close()