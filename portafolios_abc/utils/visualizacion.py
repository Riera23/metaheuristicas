import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

class Visualizacion:
    
    @staticmethod
    def formatear_resultados_tabla_4(resultados):
        """
        Imprime los resultados en consola.
        Muestra tanto la Varianza (el número que usó el paper) como el Desvío Real.
        """
        print("\nTabla 4\nResultados solo con posiciones largas")
        print("="*135)
        print(f"{'':<10} | {'Retorno':<8} | {'Montos a invertir (Activos)':^53} | {'Varianza':<10} | {'Desvío':<10} | {'Lambda':<6}")
        print(f"{'Cartera':<10} | {'cartera':<8} | {'COPEC':<8} {'CTC-A':<8} {'CAP':<8} {'COLBUN':<8} {'ENDESA':<8} {'ENTEL':<8} | {'':<10} | {'':<10} | {'':<6}")
        print("-" * 135)
        
        for i, res in enumerate(resultados):
            nombre_cartera = f"Cartera {i+1}"
            
            # Formateo a porcentajes
            retorno_str = f"{(res['retorno'] * 100):.3f}%"
            varianza_str = f"{(res['riesgo'] * 100):.3f}%"
            desvio_str = f"{(np.sqrt(res['riesgo']) * 100):.3f}%"
            
            # Formateo de los pesos a porcentajes
            pesos_str = " ".join([f"{(p * 100):>7.3f}%" for p in res['pesos']])
            
            print(f"{nombre_cartera:<10} | {retorno_str:<8} | {pesos_str} | {varianza_str:<10} | {desvio_str:<10} | {res['lambda']:<6.1f}")
        print("="*135)

    @staticmethod
    def guardar_resultados_csv(resultados, nombre_archivo):
        if not os.path.exists('resultados'):
            os.makedirs('resultados')
            
        datos = []
        for i, res in enumerate(resultados):
            fila = {
                'Cartera': f"Cartera {i+1}",
                'Retorno_cartera': res['retorno'],
                'COPEC': res['pesos'][0],
                'CTC-A': res['pesos'][1],
                'CAP': res['pesos'][2],
                'COLBUN': res['pesos'][3],
                'ENDESA': res['pesos'][4],
                'ENTEL': res['pesos'][5],
                'Varianza_cartera': res['riesgo'],
                'Desviacion': np.sqrt(res['riesgo']),
                'Lambda': res['lambda']
            }
            datos.append(fila)
            
        df = pd.DataFrame(datos)
        ruta = os.path.join('resultados', nombre_archivo)
        df.to_csv(ruta, index=False)

    @staticmethod
    def graficar_frontera_pareto(resultados, nombre_archivo="figura_2_abc.png"):
        """
        Genera el gráfico de puntos simulando los portafolios (p1, p2, etc.) de la Figura 2.
        Nota: El eje X siempre debe dibujarse con la Desviación Real (no la varianza).
        """
        if not os.path.exists('resultados'):
            os.makedirs('resultados')
            
        riesgos = [np.sqrt(res['riesgo']) * 100 for res in resultados] 
        retornos = [res['retorno'] * 100 for res in resultados]        
        
        plt.figure(figsize=(12, 6))
        
        # Ploteamos como línea punteada suave la tendencia (nuestra frontera)
        plt.plot(riesgos, retornos, color='gray', linestyle='--', alpha=0.5, label='Tendencia ABC')
        
        # Ploteamos los puntos discretos (p1, p2, p3...) usando diferentes marcadores
        marcadores = ['o', '^', 's', 'x', 'D', 'v', 'p', '*', '+', 'h', '>']
        for i in range(len(resultados)):
            etiqueta = f"p{i+1} (Lambda={resultados[i]['lambda']})"
            marcador_actual = marcadores[i % len(marcadores)]
            plt.scatter(riesgos[i], retornos[i], marker=marcador_actual, s=100, label=etiqueta)
            
        plt.title('Frontera Eficiente y Portafolios (ABC - Solo Posiciones Largas)')
        plt.xlabel('Riesgo (Desviación Estándar %)')
        plt.ylabel('Rentabilidad Esperada (%)')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        ruta = os.path.join('resultados', nombre_archivo)
        plt.savefig(ruta, dpi=300)
        plt.close()