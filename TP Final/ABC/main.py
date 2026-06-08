import sys
import os

# Aseguramos que Python encuentre nuestros módulos locales de forma dinámica
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from pruebaABC import pruebaABC
from evaluacionABC import evaluacionABC

def main():
    print("OPTIMIZACIÓN DE PORTAFOLIOS CON COLONIA ARTIFICIAL DE ABEJAS (ABC)")
    
    pruebaABC()
    
    print("\n" + "-"*70 + "\n")
    
    evaluacionABC()
    
    print("Resultados físicos guardados en la carpeta ./resultados/:")
    print("  - resultados_tabla4.csv (Datos crudos listos para Excel/Pandas)")
    print("  - frontera_pareto.png (Gráfico de la curva de eficiencia)")

if __name__ == "__main__":
    main()