import random
from abc import ABC, abstractmethod

class OptimizadorBase(ABC):
    """
    Clase base abstracta para la optimización de hiperparámetros.
    Implementa una búsqueda aleatoria (Random Search).
    """
    def __init__(self, espacio_busqueda, n_iteraciones=10):
        """
        Args:
            espacio_busqueda (dict): Diccionario donde las llaves son nombres de parámetros
                                    y los valores son listas [opciones] o tuplas (min, max).
            n_iteraciones (int): Cantidad de combinaciones a probar.
        """
        self.espacio_busqueda = espacio_busqueda
        self.n_iteraciones = n_iteraciones
        self.mejor_config = None
        self.mejor_resultado = -float('inf')

    def _muestrear_configuracion(self):
        """Genera una configuración aleatoria basada en el espacio de búsqueda."""
        config = {}
        for param, valores in self.espacio_busqueda.items():
            if isinstance(valores, list):
                config[param] = random.choice(valores)
            elif isinstance(valores, tuple) and len(valores) == 2:
                if isinstance(valores[0], int) and isinstance(valores[1], int):
                    config[param] = random.randint(valores[0], valores[1])
                else:
                    config[param] = random.uniform(valores[0], valores[1])
        return config

    @abstractmethod
    def evaluar_configuracion(self, config):
        """Debe retornar un score numérico donde mayor es mejor."""
        pass

    def optimizar(self):
        """Ejecuta el ciclo de optimización."""
        print(f"Iniciando optimización por {self.n_iteraciones} iteraciones...")
        for i in range(self.n_iteraciones):
            config = self._muestrear_configuracion()
            try:
                resultado = self.evaluar_configuracion(config)
                print(f"Iteración {i+1}: Score = {resultado:.6f} | Config = {config}")
                
                if resultado > self.mejor_resultado:
                    self.mejor_resultado = resultado
                    self.mejor_config = config
            except Exception as e:
                print(f"Error en iteración {i+1} con config {config}: {e}")
        
        print("\n--- Optimización Finalizada ---")
        print(f"Mejor Score: {self.mejor_resultado:.6f}")
        print(f"Mejor Configuración: {self.mejor_config}")
        return self.mejor_config, self.mejor_resultado