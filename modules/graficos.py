import matplotlib.pyplot as plt
import pandas as pd

class Graficador:
    
    """
    Clase para crear gráficos de regresión lineal.

    Esta clase permite graficar la relación entre las características de una matriz X 
    y un vector de etiquetas y, junto con la línea de regresión obtenida a partir de 
    las predicciones del modelo.

    Métodos:
        graficar_regresion(X, y, y_pred, **kwargs): Grafica los datos originales y la 
        línea de regresión para cada característica de X.
    """

    def __init__(self):
        """
        Inicializa un objeto Graficador.

        No se requiere ninguna configuración inicial específica.
        """
        pass

    def graficar_regresion(self, X, y, y_pred, **kwargs):
        """
        Grafica los datos originales y la línea de regresión para cada característica de X.

        Args:
            X (np.ndarray): Matriz de características.
            y (np.ndarray): Vector de etiquetas.
            y_pred (np.ndarray): Vector de predicciones.
            **kwargs: Otros argumentos para personalizar el gráfico (e.g., 'color', 'titulo', 'xlabel', 'ylabel').
        """
        num_features = X.shape[1]
        
        for i in range(num_features):
            plt.figure(figsize=(8, 6))
            plt.scatter(X[:, i], y, label='Datos')
            plt.plot(X[:, i], y_pred, color=kwargs.get('color', 'red'), label='Regresión lineal')
            plt.xlabel(kwargs.get(f'xlabel_{i}', f'Característica {i+1}'))
            plt.ylabel(kwargs.get('ylabel', 'y'))
            plt.title(kwargs.get(f'titulo_{i}', f'Regresión Lineal - Característica {i+1}'))
            plt.legend()
            plt.show()

