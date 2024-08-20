import numpy as np

class RegresionLineal:
    
    """
    Implementación de un modelo de regresión lineal desde cero.

    Esta clase implementa un modelo de regresión lineal utilizando el método de
    mínimos cuadrados ordinarios para ajustar una línea recta a los datos. 

    Atributos:
        __coef (numpy.ndarray): Vector de coeficientes de la regresión.
        __intercept (float): Término independiente de la regresión.

    Métodos:
        fit(X, y): Ajusta el modelo a los datos de entrenamiento.
        predict(X): Realiza predicciones sobre nuevos datos.
    """
    
    def __init__(self):
        
        """
        Inicializa los atributos de la clase.

        Inicializa los coeficientes y el término independiente a None.
        """
        self.__coef = None
        self.__intercept = None
        
    def fit(self, X, y):
        """
        Ajusta el modelo de regresión lineal a los datos de entrenamiento.

        Utiliza el método de mínimos cuadrados ordinarios para calcular los
        coeficientes de la regresión.

        Args:
            X (numpy.ndarray): Matriz de características de forma (n_samples, n_features).
            y (numpy.ndarray): Vector de etiquetas de forma (n_samples,).
        """
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        self.__coef = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
        self.__intercept = self.__coef[0]
        self.__coef = self.__coef[1:]
        
    def predict(self, X):
        """
        Realiza predicciones sobre nuevos datos.

        Utiliza los coeficientes y el término independiente calculados en el método
        fit para realizar predicciones.

        Args:
            X (numpy.ndarray): Matriz de características de forma (n_samples, n_features).

        Returns:
            numpy.ndarray: Vector de predicciones de forma (n_samples,).
        """
        return np.dot(X, self.__coef) + self.__intercept
        