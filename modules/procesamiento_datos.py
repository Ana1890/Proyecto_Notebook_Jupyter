from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE


class CustomScaler(BaseEstimator, TransformerMixin):
    """
    Clase para estandarizar datos usando StandardScaler de scikit-learn.

    Atributos:
        scaler (StandardScaler): Objeto StandardScaler de scikit-learn.
    """
    def __init__(self):
        
        """
        Inicializa el objeto CustomScaler.

        Crea una instancia de StandardScaler para realizar la estandarización.
        """
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        
        """
        Entrena el escalador con los datos de entrada.

        Calcula los parámetros de estandarización (media y desviación estándar)
        utilizando los datos de entrada X.

        Args:
            X (numpy.ndarray): Matriz de características.
            y (numpy.ndarray, optional): Vector de etiquetas (ignorado).

        Returns:
            CustomScaler: Objeto CustomScaler entrenado.
        """
        self.scaler.fit(X)
        return self

    def transform(self, X):
        """
        Transforma los datos de entrada aplicando la estandarización.

        Aplica la estandarización previamente calculada a los datos de entrada X.

        Args:
            X (numpy.ndarray): Matriz de características.

        Returns:
            numpy.ndarray: Matriz de características estandarizadas.
        """
        return self.scaler.transform(X)
    
class FeatureSelector:
    """
    Clase para seleccionar características usando Recursive Feature Elimination (RFE).

    Esta clase encapsula el algoritmo RFE de scikit-learn para seleccionar las
    características más importantes según un modelo base.

    Atributos:
        model (object): Modelo base utilizado para la selección de características.
        num_features (int): Número de características a seleccionar.
        selector (RFE): Objeto RFE entrenado.
    """
    def __init__(self, model, num_features):
        
        """
        Inicializa el objeto FeatureSelector.

        Args:
            model (object): Modelo base utilizado para la selección de características.
            num_features (int): Número de características a seleccionar.
        """
        self.model = model
        self.num_features = num_features
        self.selector = None

    def fit(self, X, y):
        
        """
        Entrena el selector de características con los datos de entrada.

        Crea un objeto RFE utilizando el modelo base y el número de características
        deseado, y luego lo entrena con los datos de entrada X y y.

        Args:
            X (numpy.ndarray): Matriz de características.
            y (numpy.ndarray): Vector de etiquetas.

        Returns:
            FeatureSelector: Objeto FeatureSelector entrenado.
        """
        self.selector = RFE(self.model, n_features_to_select=self.num_features)
        self.selector = self.selector.fit(X, y)

    def transform(self, X):
        
        """
        Transforma los datos de entrada aplicando la selección de características.

        Aplica la selección de características previamente entrenada a los datos
        de entrada X, devolviendo solo las características seleccionadas.

        Args:
            X (numpy.ndarray): Matriz de características.

        Returns:
            numpy.ndarray: Matriz de características seleccionadas.
        """
        return self.selector.transform(X)