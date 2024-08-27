"""Es buena práctica poner aquí de qué trata el código.
Qué es lo que se quiere predecir, qué métodos se incluyen."""
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from modules.regresion_lineal import RegresionLineal
from modules.graficos import Graficador
from modules.procesamiento_datos import CustomScaler
import pandas as pd
from sklearn.impute import SimpleImputer

"""Para hacer un código modular, todo esto se puede poner en una función.
Entonces es posible poder llamar a la función ya sea desde aquí o desde otro script/notebook."""
# Cargar los datos
data = pd.read_csv('data/hotel_bookings.csv')

# Verificar si hay valores no numéricos 
if data['lead_time'].dtype == 'object':
    print("La columna 'lead_time' contiene valores no numéricos.")
    data['lead_time'] = pd.to_numeric(data['lead_time'], errors='coerce')

# Imputar valores faltantes 
imputer_lead_time = SimpleImputer(strategy='most_frequent')
data['lead_time'] = imputer_lead_time.fit_transform(data[['lead_time']])

# Imputar valores faltantes en columnas numéricas
numeric_cols = data.select_dtypes(include=['number']).columns
imputer_numeric = SimpleImputer(strategy='mean')
data[numeric_cols] = imputer_numeric.fit_transform(data[numeric_cols])
"""Habría que tener cuidado que no impute sobre el target, ya que ahora es numérico (entiendo que el target
es lead_time).
Si imputamos sobre target, estamos agregando data falsa al modelo."""

# Seleccionar características relevantes
"""Seguro que estas variables salen de un análisis exploratorio, pero para aprender puede ser bueno armar un pequeño
informe de por qué se eligieron estas variables. Por ejemplo, si se corroboró que tenían correlación con la variable target.
"""
X = data[['adults', 'stays_in_weekend_nights']].values
Y = data['lead_time'].values

# Incorporar CustomScaler para estandarizar los datos
scaler = CustomScaler()
X_scaled = scaler.fit_transform(X)

# Crear y entrenar el modelo con las características estandarizadas
model = RegresionLineal()
model.fit(X_scaled, Y)

# Realizar predicciones
y_pred = model.predict(X_scaled)

# Calcular el error cuadrático medio
mse = mean_squared_error(Y, y_pred)
print("Error cuadrático medio:", mse)
"""Seguro lo vayan a ver en el curso, pero por lo general se reportan las métricas y las funciones de pérdida 
sobre el conjunto de validación o test.
Si se reportan sobre el mismo conjunto que se entrenó, no se podría ver el poder predictivo del modelo.
"""
graficador = Graficador()

graficador.graficar_regresion(X_scaled, Y, y_pred, 
                              color='green', 
                              xlabel_0='Adults (scaled)', 
                              xlabel_1='Stays in Weekend Nights (scaled)', 
                              ylabel='lead_time', 
                              titulo_0='Regresión del lead_time vs Adults (scaled)', 
                              titulo_1='Regresión del lead_time vs Stays in Weekend Nights (scaled)')
"""Habría que ver los parámetros que se le está pasando al método de graficar regresión, porque parece que sólo acepta 
una etiqueta por eje (x o y, así como el título).

Debido a esto, me doy cuenta cómo X_scaled tiene dos columnas, debería pasarse 1 sóla columna, porque el gráfico
acepta sólo un vector por eje."""
