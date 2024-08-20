from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from modules.regresion_lineal import RegresionLineal
from modules.graficos import Graficador
from modules.procesamiento_datos import CustomScaler
import pandas as pd
from sklearn.impute import SimpleImputer

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

# Seleccionar características relevantes
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

graficador = Graficador()

graficador.graficar_regresion(X_scaled, Y, y_pred, 
                              color='green', 
                              xlabel_0='Adults (scaled)', 
                              xlabel_1='Stays in Weekend Nights (scaled)', 
                              ylabel='lead_time', 
                              titulo_0='Regresión del lead_time vs Adults (scaled)', 
                              titulo_1='Regresión del lead_time vs Stays in Weekend Nights (scaled)')

