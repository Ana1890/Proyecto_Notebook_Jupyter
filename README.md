Proyecto de Regresión Lineal - Adoptá un Junior

Este proyecto es una implementación inicial de un modelo de regresión lineal. El objetivo principal es participar del proceso de selección de Adoptá un Junior.

Descripción

El proyecto incluye un modelo de regresión lineal implementado desde cero, así como funciones de preprocesamiento de datos y visualización de resultados. La clase Graficador permite visualizar las relaciones entre las características de los datos y las predicciones del modelo, generando gráficos de regresión para cada característica.

Este modelo es una versión inicial y, aunque funcional, está diseñado para ser expandido y mejorado en el futuro.

Estructura del Proyecto

modules/: Contiene las clases principales del proyecto:
RegresionLineal: Implementación del modelo de regresión lineal.
Graficador: Herramienta para visualizar los resultados de la regresión.
CustomScaler: Clase para estandarizar características antes del entrenamiento del modelo.
main.py: Script principal que carga los datos, realiza el preprocesamiento, entrena el modelo y genera las visualizaciones.

Requisitos

Para ejecutar este proyecto, se necesitan las siguientes librerías de Python:

numpy
pandas
matplotlib
scikit-learn

Uso

Cargar los datos en el archivo main.py.
Ejecutar el script main.py para preprocesar los datos, entrenar el modelo y visualizar los resultados.

Futuras Mejoras

Este proyecto es solo un punto de partida. Algunas ideas para futuras mejoras incluyen:

Implementación de nuevas técnicas de preprocesamiento.
Mejora de la visualización de los resultados.
Adaptación del modelo para manejar datos más complejos.
Integración de validación cruzada y ajuste de hiperparámetros.

Contribuciones

Dado que este es un proyecto en desarrollo, las contribuciones y sugerencias son bienvenidas. Si tienes alguna idea o mejora, no dudes en hacer un fork del repositorio y abrir un pull request.
