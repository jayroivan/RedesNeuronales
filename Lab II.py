# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 18:48:38 2021

Dataset Diferente al de la clase 11

@author: frank
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

#
# Lectura y visualización del set de datos
#
datos = pd.read_csv('Fish.csv', sep=",")
print(datos)

#grafica de peces/peso
datos.plot.scatter(x='Species', y='Weight')
plt.xlabel('Especie (peces)')
plt.ylabel('Peso (gramos)')
plt.show()

x = datos['Species'].values
y = datos['Weight'].values

#grafica de peces/altura
datos.plot.scatter(x='Species', y='Height')
plt.xlabel('Especie (peces)')
plt.ylabel('Altura (centimetros)')
plt.show()

x = datos['Species'].values
y = datos['Height'].values


#grafica de peces/ancho
datos.plot.scatter(x='Species', y='Width')
plt.xlabel('Especie (peces)')
plt.ylabel('Ancho (centimetros)')
plt.show()

x = datos['Species'].values
y = datos['Width'].values

#
# Modelo en Keras
#

#
#creamos un diccionario con los valores originales y los valores de reemplazo
a = {"Bream" : 0, "Roach" : 1, "Whitefish": 2, "Parkki": 3, "Perch": 4, "Pike": 5, "Smelt": 6}
#utilizamos un lambda para el reemplazo 
datos["Species"] = datos["Species"].apply(lambda x:a[x])
x = datos['Species'].values
#

np.random.seed(2)			# Para reproducibilidad del entrenamiento

input_dim = 1
output_dim = 1
modelo = Sequential()
modelo.add(Dense(output_dim, input_dim=input_dim, activation='linear'))

# Definición del método de optimización (gradiente descendiente), con una
# tasa de aprendizaje de 0.0004 y una pérdida igual al error cuadrático
# medio

sgd = SGD(lr=0.0004)
modelo.compile(loss='mse', optimizer=sgd)

# Imprimir en pantalla la información del modelo
modelo.summary()

#
# Entrenamiento: realizar la regresión lineal
#

# 40000 iteraciones y todos los datos de entrenamiento (29) se usarán en cada
# iteración (batch_size = 29)

num_epochs = 40000
batch_size = x.shape[0]
history = modelo.fit(x, y, epochs=num_epochs, batch_size=batch_size, verbose=0)

#
# Visualizar resultados del entrenamiento
#

# Imprimir los coeficientes "w" y "b"
capas = modelo.layers[0]
w, b = capas.get_weights()
print('Parámetros: w = {:.1f}, b = {:.1f}'.format(w[0][0],b[0]))

# Graficar el resultado de la regresión

y_regr = modelo.predict(x)
plt.subplot(1, 2, 2)
plt.scatter(x,y)
plt.plot(x,y_regr,'r')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Datos originales y regresión lineal')
plt.show()