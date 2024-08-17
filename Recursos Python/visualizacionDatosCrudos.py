# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 19:26:38 2023

@author: alda7
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
from pandas.plotting import scatter_matrix

filename = 'C:\\Users\\alda7\\Desktop\\mew data\\final_features_2.csv'
#namesF = ['picosR', 'amplitudR', 'potencia', 'frecDominante', 'energiaWavelet', 'entropiaShannonW', 'clase']
#data = pd.read_csv(filename), names=namesF)

data = pd.read_csv(filename)
print(data)

# print(data.shape) #Muestra la dimension
#
# print(data.head(20)) #Muestra las primeras n filas del df 
# print(data.dtypes) #Imprime el tipo de datos de cada Rasgo

# pd.set_option('display.max_columns', None) #Muestra todas las columnas
# pd.set_option('display.max_rows', None) #Muestra todas las filas
# pd.set_option('display.width', 100)
# pd.set_option('display.precision', 3)
# print(data.describe()) #Muestra una tabla de analisis estadistico, se puede observar sesgos
# data.describe().to_csv('descriptive_stats.csv', index=True) #Guardar la tabla en CSV
# print(data.groupby('clase').size()) #Ver la cantidad de datos para cada clase, para evitar el desbalanceo de datos

# Correlaciones, (Observar la interaccion entre variables)
# correlation = data.corr(method='pearson')
# print(correlation)
# print(data.skew()) #Sesgo, si es - sesgo a la izq, si es + sesgo a la der, si es 0-1 casi no tiene sesgo (gaussiano "prefecto")

# =============================================================================
#   Visualización Univariable
# =============================================================================

# Histogramas con Seaborn

#name = ["picosW","auc","potencia","frecDominante","energiaWavelet","entropiaShannonW","media","std","cv","asimetria","numInflex","clase"]

# f, axes = plt.subplots(3, 4, figsize = (14, 14))

# sns.distplot(data["picosW"], ax = axes[0,0])
# # sns.distplot(data["auc"], ax = axes[0,1])
# # sns.distplot(data["potencia"], ax = axes[0,2])
# sns.distplot(data["frecDominante"], ax = axes[0,3])
# sns.distplot(data["energiaWavelet"], ax = axes[1,0])
# sns.distplot(data["entropiaShannonW"], ax = axes[1,1])
# # sns.distplot(data["media"], ax = axes[1,2])
# # sns.distplot(data["std"], ax = axes[1,3])
# # sns.distplot(data["cv"], ax = axes[2,0])
# sns.distplot(data["asimetria"], ax = axes[2,1])
# sns.distplot(data["numInflex"], ax = axes[2,2])
# sns.distplot(data["clase"], ax = axes[2,3])


f, axes = plt.subplots(2, 3, figsize = (16, 8))
sns.distplot(data["QRS_width"], ax = axes[0,0])
sns.distplot(data["T_wave_amplitude"], ax = axes[0,1])
sns.distplot(data["Shannon_entropy"], ax = axes[0,2])
sns.distplot(data["Wavelet_entropy"], ax = axes[1,0])
sns.distplot(data["Energy_scale"], ax = axes[1,1])
f.suptitle('Histogramas',fontsize=20, y=0.9, va='bottom')

# Histogramas con matplotlib

# fig = plt.figure(figsize = (10,10))
# ax = fig.gca()
# data.hist(ax = ax)
# plt.show()

# Densidad matplotlib

# fig = plt.figure(figsize=(16,16))
# ax = fig.gca()
# data.plot(ax = ax, kind = 'density', subplots = True, layout = (3,3), sharex = False)
# plt.show()

# Densidad Seaborn

# f, axes = plt.subplots(3, 4, figsize = (14, 14))
# sns.distplot(data["picosW"], hist = False, rug = True, ax = axes[0,0])
# # sns.distplot(data["auc"], hist = False, rug = True, ax = axes[0,1])
# # sns.distplot(data["potencia"], hist = False, rug = True, ax = axes[0,2])
# sns.distplot(data["frecDominante"], hist = False, rug = True, ax = axes[0,3])
# sns.distplot(data["energiaWavelet"], hist = False, rug = True, ax = axes[1,0])
# sns.distplot(data["entropiaShannonW"], hist = False, rug = True, ax = axes[1,1])
# # sns.distplot(data["media"], hist = False, rug = True, ax = axes[1,2])
# # sns.distplot(data["std"], hist = False, rug = True, ax = axes[1,3])
# # sns.distplot(data["cv"], hist = False, rug = True, ax = axes[2,0])
# sns.distplot(data["asimetria"], hist = False, rug = True, ax = axes[2,1])
# sns.distplot(data["numInflex"], hist = False, rug = True, ax = axes[2,2])
# sns.distplot(data["clase"], hist = False, rug = True, ax = axes[2,3])

f, axes = plt.subplots(2, 3, figsize = (16, 8))
f.suptitle('Densidad',fontsize=20, y=0.9, va='bottom')
sns.distplot(data["QRS_width"], hist = False, rug = True, ax = axes[0,0])
sns.distplot(data["T_wave_amplitude"], hist = False, rug = True, ax = axes[0,1])
sns.distplot(data["Shannon_entropy"], hist = False, rug = True, ax = axes[0,2])
sns.distplot(data["Wavelet_entropy"], hist = False, rug = True, ax = axes[1,0])
sns.distplot(data["Energy_scale"], hist = False, rug = True, ax = axes[1,1])
# Grafica Boxplot matplotlib

# fig = plt.figure(figsize=(13,13))
# ax = fig.gca()
# data.plot(ax = ax, kind = 'box', subplots = True, layout = (3,3), sharex = False)
# plt.show()

# # Grafica Boxplot Seaborn

# f, axes = plt.subplots(3, 4, figsize = (14, 14))
# sns.boxplot(data["picosW"], ax = axes[0,0])
# # sns.boxplot(data["auc"], ax = axes[0,1])
# # sns.boxplot(data["potencia"], ax = axes[0,2])
# sns.boxplot(data["frecDominante"], ax = axes[0,3])
# sns.boxplot(data["energiaWavelet"], ax = axes[1,0])
# sns.boxplot(data["entropiaShannonW"], ax = axes[1,1])
# # sns.boxplot(data["media"], ax = axes[1,2])
# # sns.boxplot(data["std"], ax = axes[1,3])
# # sns.boxplot(data["cv"], ax = axes[2,0])
# sns.boxplot(data["asimetria"], ax = axes[2,1])
# sns.boxplot(data["numInflex"], ax = axes[2,2])
# sns.boxplot(data["clase"], ax = axes[2,3])

f, axes = plt.subplots(2, 3, figsize = (16, 8))
f.suptitle('Boxplot',fontsize=20, y=0.9, va='bottom')
sns.boxplot(data["QRS_width"], ax = axes[0,0])
sns.boxplot(data["T_wave_amplitude"], ax = axes[0,1])
sns.boxplot(data["Shannon_entropy"], ax = axes[0,2])
sns.boxplot(data["Wavelet_entropy"], ax = axes[1,0])
sns.boxplot(data["Energy_scale"], ax = axes[1,1])
# =============================================================================
#   Visualización Multivariable
# =============================================================================

# Matriz de correlación matplotlib (No grafica no se por qué xd)

# correlations = data.corr()
# fig = plt.figure(figsize=(15,15))
# ax = fig.add_subplot(111)
# cax = ax.matshow(correlations, vmin=1, vmax=1)
# fig.colorbar(cax)
# ticks = np.arange(0, 7, 1)
# ax.set_xticks(ticks)
# ax.set_yticks(ticks)
# ax.set_xticklabels(namesF)
# ax.set_yticklabels(namesF)
# plt.show()

# Matriz de correlación Seaborn (Si funciona)

correlation = data.corr()
print(correlation)
plt.figure(figsize=(10,10))
ax = sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='viridis')
plt.title("Correlation Matrix")
plt.show()

# Matriz de dispersión matplotlib

# plt.rcParams["figure.figsize"] = [14,10]
# scatter_matrix(data)
# plt.show()

# Matriz de dispersión Seaborn
sns.pairplot(data)
#sns.suptitle('Scattering Matrix',fontsize=20, y=0.9, va='bottom')
# Matriz de dispersión Por Clase
sns.pairplot(data, hue="Class", diag_kind="hist")

# Boxplot por Clase

# plt.figure(1)
# plt.subplots(figsize=(20,20))
# plt.subplot(421)
# sns.boxplot(x='clase', y='picosR', data=data)
# plt.title("Picos R")
# plt.grid(True)

# plt.subplot(422)
# sns.boxplot(x='clase', y='amplitudR', data=data)
# plt.title("Amplitud R")
# plt.grid(True)

# plt.subplot(423)
# sns.boxplot(x='clase', y='potencia', data=data)
# plt.title("Potencia")
# plt.grid(True)

# plt.subplot(424)
# sns.boxplot(x='clase', y='frecDominante', data=data)
# plt.title("Frecuencia Dominante")
# plt.grid(True)

# plt.subplot(425)
# sns.boxplot(x='clase', y='energiaWavelet', data=data)
# plt.title("Energía Wavelet")
# plt.grid(True)

# plt.subplot(426)
# sns.boxplot(x='clase', y='entropiaShannonW', data=data)
# plt.title("Entropía Shannon")
# plt.grid(True)

# plt.show()

# PDF
# sns.displot(data, x="picosW", hue="clase", kind="kde")
# sns.displot(data, x="auc", hue="clase", kind="kde")
# sns.displot(data, x="potencia", hue="clase", kind="kde")
# sns.displot(data, x="frecDominante", hue="clase", kind="kde")
# sns.displot(data, x="energiaWavelet", hue="clase", kind="kde")
# sns.displot(data, x="entropiaShannonW", hue="clase", kind="kde")
# sns.displot(data, x="media", hue="clase", kind="kde")
# sns.displot(data, x="std", hue="clase", kind="kde")
# sns.displot(data, x="cv", hue="clase", kind="kde")
# sns.displot(data, x="asimetria", hue="clase", kind="kde")
# sns.displot(data, x="numInflex", hue="clase", kind="kde")

# f, axes = plt.subplots(3, 4, figsize=(14, 14))
# columns_to_plot = ["picosW", "auc", "potencia", "frecDominante", "energiaWavelet", "entropiaShannonW", "media", "std", "cv", "asimetria", "numInflex"]
# for ax, column in zip(axes.flatten(), columns_to_plot):
#     sns.kdeplot(data=data, x=column, hue="clase", ax=ax)
    
f, axes = plt.subplots(2, 3, figsize=(16, 8))

#columns_to_plot = ["Interval_RR", "Peak_amplitude", "QRS_width", "T_wave_amplitude", "Dominant_frequency"]
columns_to_plot = ["QRS_width", "T_wave_amplitude", "Shannon_entropy", "Wavelet_entropy", "Energy_scale"]
for ax, column in zip(axes.flatten(), columns_to_plot):
    sns.kdeplot(data=data, x=column, hue="Class", ax=ax)
f.suptitle('Gráficas PDF',fontsize=20, y=0.9, va='bottom')
# scaler = MinMaxScaler(feature_range=(0,1))
# rescaled_X = scaler.fit_transform(X)
# np.set_printoptions(precision=3)
# print(names)
# print(rescaled_X[0:5,:])