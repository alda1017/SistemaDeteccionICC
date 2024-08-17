# -*- coding: utf-8 -*-
"""
Created on Fri May 24 19:18:12 2024

@author: alda7
"""

import pickle
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Cargar el archivo CSV con las características
df = pd.read_csv('C:\\Users\\alda7\\Desktop\\mew data\\final_features_2.csv')

# Separar las etiquetas
Y = df['Class'].values

# Eliminar la columna 'Class' del DataFrame para dejar solo las características
X = df.drop(['Class'], axis=1).values

# Mezclar y dividir los datos en conjunto de entrenamiento y prueba
X, Y = shuffle(X, Y)
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, shuffle=True)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

class0 = np.sum(y_test == 0)  # SANOS
class1 = np.sum(y_test == 1)  # ENFERMOS

# Definir y entrenar el modelo K-Nearest Neighbors
n_neighbors = 5  # Puedes ajustar este parámetro
clf = KNeighborsClassifier(n_neighbors=n_neighbors)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

# Evaluar el modelo
modelKNNScore = clf.score(x_test, y_test)
print('Model Test Accuracy: %1.4f' % (modelKNNScore))

modelKNNAccuracyScore = accuracy_score(y_test, y_pred)
print('Model Prediction Accuracy: %1.4f' % (modelKNNAccuracyScore))

# Matriz de confusión
modelKNNConfusionMatrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
print('\nConfusion matrix: Rows Actual class, Columns Predicted class.')
print('Class_0 = %d, Class_1 = %d. \nTotal test samples %d.\n' % (class0, class1, y_test.shape[0]))
print(modelKNNConfusionMatrix)


# =============================================================================
#     # PDF Feature 1
# =============================================================================
xFeature = "QRS_width"
sns.displot(df, x=xFeature, kind="kde")
plt.title("PDF Feature 1")

# =============================================================================
#     # # PDF Feature 2
# =============================================================================
yFeature = "T_wave_amplitude"
sns.displot(df, x=yFeature, kind="kde")
plt.title("PDF Feature 2")

# =============================================================================
#     # # PDF Feature 3
# =============================================================================
otherFeature = "Shannon_entropy"
sns.displot(df, x=otherFeature, kind="kde")
plt.title("PDF Feature 3")

# =============================================================================
#     # # PDF Feature 4
# =============================================================================
otherFeature = "Wavelet_entropy"
sns.displot(df, x=otherFeature, kind="kde")
plt.title("PDF Feature 4")

# =============================================================================
#     # # PDF Feature 5
# =============================================================================
otherFeature = "Energy_scale"
sns.displot(df, x=otherFeature, kind="kde")
plt.title("PDF Feature 5")


# Guardar el modelo en disco
# filename = 'KNN_model.sav'
# pickle.dump(clf, open(filename, 'wb'))

# ===================
#     Computar Sensibilidad, Especificidad, Prevalencia y VPP
# ===================
for i in range(2):
    TP = modelKNNConfusionMatrix[i, i]
    FP = np.sum(modelKNNConfusionMatrix[:, i]) - TP
    FN = np.sum(modelKNNConfusionMatrix[i, :]) - TP
    TN = np.sum(modelKNNConfusionMatrix) - (TP + FP + FN)

    # Sensibilidad, Especificidad, VPP, VPN
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    prevalence = (TP + FN) / (TP + FP + TN + FN)
    VPP = TP / (TP + FP)

    # Imprimir las métricas
    print(f"\nIndexes for class {i}:")
    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")
    print(f"Prevalence: {prevalence}")
    print(f"Positive Predictive value: {VPP}\n")
    
    
conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])

# Configurando la visualización con Seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', cbar_kws={'label': 'Escala'})
plt.title('Matriz de Confusión de Clasificación')
plt.xlabel('Etiqueta Predicha')
plt.ylabel('Etiqueta Real')

# Definir las marcas para los ejes x e y
tick_marks = np.arange(len(['Normal', 'ICC'])) + 0.5

plt.xticks(tick_marks, ['Normal', 'ICC'])  # Ajustar según las etiquetas
plt.yticks(tick_marks, ['Normal', 'ICC'], rotation=0)
plt.savefig('confusion_matrix.png')  # Guardar la imagen
plt.show()




