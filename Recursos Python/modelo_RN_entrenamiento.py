import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adamax

# Cargar el archivo CSV con las caracteristicas
df = pd.read_csv('C:\\Users\\alda7\\Desktop\\mew data\\final_features_2.csv')

# Separar las etiquetas
Y = df['Class'].values

# Eliminar la columna Class del DataFrame
X = df.drop(['Class'], axis=1).values

# Mezclar y dividir los datos en conjunto de entrenamiento y prueba
X, Y = shuffle(X, Y)
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, shuffle=True)


# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)


# Definir el modelo
model = Sequential([
    Dense(1024, input_shape=(x_train.shape[1],), activation='leaky_relu'),
    # BatchNormalization(),
    # Dropout(0.7),
    Dense(1024, activation='leaky_relu'),
    # BatchNormalization(),
    # Dropout(0.6),
    Dense(512, activation='leaky_relu'),
    Dense(1, activation='sigmoid')
])

optimizer_adamax = Adamax(learning_rate=0.005)

# Compilar el modelo
model.compile(loss='binary_crossentropy', optimizer=optimizer_adamax, metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test), verbose=1, callbacks=[early_stopping])

# Evaluar el modelo
loss, accuracy = model.evaluate(x_test, y_test)
print('Model Test Accuracy: %1.4f' % accuracy)

# Hacer predicciones
y_pred = (model.predict(x_test) > 0.5).astype("int32")

# Calcular la precision del modelo
modelNNAccuracyScore = accuracy_score(y_test, y_pred)
print('Model Prediction Accuracy: %1.4f' % (modelNNAccuracyScore))

# Matriz de confusion y reporte de clasificacion para el tipo de latido
print("Confusion Matrix:")
cm_beat = confusion_matrix(y_test, y_pred)
print(cm_beat)
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=1))

# Matriz de confusion
modelNNConfusionMatrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
print('\nConfusion matrix: Rows Actual class, Columns Predicted class.')
print('Class_0 = %d, Class_1 = %d. \nTotal test samples %d.\n' % (np.sum(y_test == 0), np.sum(y_test == 1), y_test.shape[0]))
print(modelNNConfusionMatrix)

# Guardar el modelo
model.save('RN_model.h5')

# Graficas de rendimiento para el modelo de tipo de latido
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Beat Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Funcion para graficar la matriz de confusion
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
# Nombres de las clases
class_names = ['Normal', 'ICC']

# Graficar la matriz de confusion
plt.figure(figsize=(10, 10))
plot_confusion_matrix(cm_beat, class_names, title='Confusion Matrix for Classification')
plt.show()

# ===================
#     Computar Sensibilidad, Especificidad, Prevalencia y VPP
# ===================
for i in range(2):
    TP = modelNNConfusionMatrix[i, i]
    FP = np.sum(modelNNConfusionMatrix[:, i]) - TP
    FN = np.sum(modelNNConfusionMatrix[i, :]) - TP
    TN = np.sum(modelNNConfusionMatrix) - (TP + FP + FN)

    # Sensibilidad, Especificidad, VPP, VPN
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    prevalence = (TP + FN) / (TP + FP + TN + FN)
    VPP = TP / (TP + FP)

    # Imprimir las metricas
    print(f"\nIndexes for class {i}:")
    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")
    print(f"Prevalence: {prevalence}")
    print(f"Positive Predictive value: {VPP}\n")
