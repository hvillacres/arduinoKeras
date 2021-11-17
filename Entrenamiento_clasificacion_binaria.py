import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras.utils import to_categorical

from arduinoKeras import weights
from arduinoKeras import layers
from arduinoKeras import scaling


import matplotlib.pyplot as plt

sizeClass = 10 #Cantidad de clases a clasificar

#Importar o generar conjuntos de datos
sizeInput = 18

P1 = np.load('P1.npy')
P2 = np.load('P2.npy')
P3 = np.load('P3.npy')
P4 = np.load('P4.npy')
P5 = np.load('P5.npy')
P6 = np.load('P6.npy')
P7 = np.load('P7.npy')
P8 = np.load('P8.npy')
P9 = np.load('P9.npy')
P10 = np.load('P10.npy')


#P = np.concatenate((P1, P2), axis=0)
#P = np.concatenate((P1, P2, P3, P4), axis=0)
P = np.concatenate((P1, P2, P3, P4, P5, P6, P7, P8, P9, P10), axis=0)

#T = np.concatenate((0*np.ones((P1.shape[0],1)),1*np.ones((P2.shape[0],1))), axis=0)
T = np.concatenate((0*np.ones((P1.shape[0],1)),
                    1*np.ones((P2.shape[0],1)),
                    2*np.ones((P3.shape[0],1)),
                    3*np.ones((P4.shape[0],1)),
                    4*np.ones((P5.shape[0],1)),
                    5*np.ones((P6.shape[0],1)),
                    6*np.ones((P7.shape[0],1)),
                    7*np.ones((P8.shape[0],1)),
                    8*np.ones((P9.shape[0],1)),
                    9*np.ones((P10.shape[0],1))), axis=0)

#Preprocesamiento de datos.

from sklearn.preprocessing import StandardScaler #pip install -U scikit-learn
scaler = StandardScaler().fit(P)
 
P = scaler.transform(P)


#one_hot_labels = to_categorical(T, num_classes=10)
one_hot_labels = to_categorical(T, num_classes=sizeClass)


#Dividir el conjuntos de datos en conjuntos de entrenamiento y prueba.
from sklearn.model_selection import train_test_split 

P_train, P_test, T_train, T_test = train_test_split(P,one_hot_labels, test_size=0.20, random_state=42)


#Establecer hiperparámetros de algoritmo (tasa de aprendizaje, épocas).
epochs=500
#lr=0.01
hiddenNodes = 5

#Definir la arquitectura de la red neuronal.
model = Sequential()
model.add(Dense(hiddenNodes, activation='relu', input_dim=sizeInput))
#model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

#Declara las funcion de pérdida.
loss='categorical_crossentropy'

#Optimizador.
optimizer = tf.keras.optimizers.Adam()
#optimizer= tf.keras.optimizers.SGD(learning_rate=lr)
#optimizer = tf.keras.optimizers.RMSprop()

model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['accuracy'])

#Entrenar el modelo

#history = model.fit(P, T,epochs=epochs, verbose=1)
history = model.fit(P_train, T_train,epochs=epochs, verbose=1,validation_split=0.2)
plt.xlabel('Epocas') 
plt.ylabel('Pérdida')
plt.plot(history.epoch, np.array(history.history['loss']),'b', label='Train Loss')
plt.plot(history.epoch, np.array(history.history['val_loss']), 'r', label = 'Val Loss')
plt.legend()
plt.grid()


#Evaluar el modelo
test_loss, test_acc = model.evaluate(P_test, T_test,verbose=1)
print('Exactitud de prueba: ', test_acc)


# Extraer valores pesos,normalizacion y la arquitectura de la red para Arduino
weights(model.layers,3)           
scaling(scaler,3)
layers(model.layers)


plt.show()
