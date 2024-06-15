import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from Utils.dataLoader import loadImagesFromFolder, splitDataset


# pip install opencv-python tensorflow keras matplotlib numpy pandas scikit-learn
#command to install all the libraries


# path to the anormal 
imagesASCUSLink = "Data/cells/ASC-US/"
imagesLSILLink = "Data/cells/LSIL/"
imagesASCHLink = "Data/cells/ASC-H/"
imagesHSILLink = "Data/cells/HSIL/"
imagesCarcinomaLink = "Data/cells/carcinoma/"

imagesAnormalCelulaeLink = [
    imagesASCUSLink,
    imagesLSILLink,
    imagesASCHLink,
    imagesHSILLink,
    imagesCarcinomaLink,
]


# path to negative 
imagesNegativaLink = "Data/cells/Negativa/"

# where the images are stored
negativeImagesStored = []
AnormalImagesStored = []

# populating ImagesAnormal
for folder in imagesAnormalCelulaeLink:
    imagesAnormal = loadImagesFromFolder(folder,AnormalImagesStored)

#Populating ImagesNegative
imagesNegative = loadImagesFromFolder(imagesNegativaLink, negativeImagesStored)

# Splitting the AnormalImagesStored dataset
X_train_anormal, X_val_anormal, X_test_anormal = splitDataset(AnormalImagesStored)

# Splitting the negativeImagesStored dataset
X_train_negative, X_val_negative, X_test_negative = splitDataset(negativeImagesStored)

# Definir a arquitetura da CNN
model = Sequential()

# Primeira camada convolucional
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(MaxPooling2D((2, 2)))

# Segunda camada convolucional
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Terceira camada convolucional
model.add(Conv2D(128, (3, 3), activation='relu')) 
model.add(MaxPooling2D((2, 2)))

# Camada de flatten
model.add(Flatten())

# Camada densa totalmente conectada
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

# Camada de saída
model.add(Dense(1, activation='sigmoid'))

# Compilar o modelo
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Resumo do modelo
model.summary()

# Preparar os dados para o treinamento
X_train = np.concatenate((X_train_anormal, X_train_negative), axis=0)
y_train = np.concatenate((np.ones(len(X_train_anormal)), np.zeros(len(X_train_negative))), axis=0)

X_val = np.concatenate((X_val_anormal, X_val_negative), axis=0)
y_val = np.concatenate((np.ones(len(X_val_anormal)), np.zeros(len(X_val_negative))), axis=0)

X_test = np.concatenate((X_test_anormal, X_test_negative), axis=0)
y_test = np.concatenate((np.ones(len(X_test_anormal)), np.zeros(len(X_test_negative))), axis=0)

# Treinar o modelo
history = model.fit(
    X_train, y_train,
    epochs=20,
    validation_data=(X_val, y_val),
    batch_size=32
)


# Fazer previsões no conjunto de teste
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Calcular métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Acurácia no conjunto de teste: {accuracy}')
print(f'Precisão no conjunto de teste: {precision}')
print(f'Revocação no conjunto de teste: {recall}')
print('Relatório de classificação:')
print(report)

# Plotar gráficos de acurácia e perda durante o treinamento
plt.figure(figsize=(12, 6))

# Acurácia
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Perda
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
#cv2.waitKey(0)  # Press any key to close the window
