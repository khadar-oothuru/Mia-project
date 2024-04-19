import cv2
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical

import matplotlib.pyplot as plt

image_directory = 'datasets/'
normal_images = os.listdir(image_directory + 'Normal/')
tumor_images = os.listdir(image_directory + 'Tumor/')
cyst_images = os.listdir(image_directory + 'Cyst/')
stone_images = os.listdir(image_directory + 'Stone/')

dataset = []
label = []

INPUT_SIZE = 64


for i, image_name in enumerate(normal_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(image_directory + 'Normal/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)


for i, image_name in enumerate(tumor_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(image_directory + 'Tumor/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

for i, image_name in enumerate(cyst_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(image_directory + 'Cyst/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(2)

# Load stone images
for i, image_name in enumerate(stone_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(image_directory + 'Stone/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(3)

dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

y_train = to_categorical(y_train, num_classes=4)  
y_test = to_categorical(y_test, num_classes=4)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(4))  
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=16,
          verbose=1, epochs=5,
          validation_data=(x_test, y_test),
          shuffle=False)


history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))




loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {accuracy}')


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()




model.save('trainedModel1.h5')
