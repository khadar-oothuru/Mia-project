# import cv2
# import os
# import tensorflow as tf
# from PIL import Image
# import numpy as np
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import normalize, to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
# import matplotlib.pyplot as plt
# import multiprocessing
# from mpi4py import MPI
# from sklearn.preprocessing import LabelEncoder

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()

# def load_images(image_directory, images, label_value, input_size):
#     images_list = []
#     for image_name in images:
#         if image_name.split('.')[1] == 'jpg':
#             image = cv2.imread(os.path.join(image_directory, label_value, image_name))
#             image = Image.fromarray(image, 'RGB')
#             image = image.resize((input_size, input_size))
#             images_list.append(np.array(image))
#     return images_list, [label_value] * len(images_list)

# image_directory = 'datasets/'
# INPUT_SIZE = 64

# normal_images = os.listdir(os.path.join(image_directory, 'Normal'))
# tumor_images = os.listdir(os.path.join(image_directory, 'Tumor'))
# cyst_images = os.listdir(os.path.join(image_directory, 'Cyst'))
# stone_images = os.listdir(os.path.join(image_directory, 'Stone'))

# dataset = []
# label = []

# normal_data, normal_labels = load_images(image_directory, normal_images, 'Normal', INPUT_SIZE)
# tumor_data, tumor_labels = load_images(image_directory, tumor_images, 'Tumor', INPUT_SIZE)
# cyst_data, cyst_labels = load_images(image_directory, cyst_images, 'Cyst', INPUT_SIZE)
# stone_data, stone_labels = load_images(image_directory, stone_images, 'Stone', INPUT_SIZE)

# dataset.extend(normal_data)
# dataset.extend(tumor_data)
# dataset.extend(cyst_data)
# dataset.extend(stone_data)

# label.extend(normal_labels)
# label.extend(tumor_labels)
# label.extend(cyst_labels)
# label.extend(stone_labels)

# # Encode labels into integer format
# label_encoder = LabelEncoder()
# label_encoder.fit(['Normal', 'Tumor', 'Cyst', 'Stone'])
# y_encoded = label_encoder.transform(label)

# # Split data and labels
# x_train, x_test, y_train_encoded, y_test_encoded = train_test_split(dataset, y_encoded, test_size=0.2, random_state=0)

# x_train = normalize(np.array(x_train), axis=1)
# x_test = normalize(np.array(x_test), axis=1)

# y_train_categorical = to_categorical(y_train_encoded, num_classes=4)
# y_test_categorical = to_categorical(y_test_encoded, num_classes=4)

# model = Sequential()
# model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Flatten())
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dense(4))
# model.add(Activation('softmax'))

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# # Train the model with MPI parallelism enabled
# history = model.fit(x_train, y_train_categorical, batch_size=16, epochs=5, verbose=1, validation_data=(x_test, y_test_categorical), shuffle=False)

# loss, accuracy = model.evaluate(x_test, y_test_categorical)
# print(f'Test Accuracy: {accuracy}')

# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.plot(history.history['loss'], label='Training Loss')
# plt.title('Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.title('Training Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.tight_layout()
# plt.show()

# model.save('trainedModel1.h5')

import cv2
import os
import tensorflow as tf
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import normalize, to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
import matplotlib.pyplot as plt
from mpi4py import MPI
from sklearn.preprocessing import LabelEncoder

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def load_images(image_directory, images, label_value, input_size):
    images_list = []
    for i, image_name in enumerate(images):
        if i % size == rank:
            if image_name.split('.')[1] == 'jpg':
                image = cv2.imread(os.path.join(image_directory, label_value, image_name))
                image = Image.fromarray(image, 'RGB')
                image = image.resize((input_size, input_size))
                images_list.append(np.array(image))
    return images_list, [label_value] * len(images_list)

image_directory = 'datasets/'
INPUT_SIZE = 64

normal_images = os.listdir(os.path.join(image_directory, 'Normal'))
tumor_images = os.listdir(os.path.join(image_directory, 'Tumor'))
cyst_images = os.listdir(os.path.join(image_directory, 'Cyst'))
stone_images = os.listdir(os.path.join(image_directory, 'Stone'))

dataset = []
label = []

normal_data, normal_labels = load_images(image_directory, normal_images, 'Normal', INPUT_SIZE)
tumor_data, tumor_labels = load_images(image_directory, tumor_images, 'Tumor', INPUT_SIZE)
cyst_data, cyst_labels = load_images(image_directory, cyst_images, 'Cyst', INPUT_SIZE)
stone_data, stone_labels = load_images(image_directory, stone_images, 'Stone', INPUT_SIZE)

local_dataset = []
local_dataset.extend(normal_data)
local_dataset.extend(tumor_data)
local_dataset.extend(cyst_data)
local_dataset.extend(stone_data)

local_label = []
local_label.extend(normal_labels)
local_label.extend(tumor_labels)
local_label.extend(cyst_labels)
local_label.extend(stone_labels)

dataset = comm.gather(local_dataset, root=0)
label = comm.gather(local_label, root=0)

if rank == 0:
    dataset = [item for sublist in dataset for item in sublist]
    label = [item for sublist in label for item in sublist]

    # Encode labels into integer format
    label_encoder = LabelEncoder()
    label_encoder.fit(['Normal', 'Tumor', 'Cyst', 'Stone'])
    y_encoded = label_encoder.transform(label)

    # Split data and labels
    x_train, x_test, y_train_encoded, y_test_encoded = train_test_split(dataset, y_encoded, test_size=0.2, random_state=0)

    x_train = normalize(np.array(x_train), axis=1)
    x_test = normalize(np.array(x_test), axis=1)

    y_train_categorical = to_categorical(y_train_encoded, num_classes=4)
    y_test_categorical = to_categorical(y_test_encoded, num_classes=4)

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
    model.add(Dense(4))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model using OpenMP parallelism
    history = model.fit(x_train, y_train_categorical, batch_size=16, epochs=5, verbose=1, validation_data=(x_test, y_test_categorical), shuffle=False, use_multiprocessing=True)
    
    loss, accuracy = model.evaluate(x_test, y_test_categorical)
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
