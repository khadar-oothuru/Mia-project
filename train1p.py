# import cProfile
# import cv2
# import os
# from PIL import Image
# import numpy as np
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import normalize, to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
# from mpi4py import MPI
# from sklearn.preprocessing import LabelEncoder

# def load_images(image_directory, images, label_value, input_size, rank, size):
#     images_list = []
#     for i, image_name in enumerate(images):
#         if i % size == rank:
#             if image_name.split('.')[1] == 'jpg':
#                 image = cv2.imread(os.path.join(image_directory, label_value, image_name))
#                 image = Image.fromarray(image, 'RGB')
#                 image = image.resize((input_size, input_size))
#                 images_list.append(np.array(image))
#     return images_list, [label_value] * len(images_list)

# def main():
#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     size = comm.Get_size()

#     image_directory = 'datasets/'
#     INPUT_SIZE = 64

#     normal_images = os.listdir(os.path.join(image_directory, 'Normal'))
#     tumor_images = os.listdir(os.path.join(image_directory, 'Tumor'))
#     cyst_images = os.listdir(os.path.join(image_directory, 'Cyst'))
#     stone_images = os.listdir(os.path.join(image_directory, 'Stone'))

#     dataset = []
#     label = []

#     normal_data, normal_labels = load_images(image_directory, normal_images, 'Normal', INPUT_SIZE, rank, size)
#     tumor_data, tumor_labels = load_images(image_directory, tumor_images, 'Tumor', INPUT_SIZE, rank, size)
#     cyst_data, cyst_labels = load_images(image_directory, cyst_images, 'Cyst', INPUT_SIZE, rank, size)
#     stone_data, stone_labels = load_images(image_directory, stone_images, 'Stone', INPUT_SIZE, rank, size)

#     local_dataset = []
#     local_dataset.extend(normal_data)
#     local_dataset.extend(tumor_data)
#     local_dataset.extend(cyst_data)
#     local_dataset.extend(stone_data)

#     local_label = []
#     local_label.extend(normal_labels)
#     local_label.extend(tumor_labels)
#     local_label.extend(cyst_labels)
#     local_label.extend(stone_labels)

#     dataset = comm.gather(local_dataset, root=0)
#     label = comm.gather(local_label, root=0)

#     if rank == 0:
#         dataset = [item for sublist in dataset for item in sublist]
#         label = [item for sublist in label for item in sublist]

#         label_encoder = LabelEncoder()
#         label_encoder.fit(['Normal', 'Tumor', 'Cyst', 'Stone'])
#         y_encoded = label_encoder.transform(label)

#         x_train, x_test, y_train_encoded, y_test_encoded = train_test_split(dataset, y_encoded, test_size=0.2, random_state=0)

#         x_train = normalize(np.array(x_train), axis=1)
#         x_test = normalize(np.array(x_test), axis=1)

#         y_train_categorical = to_categorical(y_train_encoded, num_classes=4)
#         y_test_categorical = to_categorical(y_test_encoded, num_classes=4)

#         model = Sequential()
#         model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
#         model.add(Activation('relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2)))

#         model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
#         model.add(Activation('relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2)))

#         model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
#         model.add(Activation('relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2)))

#         model.add(Flatten())
#         model.add(Dense(64))
#         model.add(Activation('relu'))
#         model.add(Dense(4))
#         model.add(Activation('softmax'))

#         model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#         history = model.fit(x_train, y_train_categorical, batch_size=16, epochs=5, verbose=1, validation_data=(x_test, y_test_categorical), shuffle=False, use_multiprocessing=True)
        
#         loss, accuracy = model.evaluate(x_test, y_test_categorical)
#         print(f'Test Accuracy: {accuracy}')

# if __name__ == "__main__":
#     main()
#     cProfile.run('main()', filename='profiling_results.prof')
import cProfile
import cv2
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import normalize, to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
from mpi4py import MPI
from sklearn.preprocessing import LabelEncoder
import threading

def load_images(image_directory, images, label_value, input_size, rank, size):
    images_list = []
    for i, image_name in enumerate(images):
        if i % size == rank:
            if image_name.split('.')[1] == 'jpg':
                image = cv2.imread(os.path.join(image_directory, label_value, image_name))
                image = Image.fromarray(image, 'RGB')
                image = image.resize((input_size, input_size))
                images_list.append(np.array(image))
    return images_list, [label_value] * len(images_list)

def load_images_parallel(image_directory, images, label_value, input_size, rank, size):
    local_data = []
    local_labels = []
    for i, image_name in enumerate(images):
        if i % size == rank:
            if image_name.split('.')[1] == 'jpg':
                image = cv2.imread(os.path.join(image_directory, label_value, image_name))
                image = Image.fromarray(image, 'RGB')
                image = image.resize((input_size, input_size))
                local_data.append(np.array(image))
                local_labels.append(label_value)
    return local_data, local_labels

def load_images_parallel_thread(image_directory, images, label_value, input_size, rank, size, thread_id, num_threads, local_datasets, local_labels):
    local_data = []
    local_labels = []
    for i, image_name in enumerate(images):
        if i % size == rank and i % num_threads == thread_id:
            if image_name.split('.')[1] == 'jpg':
                image = cv2.imread(os.path.join(image_directory, label_value, image_name))
                image = Image.fromarray(image, 'RGB')
                image = image.resize((input_size, input_size))
                local_data.append(np.array(image))
                local_labels.append(label_value)
    local_datasets[thread_id] = local_data
    local_labels[thread_id] = local_labels

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    image_directory = 'datasets/'
    INPUT_SIZE = 64

    normal_images = os.listdir(os.path.join(image_directory, 'Normal'))
    tumor_images = os.listdir(os.path.join(image_directory, 'Tumor'))
    cyst_images = os.listdir(os.path.join(image_directory, 'Cyst'))
    stone_images = os.listdir(os.path.join(image_directory, 'Stone'))

    dataset = []
    label = []

    normal_data, normal_labels = load_images(image_directory, normal_images, 'Normal', INPUT_SIZE, rank, size)
    tumor_data, tumor_labels = load_images(image_directory, tumor_images, 'Tumor', INPUT_SIZE, rank, size)
    cyst_data, cyst_labels = load_images(image_directory, cyst_images, 'Cyst', INPUT_SIZE, rank, size)
    stone_data, stone_labels = load_images(image_directory, stone_images, 'Stone', INPUT_SIZE, rank, size)

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

        label_encoder = LabelEncoder()
        label_encoder.fit(['Normal', 'Tumor', 'Cyst', 'Stone'])
        y_encoded = label_encoder.transform(label)

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

        history = model.fit(x_train, y_train_categorical, batch_size=16, epochs=5, verbose=1, validation_data=(x_test, y_test_categorical), shuffle=False, use_multiprocessing=True)
        
        loss, accuracy = model.evaluate(x_test, y_test_categorical)
        print(f'Test Accuracy: {accuracy}')

if __name__ == "__main__":
    main()
    cProfile.run('main()', filename='profiling_results.prof')
