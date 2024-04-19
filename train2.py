import cv2
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
from mpi4py import MPI
import multiprocessing

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Initialize OpenMP
import os
os.environ["OMP_NUM_THREADS"] = "4"  # Set the number of threads according to your system

# Define function to load and preprocess images
def load_images(image_directory, image_list):
    images = []
    labels = []
    for image_name in image_list:
        if image_name.split('.')[1] == 'jpg':
            image = cv2.imread(os.path.join(image_directory, image_name))
            image = Image.fromarray(image, 'RGB')
            image = image.resize((INPUT_SIZE, INPUT_SIZE))
            images.append(np.array(image))
            labels.append(image_directory.split('/')[-2])  # Extract label from directory name
    return images, labels

# Define function to parallelize data loading
def parallel_load_data(image_directory):
    normal_images = os.listdir(os.path.join(image_directory, 'Normal/'))
    tumor_images = os.listdir(os.path.join(image_directory, 'Tumor/'))
    cyst_images = os.listdir(os.path.join(image_directory, 'Cyst/'))
    stone_images = os.listdir(os.path.join(image_directory, 'Stone/'))
    
    if rank == 0:
        normal_images_chunks = np.array_split(normal_images, size)
        tumor_images_chunks = np.array_split(tumor_images, size)
        cyst_images_chunks = np.array_split(cyst_images, size)
        stone_images_chunks = np.array_split(stone_images, size)
    else:
        normal_images_chunks = None
        tumor_images_chunks = None
        cyst_images_chunks = None
        stone_images_chunks = None
    
    # Distribute data across processes
    normal_images_chunk = comm.scatter(normal_images_chunks, root=0)
    tumor_images_chunk = comm.scatter(tumor_images_chunks, root=0)
    cyst_images_chunk = comm.scatter(cyst_images_chunks, root=0)
    stone_images_chunk = comm.scatter(stone_images_chunks, root=0)
    
    # Load and preprocess images
    normal_data, normal_labels = load_images(os.path.join(image_directory, 'Normal/'), normal_images_chunk)
    tumor_data, tumor_labels = load_images(os.path.join(image_directory, 'Tumor/'), tumor_images_chunk)
    cyst_data, cyst_labels = load_images(os.path.join(image_directory, 'Cyst/'), cyst_images_chunk)
    stone_data, stone_labels = load_images(os.path.join(image_directory, 'Stone/'), stone_images_chunk)
    
    return normal_data, normal_labels, tumor_data, tumor_labels, cyst_data, cyst_labels, stone_data, stone_labels

# Define function to flatten nested lists
def flatten(lst):
    return [item for sublist in lst for item in sublist]

# Load data in parallel
if rank == 0:
    normal_data, normal_labels, tumor_data, tumor_labels, cyst_data, cyst_labels, stone_data, stone_labels = parallel_load_data(image_directory)
else:
    normal_data, normal_labels, tumor_data, tumor_labels, cyst_data, cyst_labels, stone_data, stone_labels = None, None, None, None, None, None, None, None

# Gather data from all processes
normal_data_all = comm.gather(normal_data, root=0)
normal_labels_all = comm.gather(normal_labels, root=0)
tumor_data_all = comm.gather(tumor_data, root=0)
tumor_labels_all = comm.gather(tumor_labels, root=0)
cyst_data_all = comm.gather(cyst_data, root=0)
cyst_labels_all = comm.gather(cyst_labels, root=0)
stone_data_all = comm.gather(stone_data, root=0)
stone_labels_all = comm.gather(stone_labels, root=0)

# Combine data from all processes
if rank == 0:
    normal_data_combined = flatten(normal_data_all)
    normal_labels_combined = flatten(normal_labels_all)
    tumor_data_combined = flatten(tumor_data_all)
    tumor_labels_combined = flatten(tumor_labels_all)
    cyst_data_combined = flatten(cyst_data_all)
    cyst_labels_combined = flatten(cyst_labels_all)
    stone_data_combined = flatten(stone_data_all)
    stone_labels_combined = flatten(stone_labels_all)

    # Combine all data and labels
    dataset = np.concatenate((normal_data_combined, tumor_data_combined, cyst_data_combined, stone_data_combined), axis=0)
    labels = np.concatenate((normal_labels_combined, tumor_labels_combined, cyst_labels_combined, stone_labels_combined), axis=0)

    # Shuffle dataset and labels
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    dataset = dataset[indices]
    labels = labels[indices]

# Split dataset into training and testing sets
if rank == 0:
    x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=0)

    # Preprocess data
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    y_train = to_categorical(y_train, num_classes=4)
    y_test = to_categorical(y_test, num_classes=4)

# Parallelize model training using OpenMP
if rank == 0:
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

    # Set number of epochs
    epochs = 5

    # Parallelize training loop with OpenMP
    with multiprocessing.Pool(processes=size) as pool:
        results = pool.starmap(model.fit, [(x_train, y_train)] * size)

    # Combine training results from all processes
    for result in results:
        model.history.history['loss'].extend(result.history['loss'])
        model.history.history['accuracy'].extend(result.history['accuracy'])
        model.history.history['val_loss'].extend(result.history['val_loss'])
        model.history.history['val_accuracy'].extend(result.history['val_accuracy'])

# Synchronize all processes
comm.Barrier()

# Get training history from rank 0 process
history = comm.bcast(model.history.history, root=0)

# Evaluate model
if rank == 0:
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Test Accuracy: {accuracy}')

    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Save model
    model.save('trainedModel1.h5')
