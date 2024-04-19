import cProfile
import cv2
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import normalize, to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation

# Function to load images and labels
def load_images(image_directory, images, label_value, input_size):
    images_list = []
    for i, image_name in enumerate(images):
        if image_name.split('.')[1] == 'jpg':
            image = cv2.imread(os.path.join(image_directory, label_value, image_name))
            image = Image.fromarray(image, 'RGB')
            image = image.resize((input_size, input_size))
            images_list.append(np.array(image))
    return images_list, [label_value] * len(images_list)

def main():
    image_directory = 'datasets/'
    INPUT_SIZE = 64

    normal_images = os.listdir(os.path.join(image_directory, 'Normal'))
    tumor_images = os.listdir(os.path.join(image_directory, 'Tumor'))
    cyst_images = os.listdir(os.path.join(image_directory, 'Cyst'))
    stone_images = os.listdir(os.path.join(image_directory, 'Stone'))

    dataset = []
    label = []

    # Load normal images
    normal_data, normal_labels = load_images(image_directory, normal_images, 'Normal', INPUT_SIZE)
    dataset.extend(normal_data)
    label.extend(normal_labels)

    # Load tumor images
    tumor_data, tumor_labels = load_images(image_directory, tumor_images, 'Tumor', INPUT_SIZE)
    dataset.extend(tumor_data)
    label.extend(tumor_labels)

    # Load cyst images
    cyst_data, cyst_labels = load_images(image_directory, cyst_images, 'Cyst', INPUT_SIZE)
    dataset.extend(cyst_data)
    label.extend(cyst_labels)

    # Load stone images
    stone_data, stone_labels = load_images(image_directory, stone_images, 'Stone', INPUT_SIZE)
    dataset.extend(stone_data)
    label.extend(stone_labels)

    dataset = np.array(dataset)
    label = np.array(label)

    # Split data and labels
    x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

    # Normalize pixel values
    x_train = normalize(x_train, axis=1)
    x_test = normalize(x_test, axis=1)

    # Encode labels into integer format
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Convert integer labels to categorical format
    y_train_categorical = to_categorical(y_train_encoded, num_classes=4)
    y_test_categorical = to_categorical(y_test_encoded, num_classes=4)

    # Define the model architecture
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

    # Train the model
    history = model.fit(x_train, y_train_categorical, batch_size=16, epochs=5, verbose=1, validation_data=(x_test, y_test_categorical), shuffle=False)

    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test_categorical)
    print(f'Test Accuracy: {accuracy}')

    # Plot training history
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

    # Save the trained model
    model.save('trainedModel1.h5')

if __name__ == "__main__":
    cProfile.run('main()', filename='profilingp_results.prof')
