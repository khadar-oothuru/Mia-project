import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt



model = load_model('trainedModel.h5')


EXPECTED_INPUT_SIZE = (64, 64)


test_image_path = "./datasets/Tumor/Tumor- (1047).jpg"

test_image = cv2.imread(test_image_path)
test_image = cv2.resize(test_image, EXPECTED_INPUT_SIZE)
test_image = np.reshape(test_image, [1, *EXPECTED_INPUT_SIZE, 3]) 
test_image = test_image / 255.0 

predictions = model.predict(test_image)

predicted_class = np.argmax(predictions)

class_labels = {0: 'Normal', 1: 'Tumor', 2: 'Cyst', 3: 'Stone'}
predicted_label = class_labels[predicted_class]

plt.imshow(cv2.cvtColor(cv2.imread(test_image_path), cv2.COLOR_BGR2RGB))
plt.title(f'Predicted Class: {predicted_label}')
plt.axis('off')  
plt.show()


print(f'The model predicts that the image belongs to class: {predicted_label}')

