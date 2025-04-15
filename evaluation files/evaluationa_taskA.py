#Author: Alap Dhruva (400490512)

import torch
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model, Model

from skimage.transform import resize

def plot_sample(X, y, index, class_names, model,data_num):
    plt.figure(figsize=(4, 4))
    plt.imshow(X[index])
    plt.xlabel(f"True: {class_names[y[index]]}")
    plt.axis('off')
    plt.show()

    # Resize to match model's expected input (e.g., 48x48x3 -> 4608)
    input_image = resize(X[index], (48, 48, 3))  # Resize to 48x48x3
    input_image_flattened = input_image.reshape(1, -1)  # Now shape (1, 6912)

    # Select first 4608 features (or use PCA/other methods to match)
    input_image_flattened = input_image_flattened[:, :4608]  # Truncate to 4608

    predictions = model.predict(input_image_flattened)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_label = class_names[predicted_class_index+data_num]

    print(f"Predicted class: {predicted_class_label}, index is {predicted_class_index}")

from google.colab import drive
drive.mount('/content/drive')

train_path_1 = "/content/drive/MyDrive/Task1_data/Model1/model1_test.pth"
test_path_1 = "/content/drive/MyDrive/Task1_data/Model1/model1_train.pth"

train_path_2 = "/content/drive/MyDrive/Task1_data/Model2/model2_test.pth"
test_path_2 = "/content/drive/MyDrive/Task1_data/Model2/model2_train.pth"

train_path_3 = "/content/drive/MyDrive/Task1_data/Model3/model3_test.pth"
test_path_3 = "/content/drive/MyDrive/Task1_data/Model3/model3_train.pth"

label_mapping = {
    0: 0,   # apple -> 0
    10: 1,  # bowl -> 1
    20: 2,  # chair -> 2
    30: 3,  # dolphin -> 3
    40: 4,   # lamp -> 4

    1: 5,   # aquarium_fish -> 0
    11: 6,  # boy -> 1
    21: 7,  # chimpanzee -> 2
    31: 8,  # elephant -> 3
    41: 9,   # lawn_mower -> 4

    2: 10,   # baby -> 0
    12: 11,  # bridge -> 1
    22: 12,  # clock -> 2
    32: 13,  # flatfish -> 3
    42: 14   # leopard -> 4
}
# Define class labels (edit as per your model)
class_labels = ['apple', 'bowl', 'chair', 'dolphin', 'lamp',
              'aquarium_fish', 'boy', 'chimpanzee', 'elephant', 'lawn_mower',
              'baby', 'bridge', 'clock', 'flatfish', 'leopard']

model = load_model('/content/drive/MyDrive/fusion_model.h5')

# Load all test datasets
test_data_1 = torch.load(test_path_1)
test_data_2 = torch.load(test_path_2)
test_data_3 = torch.load(test_path_3)

# Convert to numpy and transpose to (N, H, W, C)
X_test_1 = np.transpose(test_data_1['data'].numpy(), (0, 2, 3, 1))
X_test_2 = np.transpose(test_data_2['data'].numpy(), (0, 2, 3, 1))
X_test_3 = np.transpose(test_data_3['data'].numpy(), (0, 2, 3, 1))

# Map labels using label_mapping
y_test_1 = np.array([label_mapping[label] for label in test_data_1['labels']])
y_test_2 = np.array([label_mapping[label] for label in test_data_2['labels']])
y_test_3 = np.array([label_mapping[label] for label in test_data_3['labels']])

# Combine all datasets
X_test_all = np.concatenate([X_test_1, X_test_2, X_test_3])
y_test_all = np.concatenate([y_test_1, y_test_2, y_test_3])


from skimage.transform import resize
import tensorflow as tf # Make sure tensorflow is imported

# **Change 1: Resize to the correct input shape**
# Assuming the model expects input shape (48, 48, 3)
X_test_resized = np.array([
    resize(img, (48, 48, 3)) for img in X_test_all  # **Resize and keep the 3 channels**
])


# One-hot encode the labels
y_test_cat = tf.keras.utils.to_categorical(y_test_all, num_classes=15) # Assuming 15 classes

# **Change 2: Print the model's input shape for verification**
print(f"Model input shape: {model.input_shape}")

loss, accuracy = model.evaluate(X_test_resized, y_test_cat, verbose=1)
print(f"Fusion Model Test Accuracy: {accuracy * 100:.2f}%")

# Evaluate the model on your test data
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Model Test Accuracy: {accuracy * 100:.2f}%")

def plot_accuracy(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='x')
    plt.title('Model Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Call the function with your history object
# plot_accuracy(history)  # Uncomment this after training or loading the history

# Plot a sample image from Model 1
data_num = 0
test_data_1 = torch.load(test_path_1)
X_test_1 = np.transpose(test_data_1['data'].numpy(), (0, 2, 3, 1))
y_test_1 = np.array([label_mapping[label] for label in test_data_1['labels']])
plot_sample(X_test_1, y_test_1, 9, class_labels,model,data_num)

data_num = 5
test_data_2 = torch.load(test_path_2)
X_test_2 = np.transpose(test_data_2['data'].numpy(), (0, 2, 3, 1))
y_test_2 = np.array([label_mapping[label] for label in test_data_2['labels']])
plot_sample(X_test_2, y_test_2, 11, class_labels,model,data_num)

data_num = 4
test_data_2 = torch.load(test_path_2)
X_test_1 = np.transpose(test_data_2['data'].numpy(), (0, 2, 3, 1))
y_test_1 = np.array([label_mapping[label] for label in test_data_1['labels']])
plot_sample(X_test_1, y_test_1, 9, class_labels, model, data_num)

data_num = 6
test_data_3 = torch.load(test_path_3)
X_test_1 = np.transpose(test_data_3['data'].numpy(), (0, 2, 3, 1))
y_test_1 = np.array([label_mapping[label] for label in test_data_1['labels']])
plot_sample(X_test_1, y_test_1, 5, class_labels, model, data_num)

data_num = 3
test_data_3 = torch.load(test_path_3)
X_test_2 = np.transpose(test_data_3['data'].numpy(), (0, 2, 3, 1))
y_test_2 = np.array([label_mapping[label] for label in test_data_3['labels']])
plot_sample(X_test_1, y_test_1, 9, class_labels, model, data_num)

data_num = 9
test_data_2 = torch.load(test_path_3)
X_test_2 = np.transpose(test_data_2['data'].numpy(), (0, 2, 3, 1))
y_test_2 = np.array([label_mapping[label] for label in test_data_3['labels']])
plot_sample(X_test_1, y_test_1, 8, class_labels, model, data_num)

data_num = 6
test_data_3 = torch.load(test_path_3)
X_test_2 = np.transpose(test_data_3['data'].numpy(), (0, 2, 3, 1))
y_test_2 = np.array([label_mapping[label] for label in test_data_3['labels']])
plot_sample(X_test_1, y_test_1, 3, class_labels, model, data_num)

data_num = 8
test_data_3 = torch.load(test_path_3)
X_test_2 = np.transpose(test_data_3['data'].numpy(), (0, 2, 3, 1))
y_test_2 = np.array([label_mapping[label] for label in test_data_3['labels']])
plot_sample(X_test_1, y_test_1, 1, class_labels, model, data_num)

data_num = 10
test_data_3 = torch.load(test_path_3)
X_test_2 = np.transpose(test_data_3['data'].numpy(), (0, 2, 3, 1))
y_test_2 = np.array([label_mapping[label] for label in test_data_3['labels']])
plot_sample(X_test_1, y_test_1, 10, class_labels, model, data_num)

data_num = 7
test_data_3 = torch.load(test_path_2)
X_test_2 = np.transpose(test_data_3['data'].numpy(), (0, 2, 3, 1))
y_test_2 = np.array([label_mapping[label] for label in test_data_3['labels']])
plot_sample(X_test_1, y_test_1, 2, class_labels, model, data_num)