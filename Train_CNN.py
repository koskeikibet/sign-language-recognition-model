import matplotlib.pyplot as plt  # Import Matplotlib for plotting
import tensorflow as tf  # Import TensorFlow for deep learning operations
from tensorflow import keras  # Import Keras for building models
from keras.models import Sequential  # Import Sequential model type
from keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout  # Import various layers
from keras.optimizers import Adam, SGD  # Import optimizers for compiling the model
from keras.metrics import categorical_crossentropy  # Import categorical cross-entropy loss
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Import ImageDataGenerator for data augmentation
import itertools  # Import itertools for efficient looping
import random  # Import random for random operations
import warnings  # Import warnings to ignore any unwanted warnings
import numpy as np  # Import NumPy for array manipulation
import cv2  # Import OpenCV for image processing
from keras.callbacks import ReduceLROnPlateau  # Import callback to reduce learning rate on plateau
from keras.callbacks import ModelCheckpoint, EarlyStopping  # Import callbacks for model checkpoint and early stopping
warnings.simplefilter(action='ignore', category=FutureWarning)  # Ignore future warnings for cleaner output

# Define paths for training and testing data
train_path = r'I:/PROJECT/amrican_indian_german sign language/asl_dataset'
test_path = r'I:/PROJECT/amrican_indian_german sign language/ISL_Dataset'

# Load training data with VGG16 preprocessing and data augmentation
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    directory=train_path, target_size=(64,64), class_mode='categorical', batch_size=10, shuffle=True)

# Load testing data with VGG16 preprocessing and data augmentation
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    directory=test_path, target_size=(64,64), class_mode='categorical', batch_size=10, shuffle=True)

imgs, labels = next(train_batches)  # Get a batch of training images and labels

# Function to plot images
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(30,20))  # Create subplot with 1 row, 10 columns
    axes = axes.flatten()  # Flatten axes array for easy looping
    for img, ax in zip(images_arr, axes):  # Iterate over images and axes
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR image to RGB
        ax.imshow(img)  # Display the image
        ax.axis('off')  # Hide axes
    plt.tight_layout()  # Adjust layout
    plt.show()  # Show plot

plotImages(imgs)  # Plot images from training batch
print(imgs.shape)  # Print shape of image batch
print(labels)  # Print labels of image batch

model = Sequential()  # Initialize sequential model

# First convolutional layer with 32 filters, 3x3 kernel, ReLU activation, and input shape 64x64x3
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(64,64,3)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))  # Max pooling layer with 2x2 pool size

# Second convolutional layer with 64 filters, 3x3 kernel, ReLU activation, and padding 'same'
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))  # Max pooling layer with 2x2 pool size

# Third convolutional layer with 128 filters, 3x3 kernel, ReLU activation, and padding 'valid'
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))  # Max pooling layer with 2x2 pool size

model.add(Flatten())  # Flatten the output for fully connected layers

model.add(Dense(64, activation="relu"))  # Dense layer with 64 units and ReLU activation
model.add(Dense(128, activation="relu"))  # Dense layer with 128 units and ReLU activation
#model.add(Dropout(0.2))  # Optional dropout layer for regularization
model.add(Dense(128, activation="relu"))  # Another Dense layer with 128 units
#model.add(Dropout(0.3))  # Optional dropout layer for regularization
model.add(Dense(10, activation="softmax"))  # Output layer with 10 units and softmax activation for classification

# Compile model with Adam optimizer
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)  # Reduce LR on plateau
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')  # Early stopping callback

# Compile model again with SGD optimizer for comparison
model.compile(optimizer=SGD(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0005)  # Reduce LR on plateau
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')  # Early stopping callback

# Train the model for 10 epochs
history2 = model.fit(train_batches, epochs=10, callbacks=[reduce_lr, early_stop], validation_data=test_batches)

imgs, labels = next(train_batches)  # Get another batch of images for evaluation

imgs, labels = next(test_batches)  # Get a batch of test images
scores = model.evaluate(imgs, labels, verbose=0)  # Evaluate model on test batch
print(f'{model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')  # Print loss and accuracy

model.save('I:/PROJECT/model_save')  # Save the trained model

print(history2.history)  # Print training history for insights

imgs, labels = next(test_batches)  # Get another test batch

model = keras.models.load_model(r"best_model_dataflair3.h5")  # Load the saved model

scores = model.evaluate(imgs, labels, verbose=0)  # Evaluate loaded model on test batch
print(f'{model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')  # Print loss and accuracy

model.summary()  # Print summary of model architecture

scores  # Display evaluation scores on test data
model.metrics_names  # Display metric names

# Dictionary for class labels to human-readable labels
word_dict = {0: 'One', 1: 'Ten', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine'}

predictions = model.predict(imgs, verbose=0)  # Predict on test images
print("predictions on a small set of test data--")  # Print predictions
print("")
for ind, i in enumerate(predictions):  # Iterate over predictions
    print(word_dict[np.argmax(i)], end='   ')  # Print predicted label

plotImages(imgs)  # Plot test images

print('Actual labels')  # Print actual labels
for i in labels:  # Iterate over true labels
    print(word_dict[np.argmax(i)], end='   ')  # Print actual label

print(imgs.shape)  # Print shape of test images
