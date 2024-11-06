import matplotlib.pyplot as plt  # Import Matplotlib for plotting
import tensorflow as tf  # Import TensorFlow for deep learning operations
from tensorflow import keras  # Import Keras for building models
from keras.models import Sequential  # Import Sequential model type
from keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout  # Import various layers
from keras.optimizers import Adam, SGD  # Import optimizers for compiling the model
from keras.metrics import categorical_crossentropy  # Import categorical cross-entropy loss
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Import ImageDataGenerator for data augmentation
import numpy as np  # Import NumPy for array manipulation
import cv2  # Import OpenCV for image processing
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping  # Import callbacks

# Define paths for training and testing data
train_path = r'I:/PROJECT/amrican_indian_german sign language/asl_dataset'
test_path = r'I:/PROJECT/amrican_indian_german sign language/ISL_Dataset'

# Load training data with VGG16 preprocessing and data augmentation
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    directory=train_path, target_size=(64, 64), class_mode='categorical', batch_size=10, shuffle=True)

# Load testing data with VGG16 preprocessing
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    directory=test_path, target_size=(64, 64), class_mode='categorical', batch_size=10, shuffle=False)

# Function to plot images
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(30, 20))  # Create subplot with 1 row, 10 columns
    axes = axes.flatten()  # Flatten axes array for easy looping
    for img, ax in zip(images_arr, axes):  # Iterate over images and axes
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR image to RGB
        ax.imshow(img)  # Display the image
        ax.axis('off')  # Hide axes
    plt.tight_layout()  # Adjust layout
    plt.show()  # Show plot

# Visualize some training data
imgs, labels = next(train_batches)  # Get a batch of training images and labels
plotImages(imgs)  # Plot images from training batch

# Model Creation
model = Sequential()  # Initialize sequential model

# First convolutional layer with 32 filters, 3x3 kernel, ReLU activation
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPool2D(pool_size=(2, 2)))  # Max pooling layer

# Second convolutional layer with 64 filters
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2)))  # Max pooling layer

# Third convolutional layer with 128 filters
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid'))
model.add(MaxPool2D(pool_size=(2, 2)))  # Max pooling layer

model.add(Flatten())  # Flatten the output for fully connected layers
model.add(Dense(128, activation='relu'))  # Dense layer
model.add(Dense(10, activation='softmax'))  # Output layer with 10 classes (change if more classes are added)

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)  # Reduce LR on plateau
early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=0)  # Early stopping callback

# Train the model for 10 epochs
history = model.fit(train_batches, epochs=10, callbacks=[reduce_lr, early_stop], validation_data=test_batches)

# Evaluate the model
imgs, labels = next(test_batches)  # Get a batch of test images
scores = model.evaluate(imgs, labels, verbose=0)  # Evaluate model on test batch
print(f'Loss: {scores[0]}; Accuracy: {scores[1]*100}%')  # Print loss and accuracy

# Save the trained model
model.save('I:/PROJECT/amrican_indian_german sign language/save_train_model')  # Save the trained model

# Print training history
print(history.history)

# Display model summary
model.summary()  # Print summary of model architecture
