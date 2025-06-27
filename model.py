# model.py

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def main():
    # Load CIFAR-10 dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    # Normalize to [-1, 1]
    train_images = train_images / 127.5 - 1
    test_images = test_images / 127.5 - 1

    # Class names
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Build the CNN model
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(120, activation='relu'),
        layers.Dense(84, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    print(f'Accuracy on test images: {test_accuracy * 100:.2f}%')

    # Save the trained model
    model.save('cifar10_model.h5')

if __name__ == "__main__":
    main()
