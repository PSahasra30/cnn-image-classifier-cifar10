# 🧠 Image Classification with CNN on CIFAR-10

This project implements a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify images from the CIFAR-10 dataset into 10 categories. The model is also integrated with OpenCV for real-time image classification using the laptop's webcam.

## 📂 Dataset
- **CIFAR-10**: Contains 60,000 32x32 color images in 10 classes, with 6,000 images per class.
- Automatically loaded using `tf.keras.datasets.cifar10`.

## 🏗️ Model Architecture
- Conv2D → MaxPooling
- Conv2D → MaxPooling
- Flatten → Dense(120) → Dense(84) → Dense(10)

Achieved **~72% test accuracy** on the CIFAR-10 dataset.

## 🧪 Features
- Model training and evaluation on CIFAR-10
- Real-time prediction using webcam
- Output prediction probabilities displayed

## 📸 Sample Output
![Sample](samples/result.png)

## 🛠️ Requirements

Install dependencies:
