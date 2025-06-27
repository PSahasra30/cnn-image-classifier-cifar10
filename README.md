CIFAR-10 Image Classifier using CNN

This project involves building a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset contains 60,000 32x32 color images across 10 categories: plane, car, bird, cat, deer, dog, frog, horse, ship, and truck. The model is trained to achieve around 72% accuracy on the test dataset. After training, the model can also be used to classify real-time images using your laptop webcam.

The project is implemented in Python and uses the following technologies: TensorFlow, NumPy, Matplotlib, and OpenCV. The CNN architecture includes two convolutional layers with ReLU activation, followed by max-pooling layers and fully connected dense layers leading to a softmax output layer.

You can train the model by running model.py, which loads the CIFAR-10 dataset, preprocesses it, builds the CNN model, trains it for 10 epochs, evaluates its performance, and saves the trained model as cifar10_model.h5. After that, you can use webcam_predict.py to capture images from your laptop camera and predict their class in real time using the saved model.

To run the project:
1. Clone the repository.
2. Install the required packages using pip install -r requirements.txt.
3. Run model.py to train and save the model.
4. Run webcam_predict.py to perform live classification using your webcam.

Project files:
- model.py: Code for building, training, evaluating, and saving the CNN model.
- webcam_predict.py: Code for capturing webcam frames and predicting CIFAR-10 classes in real time.
- requirements.txt: List of required Python packages.
- README.md: Project overview and instructions.
- samples/result.png: Optional image showing the webcam prediction result.

Test Accuracy: ~72%
Model File: cifar10_model.h5

You can improve this project further by adding GUI support with Streamlit, deploying it as a web app, applying data augmentation, or fine-tuning the model for better accuracy.

Author: Papisetty Sahasra
GitHub: https://github.com/PSahasra30
