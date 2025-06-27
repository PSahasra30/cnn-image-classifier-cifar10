import cv2
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model('cifar10_model.h5')

# Class labels
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

# Open laptop camera (0 is usually the default)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    # Resize to 32x32 as required by the model
    resized = cv2.resize(frame, (32, 32))
    normalized = resized / 127.5 - 1  # normalize to [-1, 1]
    input_image = np.expand_dims(normalized, axis=0)  # shape: (1, 32, 32, 3)

    # Predict
    predictions = model.predict(input_image)
    class_index = np.argmax(predictions)
    confidence = predictions[0][class_index]
    label = f"{classes[class_index]} ({confidence*100:.2f}%)"

    # Show label on the original frame
    cv2.putText(frame, label, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Webcam Classification", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
