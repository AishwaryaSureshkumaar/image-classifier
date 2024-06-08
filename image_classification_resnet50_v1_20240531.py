import tensorflow as tf
import numpy as np
import cv2
from keras.layers import TFSMLayer

# Load the SavedModel
model_path = r'C:\Users\krish\OneDrive\Desktop\converted_savedmodel\model.savedmodel'
model_layer = TFSMLayer(model_path, call_endpoint='serving_default')

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))  # Resize the image to match the model input
    image = image.astype('float32') / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to get prediction
def get_prediction(image):
    preprocessed_image = preprocess_image(image)
    prediction = model_layer(preprocessed_image)
    return prediction


# Capture video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get prediction for the current frame
    prediction = get_prediction(frame)

    # Display the prediction
    cv2.putText(frame, f'Prediction: {prediction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
