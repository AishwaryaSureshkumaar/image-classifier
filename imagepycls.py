import tensorflow as tf
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import cv2

# Load the SavedModel
model_path = r'C:\Users\krish\OneDrive\Desktop\converted_savedmodel - Copy\model.savedmodel'
try:
    model_layer = tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Labels for classification
labels = ['Action Movie', 'Horror Movie', 'Anime Movie']

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))  # Resize the image to match the model input
    image = image.astype('float32') / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to get prediction
def get_prediction(image):
    preprocessed_image = preprocess_image(image)
    prediction_dict = model_layer(preprocessed_image)
    
    # Print prediction dictionary keys for debugging
    print(f"Prediction dictionary keys: {prediction_dict.keys()}")

    # Extract the prediction from the dictionary using the appropriate key
    try:
        prediction = prediction_dict['sequential_3']
    except KeyError:
        print("Error: Key 'sequential_3' not found in prediction dictionary.")
        return None, None

    # Convert the prediction tensor to a numpy array
    if tf.is_tensor(prediction):
        prediction = prediction.numpy()

    # Ensure prediction is a 2D array
    if len(prediction.shape) == 1:
        prediction = np.expand_dims(prediction, axis=0)

    predicted_class = np.argmax(prediction, axis=1)[0]  # Get the index of the highest score
    confidence = np.max(prediction)  # Get the highest score
    
    # If the predicted_class is out of bounds or confidence is None, return None
    if predicted_class >= len(labels) or confidence is None:
        return None, None

    return labels[predicted_class], confidence

# Function to load image from URL
def load_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        image = Image.open(BytesIO(response.content))
        image = image.convert('RGB')  # Ensure image is in RGB format
        image = np.array(image)
        return image
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the image: {e}")
        return None
    except Image.UnidentifiedImageError:
        print("Error: The URL does not contain a valid image.")
        return None

# Main loop
while True:
    # Get image URL input from the user
    image_url = input("Please enter the URL of the image (or type 'exit' to quit): ")

    if image_url.lower() == 'exit':
        break

    # Load the image from the URL
    image = load_image_from_url(image_url)

    if image is not None:
        # Get prediction for the image
        predicted_label, confidence = get_prediction(image)

        if predicted_label is not None and confidence is not None:
            # Display the result
            print(f'Predicted Label: {predicted_label} with confidence {confidence:.2f}')

            # Display the image using OpenCV
            cv2.putText(image, f'{predicted_label} ({confidence:.2f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Image', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Failed to get a valid prediction.")
    else:
        print("Failed to load the image.")
