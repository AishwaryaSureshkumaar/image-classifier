# Movie Genre Classifier

## Overview

The Movie Genre Classifier is a machine learning project that classifies images of movies into three genres: **Action, Horror, and Anime**. 
The model used in this project was trained using **Teachable Machine** and is used to predict the genre of a given movie image. 
The classifier is implemented using Python and popular machine learning libraries like **TensorFlow** and **OpenCV**.

## Project Structure

Movie-Genre-Classifier/


-> **model**              # Folder containing the trained model file from Teachable Machine
-> **images**              # Folder with images to test and classify
-> **script**              # Python script for loading the model and making predictions
-> **requirements.txt**     # List of dependencies
-> **README.md**            # Project documentation
-> **imagepycls.py**        # Python script for image classification


## Labels
Action Movie--
Horror Movie--
Anime Movie

## Requirements
Before running the project, ensure that you have the necessary Python packages installed. You can install all dependencies by running:

    ```
    pip install -r requirements.txt
     
    ```

### How to Use the Classifier

1. Clone the repository to your local machine
2. Navigate to the project directory
3. Install dependencies
4. Prepare your images in the images/ folder
5. Run the classifier using the imagepycls.py script
6. View the output, which will display the predicted genre for the input image, such as:


    ```
    Predicted Genre: Action Movie

    ```

