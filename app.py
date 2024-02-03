from flask import Flask, render_template, request
from tensorflow import keras
from keras.models import load_model
from keras.layers import BatchNormalization
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
import cv2
import numpy as np
import pickle
from werkzeug.utils import secure_filename
import imghdr
import logging

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

custom_objects = {'BatchNormalization': BatchNormalization}
# Load the pre-trained machine learning model
model = load_model('model_cnn.h5', custom_objects=custom_objects)

# Load label binarizer
label_binarizer = pickle.load(open('label_transform.pkl', 'rb'))

# Image processing function
def process_uploaded_image(file):
    try:
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
        filename = secure_filename(file.filename)

        # Validate file extension
        if '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions:
            # Read and resize the image
            image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                image = cv2.resize(image, (224, 224))
                return img_to_array(image)
        else:
            app.logger.info("Invalid file format")
            return None
    except Exception as e:
        app.logger.error(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the POST request has the file part
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    # If the user submits an empty form
    if file.filename == '':
        return render_template('index.html', error='No selected file')

    # Process the uploaded image
    processed_image = process_uploaded_image(file)

    if processed_image is not None:
        # Reshape the processed image for prediction
        input_features = np.expand_dims(processed_image, axis=0)

        # Make a prediction using the pre-trained model
        prediction_probs = model.predict(input_features)
        prediction_class = np.argmax(prediction_probs)

        # Transform the prediction class back to the original label
        predicted_label = label_binarizer.classes_[prediction_class]

        return render_template('result.html', result=predicted_label)

    return render_template('index.html', error='Error processing image')

if __name__ == '_main_':
    app.run(debug=True)
