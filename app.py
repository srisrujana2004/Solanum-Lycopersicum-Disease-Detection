# app.py
import io
import base64
from flask import Flask, render_template, request, jsonify
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the pretrained Keras model
model = keras.models.load_model('AlexNet.h5')  # Replace 'AlexNet.h5' with the path to your model file

# Define the image size expected by the model
img_size = (224, 224)

# Class name dictionary
class_name_dict = {
    0: 'Bacterial spot',
    1: 'Early blight',
    2: 'Fusarium wilt',
    3: 'Late blight',
    4: 'Leaf mold',
    5: 'Septoria leaf spot',
    6: 'Spider mites',
    7: 'Target spot',
    8: 'Yellow leaf curl virus',
    9: 'Mosaic virus',
    10: 'Healthy'
}

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for image detection
@app.route('/detect', methods=['POST'])
# ...

# Route for image detection
# ...

# Route for image detection
@app.route('/detect', methods=['POST'])
def detect():
    # Get the uploaded image file
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Check if the file is an allowed extension
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({'error': 'Invalid file extension'})

    # Convert the SpooledTemporaryFile to BytesIO
    img_bytes_io = io.BytesIO(file.read())

    # Load and preprocess the image
    img = image.load_img(img_bytes_io, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values

    # Make a prediction
    predictions = model.predict(img_array)

    # Get the predicted class index
    predicted_class = np.argmax(predictions)

    # Get the corresponding class label
    predicted_label = class_name_dict.get(predicted_class, 'Unknown')

    result = {
        'image': f'data:image/png;base64,{base64.b64encode(img_bytes_io.getvalue()).decode()}',
        'class': predicted_label
    }

    return render_template('index.html', result=result)

# ...



if __name__ == '__main__':
    app.run(debug=True)
