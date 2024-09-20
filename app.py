from flask import Flask, render_template, request, redirect, url_for
import os
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load the pre-trained MobileNet model
model = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet')

def preprocess_image(image_path):
    img = Image.open(image_path)
    
    # Convert the image to RGB if it's not already in that mode (i.e., if it's RGBA or L mode)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img = img.resize((224, 224))  # Resize to the expected size
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array
# Load the custom trained model
model = tf.keras.models.load_model('vehicle_classifier_model.h5')


def classify_image(image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    labels = ['bus',  'car', 'truck', 'motorcycle']
    predicted_label = labels[np.argmax(predictions)]
    confidence = np.max(predictions)
    return predicted_label, confidence


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if an image was uploaded
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        # Save the file and classify it
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        prediction = classify_image(file_path)
        
        # Render results
        return render_template("result.html", filename=file.filename, label=prediction[0], confidence=round(prediction[1] * 100, 2))

    return render_template("index.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return redirect(url_for('static', filename=f'uploads/{filename}'))

if __name__ == "__main__":
    app.run(debug=True)
