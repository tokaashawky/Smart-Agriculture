from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image

app = Flask(__name__)

# Load your single model here
model = tf.keras.models.load_model('model.h5', compile=False)

classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
           'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
           'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
           'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
           'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
           'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
           'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
           'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
           'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
           'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
           'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
           'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']


def prepare_image(image, target):
    # Convert image to RGB if it's not
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize and preprocess the image
    image = image.resize(target)
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)

    return image



@app.route('/predict_image', methods=['POST'])
def predict():
    data = {"success": False}

    if request.method == "POST":
        if "image" in request.files:
            try:
                image_file = request.files["image"]
                image = Image.open(image_file)

                if image:
                    # Prepare the image for prediction
                    processed_image = prepare_image(image, target=(256, 256))

                    # Predict
                    preds = model.predict(processed_image)
                    predicted_class_index = np.argmax(preds)
                    predicted_class = classes[predicted_class_index]
                    confidence = preds[0][predicted_class_index]

                    data = {"label": predicted_class, "probability": round(float(confidence), 2)}


            except Exception as e:
                data["error"] = str(e)

    return jsonify(data)


if __name__ == "__main__":
    app.run()
