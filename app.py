# # app.py - Flask API for bird classification
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import tensorflow as tf
# import tensorflow_hub as hub
# import numpy as np
# from PIL import Image
# import io
# import base64

# app = Flask(__name__)
# CORS(app)  # Allow cross-origin requests

# # Load the bird classification model
# IMAGE_RES = 224  # input dimensions required by the model
# URL = 'https://tfhub.dev/google/aiy/vision/classifier/birds_V1/1'

# print("Loading bird classification model...")
# # Use a more compatible approach to load the model
# try:
#     # Try the direct tf.keras approach first
#     model = tf.keras.Sequential()
#     hub_layer = hub.KerasLayer(URL, input_shape=(IMAGE_RES, IMAGE_RES, 3), trainable=False)
#     model.add(hub_layer)
#     model.build([None, IMAGE_RES, IMAGE_RES, 3])
#     print("Model loaded successfully with tf.keras approach")
# except ValueError:
#     # Fallback to using the module directly if Sequential fails
#     print("Falling back to direct module loading...")
#     hub_module = hub.load(URL)
#     def model_predict(image):
#         return hub_module.signatures['default'](tf.convert_to_tensor(image))
#     # Create a wrapper function to match the expected API
#     class ModelWrapper:
#         def predict(self, image):
#             result = model_predict(image)
#             # Convert the result to the expected format
#             return result['default'].numpy()
    
#     model = ModelWrapper()
#     print("Model loaded successfully with module wrapper approach")

# # Load bird labels directly from the official source
# import requests

# labels = []
# try:
#     response = requests.get('https://www.gstatic.com/aihub/tfhub/labelmaps/aiy_birds_V1_labelmap.csv')
#     response.raise_for_status()  # Raise an exception for HTTP errors
    
#     for line in response.text.splitlines():
#         if line.strip():
#             parts = line.strip().split(',')
#             if len(parts) > 1:
#                 labels.append(parts[1])
#             else:
#                 labels.append(f"Bird {line.strip()}")
    
#     print(f"Loaded {len(labels)} bird labels from official source")
# except Exception as e:
#     print(f"Warning: Could not load bird labels: {str(e)}")
#     # Create placeholder labels
#     labels = [f"Class {i}" for i in range(1000)]

# def preprocess_image(image_bytes):
#     """Convert image bytes to tensor with proper preprocessing"""
#     img = Image.open(io.BytesIO(image_bytes))
    
#     # Resize image to model input dimensions
#     img = img.resize((IMAGE_RES, IMAGE_RES))
    
#     # Convert to numpy array and normalize to [0,1]
#     img_array = np.array(img, dtype=np.float32) / 255.0  # Explicitly use float32
    
#     # Add batch dimension
#     img_tensor = np.expand_dims(img_array, axis=0)
    
#     # Convert to TensorFlow tensor with explicit float32 type
#     img_tensor = tf.convert_to_tensor(img_tensor, dtype=tf.float32)
    
#     return img_tensor

# @app.route('/classify', methods=['POST'])
# def classify_image():
#     if 'image' not in request.files and 'base64_image' not in request.json:
#         return jsonify({'error': 'No image provided'}), 400
    
#     try:
#         # Get image data either from file upload or base64
#         if 'image' in request.files:
#             image_bytes = request.files['image'].read()
#         else:
#             # Get base64 string and convert to bytes
#             base64_data = request.json['base64_image']
#             # Remove any header like "data:image/jpeg;base64,"
#             if ',' in base64_data:
#                 base64_data = base64_data.split(',')[1]
#             image_bytes = base64.b64decode(base64_data)
        
#         # Preprocess the image
#         img_tensor = preprocess_image(image_bytes)
        
#         # Run prediction
#         predictions = model.predict(img_tensor)

#         # In your classify_image endpoint, add this before calculating the top indices:
#         raw_predictions = predictions[0]
#         print(f"Raw prediction array shape: {raw_predictions.shape}")
#         print(f"Top 5 raw values: {np.sort(raw_predictions)[-5:]}")
#         print(f"Top 5 indices: {np.argsort(raw_predictions)[-5:]}")

#         # If possible, print the actual label for each of these top indices
#         for idx in np.argsort(raw_predictions)[-5:]:
#             if idx < len(labels):
#                 print(f"Index {idx}: {labels[idx]} - {raw_predictions[idx]}")
        
#         # Get top indices and convert to bird species
#         top_indices = np.argsort(predictions[0])[-10:][::-1]  # Top 10 results
        
#         # Format results
#         results = []
#         for idx in top_indices:
#             if idx < len(labels):
#                 species = labels[idx]
#             else:
#                 species = f"Class {idx}"
            
#             results.append({
#                 "species": species,
#                 "probability": float(predictions[0][idx])
#             })
        
#         # Return results sorted by probability
#         return jsonify({
#             "topResult": results[0]["species"],
#             "allResults": results
#         })
    
#     except Exception as e:
#         print(f"Error processing image: {str(e)}")
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8000)

# app.py - Using direct module loading for MobileNet
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import io
import base64
import json
import requests

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Use MobileNet model with direct module loading
IMAGE_RES = 224  # input dimensions required by the model
URL = 'https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/5'

print("Loading MobileNet classification model...")
try:
    # Load the model directly as a module
    hub_module = hub.load(URL)
    
    # Create a wrapper function for prediction
    def model_predict(image):
        # For MobileNet, we need to ensure the input is correctly formatted
        return hub_module(image)
    
    print("MobileNet model loaded successfully with direct module approach")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

# Load ImageNet labels
labels = []
try:
    # Download ImageNet labels
    response = requests.get('https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
    response.raise_for_status()
    
    # Parse labels - each line is a label
    for line in response.text.splitlines():
        if line.strip():
            labels.append(line.strip())
    
    print(f"Loaded {len(labels)} ImageNet labels")
except Exception as e:
    print(f"Warning: Could not load ImageNet labels: {str(e)}")
    # Create placeholder labels
    labels = [f"Class {i}" for i in range(1001)]

def preprocess_image(image_bytes):
    """Convert image bytes to tensor with proper preprocessing for MobileNet"""
    img = Image.open(io.BytesIO(image_bytes))
    
    # Resize image to model input dimensions
    img = img.resize((IMAGE_RES, IMAGE_RES))
    
    # Convert to numpy array and normalize to [0,1]
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Add batch dimension
    img_tensor = np.expand_dims(img_array, axis=0)
    
    # Convert to TensorFlow tensor with explicit float32 type
    img_tensor = tf.convert_to_tensor(img_tensor, dtype=tf.float32)
    
    return img_tensor

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files and 'base64_image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Get image data either from file upload or base64
        if 'image' in request.files:
            image_bytes = request.files['image'].read()
        else:
            # Get base64 string and convert to bytes
            base64_data = request.json['base64_image']
            # Remove any header like "data:image/jpeg;base64,"
            if ',' in base64_data:
                base64_data = base64_data.split(',')[1]
            image_bytes = base64.b64decode(base64_data)
        
        # Preprocess the image
        img_tensor = preprocess_image(image_bytes)
        
        # Run prediction using our wrapper function
        predictions = model_predict(img_tensor)
        
        # Convert to numpy for easier handling
        predictions_np = predictions.numpy()
        
        # Debug information
        raw_predictions = predictions_np[0]
        print(f"Raw prediction array shape: {raw_predictions.shape}")
        print(f"Top 5 raw values: {np.sort(raw_predictions)[-5:]}")
        print(f"Top 5 indices: {np.argsort(raw_predictions)[-5:]}")
        
        # Print the actual labels for top predictions
        top_indices = np.argsort(raw_predictions)[-5:][::-1]
        for idx in top_indices:
            if idx < len(labels):
                print(f"Index {idx}: {labels[idx]} - {raw_predictions[idx]}")
        
        # Format results
        results = []
        for idx in top_indices:
            if idx < len(labels):
                species = labels[idx]
            else:
                species = f"Class {idx}"
            
            results.append({
                "species": species,
                "probability": float(raw_predictions[idx])
            })
        
        # Return results sorted by probability
        return jsonify({
            "topResult": results[0]["species"],
            "allResults": results
        })
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)