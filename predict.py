import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the model
model = tf.keras.models.load_model('bird_classifier_model_200_species.h5')

# Load class names
with open('class_names.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

def predict_bird(image_path):
    # Load and preprocess the image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    predictions = model.predict(img_array)
    
    # Get top 5 predictions
    top_5_indices = np.argsort(predictions[0])[-5:][::-1]
    top_5_probabilities = predictions[0][top_5_indices]
    top_5_class_names = [class_names[i] for i in top_5_indices]
    
    # Format results
    results = []
    for i in range(5):
        species_name = top_5_class_names[i].replace('_', ' ').title()
        results.append({
            'species': species_name,
            'confidence': float(top_5_probabilities[i])
        })
    
    return results

def show_prediction(image_path, results):
    img = Image.open(image_path)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Input Image')
    
    plt.subplot(1, 2, 2)
    species = [r['species'] for r in results]
    confidence = [r['confidence'] for r in results]
    y_pos = np.arange(len(species))
    
    plt.barh(y_pos, confidence, align='center')
    plt.yticks(y_pos, species)
    plt.xlabel('Confidence')
    plt.title('Top 5 Predictions')
    plt.tight_layout()
    plt.show()

# Example usage
image_path = "pigeon_test1.jpg"  # Replace with your test image
results = predict_bird(image_path)
print("Top 5 predictions:")
for i, result in enumerate(results):
    print(f"{i+1}. {result['species']} - {result['confidence']:.2%}")

show_prediction(image_path, results)
