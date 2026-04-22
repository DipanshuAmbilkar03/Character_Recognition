import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from model_utils import load_model_compat

# Load the trained model
model = load_model_compat('char_recognition.keras')

def preprocess_custom_image(image_path, target_size=(28, 28)):
    # Load image while preserving alpha channel
    img = Image.open(image_path).convert('RGBA')
    
    # Create white background and paste the image
    background = Image.new('RGBA', img.size, (255, 255, 255))
    img = Image.alpha_composite(background, img).convert('L')
    
    # Invert colors (MNIST/EMNIST style: white on black)
    img = ImageOps.invert(img)
    
    # Resize with aspect ratio preservation
    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    
    # Create 28x28 canvas with black background
    processed_img = Image.new('L', target_size, 0)
    # Paste centered image
    processed_img.paste(img, (
        (target_size[0] - img.size[0]) // 2,
        (target_size[1] - img.size[1]) // 2
    ))
    
    # Convert to numpy array and normalize
    img_array = np.array(processed_img) / 255.0
    img_array = img_array.reshape(1, *target_size, 1)
    
    return img_array, processed_img

def predict_and_visualize(image_path):
    # Preprocess image
    img_array, processed_img = preprocess_custom_image(image_path)
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)
    
    # MNIST class mapping (10 classes)
    classes = [
        *'0123456789'
    ]
    
    # Visualization
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 3, 1)
    original_img = Image.open(image_path)
    plt.imshow(original_img)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(processed_img, cmap='gray')
    plt.title('Processed Input')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.barh(classes, predictions[0])
    plt.xlabel('Probability')
    plt.title('Prediction Probabilities')
    plt.tight_layout()
    
    print(f'Predicted character: {classes[predicted_class]}')
    print(f'Confidence: {confidence:.2%}')
    plt.show()

# Example usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help='Path to image file')
    args = parser.parse_args()
    
    predict_and_visualize(args.image_path)