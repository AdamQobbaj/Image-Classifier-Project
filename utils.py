import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import json
from PIL import Image

IMAGE_SIZE = 224

def load_keras_model(model_path):
    """
    Loads a Keras model, handling the custom KerasLayer from TensorFlow Hub.

    Args:
        model_path (str): Path to the saved Keras model file (.h5 or .keras).

    Returns:
        tf.keras.Model: The loaded model object.
    """
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'KerasLayer': hub.KerasLayer},
            compile=False # Don't recompile if not needed for inference
        )
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None

def process_image(image_path):
    """
    Loads an image, pre-processes it, and returns it ready for prediction.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: The processed image with shape (1, 224, 224, 3) and 
                    normalized pixel values (0-1).
    """
    try:
        # 1. Load the image using PIL and convert to NumPy array
        img = Image.open(image_path)
        image_np_array = np.asarray(img)
        
        # 2. Convert to a TensorFlow Tensor for easy resizing
        image = tf.convert_to_tensor(image_np_array, dtype=tf.float32)

        # 3. Resize to the target size (224, 224)
        image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

        # 4. Normalize pixel values (0-255 to 0-1)
        image /= 255.0
        
        # 5. Add the batch dimension (from (224, 224, 3) to (1, 224, 224, 3))
        input_tensor = tf.expand_dims(image.numpy(), axis=0)
        
        return input_tensor
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def predict(image_tensor, model, top_k):
    """
    Predicts the top K classes for a processed image tensor.

    Args:
        image_tensor (tf.Tensor or np.ndarray): The processed image ready for inference.
        model (tf.keras.Model): The trained Keras model.
        top_k (int): The number of top probabilities/classes to return.

    Returns:
        tuple: (probs, classes) - A tuple containing two lists:
               - probs (list): Top K probabilities (floats).
               - classes (list): Top K class labels (original dataset labels '1'-'102' as strings).
    """
    # Make the prediction
    # model.predict returns an array of probabilities (shape: (1, num_classes))
    probabilities = model.predict(image_tensor, verbose=0)[0]
    
    # Get the indices of the top K probabilities
    top_k_indices = np.argsort(probabilities)[::-1][:top_k]
    
    # Get the corresponding probabilities
    top_k_probs = probabilities[top_k_indices].tolist()
    
    # Convert model indices (0-101) back to original dataset labels (1-102)
    # and convert them to strings to match the required output format (e.g., '70')
    top_k_classes_labels = (top_k_indices + 1).astype(str).tolist()

    return top_k_probs, top_k_classes_labels

def load_class_names(json_path):
    """
    Loads the label-to-name mapping from a JSON file.

    Args:
        json_path (str): Path to the category names JSON file.

    Returns:
        dict: The loaded dictionary mapping string labels to flower names.
    """
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Category names file not found at {json_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not parse JSON file at {json_path}")
        return None