import argparse
import sys
import numpy as np
import os

# Import functions from the utility module
try:
    from utils import load_keras_model, process_image, predict, load_class_names
except ImportError:
    # Fallback for environments where utils.py might be in the same script/notebook
    print("Warning: Could not import utility functions from 'utils.py'. Ensure the file exists.")
    sys.exit(1)

def main():
    """
    Main function to parse arguments, run inference, and print results.
    """
    parser = argparse.ArgumentParser(
        description='Predict the top K flower classes from an image using a trained Keras model.',
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Required positional arguments
    parser.add_argument(
        'image_path', 
        type=str, 
        help='Path to the input image file (e.g., ./test_images/orchid.jpg)'
    )
    parser.add_argument(
        'model_path', 
        type=str, 
        help='Path to the saved Keras model file (e.g., my_model.h5 or my_model.keras)'
    )

    parser.add_argument(
        '--top_k', 
        type=int, 
        default=1, 
        help='Return the top K most likely classes (default: 1)'
    )
    parser.add_argument(
        '--category_names', 
        type=str, 
        default=None, 
        help='Path to a JSON file mapping labels to flower names (e.g., label_map.json)'
    )

    args = parser.parse_args()
    
    # --- 1. Load Model ---
    model = load_keras_model(args.model_path)
    if model is None:
        sys.exit(1)
        
    # --- 2. Process Image ---
    image_tensor = process_image(args.image_path)
    if image_tensor is None:
        sys.exit(1)

    # --- 3. Run Prediction ---
    probs, classes = predict(image_tensor, model, args.top_k)

    # --- 4. Load Category Names (Optional) ---
    cat_to_name = None
    if args.category_names:
        cat_to_name = load_class_names(args.category_names)

    # --- 5. Print Results ---
    print(f"\nPrediction for: {args.image_path} (Using model: {os.path.basename(args.model_path)})")
    print("-" * 50)
    
    if cat_to_name:
        print(f"Top {args.top_k} Predictions:")
        for prob, label in zip(probs, classes):
            # Look up name using the string label (e.g., '70')
            name = cat_to_name.get(label, f"Unknown Label {label}")
            print(f"  {name:<30} (Label: {label}): {prob*100:.3f}%")
    else:
        # Basic output if category names are not provided
        print(f"Probabilities (Probs):\n  {np.array(probs)}")
        print(f"Class Labels (Classes):\n  {classes}")

if __name__ == '__main__':
    main()