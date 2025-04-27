import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from inference_sdk import InferenceHTTPClient

# Define paths
DATASET_PATH = r"D:\tumor detection\valid"
OUTPUT_DIR = Path("detection_results")

# Roboflow API setup
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="RAO3qcOxTrwgNcnSGrMD"
)

# Model ID
MODEL_ID = "brain-tumor-m2pbp/1"

# Class names (update if model classes differ)
CLASSES = ['no_tumor', 'glioma', 'meningioma', 'pituitary']

def predict_image(img_path, output_dir):
    """Predict and save result for a single image using Roboflow YOLOv5 model."""
    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Failed to load image: {img_path}")
        return None, []
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Perform inference
    try:
        result = CLIENT.infer(str(img_path), model_id=MODEL_ID)
    except Exception as e:
        print(f"Inference failed for {img_path}: {e}")
        return img_rgb, []
    
    # Process detections
    detections = []
    if 'predictions' in result and result['predictions']:
        print(f"Detections for {img_path.name}:")
        for pred in result['predictions']:
            class_name = pred.get('class', 'unknown')
            class_id = CLASSES.index(class_name) if class_name in CLASSES else -1
            confidence = pred.get('confidence', 0.0)
            if confidence > 0.3:  # Confidence threshold
                print(f" - Class: {class_name}, Confidence: {confidence:.2f}")
                x = pred.get('x', 0)
                y = pred.get('y', 0)
                width = pred.get('width', 0)
                height = pred.get('height', 0)
                # Calculate bounding box coordinates
                x1 = int(x - width / 2)
                y1 = int(y - height / 2)
                x2 = int(x + width / 2)
                y2 = int(y + height / 2)
                # Draw bounding box
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # Add label
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(img_rgb, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (255, 0, 0), 2)
                detections.append((class_id, confidence, [x1, y1, x2, y2]))
    else:
        print(f"No detections for {img_path.name}")
        cv2.putText(img_rgb, "No tumor detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (0, 0, 255), 2)
    
    # Save individual result
    output_path = output_dir / f"result_{img_path.name}"
    cv2.imwrite(str(output_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    print(f"Saved result to {output_path}")
    
    return img_rgb, detections

def display_all_results(images, detections_list, image_names):
    """Display all result images in a grid."""
    num_images = len(images)
    if num_images == 0:
        print("No valid images to display.")
        return
    
    cols = 3
    rows = (num_images + cols - 1) // cols
    
    plt.figure(figsize=(15, 5 * rows))
    for i, (img, detections, img_name) in enumerate(zip(images, detections_list, image_names)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(f"{img_name}\n{len(detections)} detections", fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('all_detections.png')
    plt.close()
    print("All results saved as 'all_detections.png'")

def main():
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Get all validation images
    val_images_path = Path(DATASET_PATH) / 'images'
    if not val_images_path.exists():
        raise FileNotFoundError(f"Images directory not found at {val_images_path}")
    
    image_files = list(val_images_path.glob("*.[jp][pn][gf]"))  # JPG, PNG
    if not image_files:
        raise FileNotFoundError("No images found in images directory")
    
    print(f"Found {len(image_files)} validation images.")
    
    # Process all images
    result_images = []
    detections_list = []
    image_names = []
    
    for img_path in image_files:
        print(f"\nProcessing image: {img_path}")
        img, detections = predict_image(img_path, OUTPUT_DIR)
        if img is not None:
            result_images.append(img)
            detections_list.append(detections)
            image_names.append(img_path.name)
    
    # Display all results
    display_all_results(result_images, detections_list, image_names)

if __name__ == "__main__":
    main()