# Brain-Tumor-Detection
BrainTumorDetection-YOLOv5-Binary

This project uses a YOLOv5 model hosted on Roboflow to detect brain tumors in MRI images, classifying them as with tumor or without tumor. The model (brain-tumor-m2pbp/1) processes images from a preprocessed dataset, draws bounding boxes around detected tumors, and displays results in a grid.

Features





Performs object detection using the Roboflow-hosted YOLOv5 model (brain-tumor-m2pbp/1).



Processes all images in the validation dataset (D:\tumor detection\valid\images).



Visualizes detections with red bounding boxes and labels (class name and confidence).



Saves individual results and a combined grid of all detections.



Includes debugging to diagnose detection issues (e.g., no detections).

Dataset

The dataset is located at D:\tumor detection\valid and has the following structure:

valid/
├── images/
│   ├── image1.jpg
│   ├── image2.png
│   ├── ...
├── labeltxt/
│   ├── image1.txt
│   ├── image2.txt
│   ├── ...





images: Contains MRI images (JPG or PNG).



labeltxt: Contains YOLO-format .txt files with annotations (<class_id> <x_center> <y_center> <width> <height>), where class IDs represent with tumor (e.g., 0) or without tumor (e.g., 1).

The dataset was used to train the Roboflow YOLOv5 model (brain-tumor-m2pbp/1), which performs binary classification.

Prerequisites





Python 3.8 or higher



Internet connection (for Roboflow API)



Roboflow API key (provided in the script)



Dataset at D:\tumor detection\valid

Installation





Clone the repository:

git clone https://github.com/<your-username>/BrainTumorDetection-YOLOv5-Binary.git
cd BrainTumorDetection-YOLOv5-Binary



Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate



Install dependencies:

pip install -r requirements.txt



Verify dataset: Ensure the dataset is at D:\tumor detection\valid with images and labeltxt folders.

Usage





Run the detection script:

python src/detect_brain_tumor_roboflow_yolov5.py





The script processes all images in D:\tumor detection\valid\images.



Outputs:





Individual results: detection_results/result_<image_name>



Combined grid: all_detections.png



Console output: Detection details for each image



Customize (optional):





Update CLASSES in the script if the Roboflow model uses different class names (e.g., ['tumor', 'no_tumor']).



Modify the confidence threshold (0.3) in predict_image for sensitivity.



Process a specific image by editing image_files in main:

image_files = [Path(r"D:\tumor detection\valid\images\your_image.jpg")]

Project Structure

BrainTumorDetection-YOLOv5-Binary/
├── src/
│   └── detect_brain_tumor_roboflow_yolov5.py  # Main detection script
├── detection_results/                        # Output directory for results
├── README.md                                # Project documentation
├── requirements.txt                         # Dependencies

Troubleshooting

If no tumors are detected or bounding boxes are missing:





Check Class Names:





Verify the Roboflow model’s classes in the brain-tumor-m2pbp project (e.g., via Roboflow dashboard or inference response).



Update CLASSES in the script if they differ (e.g., ['tumor', 'no_tumor']).



Inspect the inference response:

result = CLIENT.infer(str(img_path), model_id=MODEL_ID)
print([pred['class'] for pred in result.get('predictions', [])])



API Issues:





Ensure the API key (RAO3qcOxTrwgNcnSGrMD) is valid (check Roboflow workspace settings).



Confirm the model ID (brain-tumor-m2pbp/1) is correct and deployed.



Image Compatibility:

Verify images in D:\tumor detection\valid\images are JPG/PNG and match the training data (e.g., similar preprocessing, contrast).

Check labels in labeltxt for correct class IDs (e.g., 0 for with tumor, 1 for without tumor).



Lower Confidence:





Change if confidence > 0.3 to if confidence > 0.1 in the script to capture more detections.



Roboflow Model:





Check the brain-tumor-m2pbp project in Roboflow for training details (e.g., dataset size, annotations).



Retrain if the model isn’t performing well.
