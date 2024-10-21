import os
import cv2
import pytesseract
import numpy as np
from PIL import Image, ImageFilter
from ultralytics import YOLO
import easyocr
import cv2
import matplotlib.pyplot as plt



# Load the model
model_path = r"C:\Users\subhs\OneDrive\Documents\pythonProjects\CarLicensePlate\automatic-number-plate-recognition-python-yolov8\models\best.pt"
model = YOLO(model_path)

# Set the confidence threshold 
confidence_threshold = 0.27

# Predict on a single image
image_path = 'image4.jpg'
results = model.predict(source=image_path, imgsz=640, conf=confidence_threshold)

# Create results directory if it doesn't exist
results_dir = r'testing_plate\results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Process results
for i, result in enumerate(results):
    # Save the result image
    save_path = os.path.join(results_dir, f'result_{i}.jpg')
    result.save(save_path)
    
    # Extract the bounding boxes
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get coordinates
        confidence = box.conf[0]  # Get confidence score
        
        if confidence >= confidence_threshold:
            # Read the image using OpenCV
            image = cv2.imread(image_path)

            # Crop the license plate from the image
            cropped_plate = image[y1:y2, x1:x2]

            # Optionally, save the cropped image
            cropped_path = os.path.join(results_dir, f'cropped_plate_{i}.jpg')
            cv2.imwrite(cropped_path, cropped_plate)
            


            # Initialize EasyOCR reader for English language (you can add more languages if needed)
reader = easyocr.Reader(['en'])

# Load the image of the license plate
#image_path = 'demo.jpg'
image = cv2.imread(save_path)

# Use EasyOCR to read text from the image
results = reader.readtext(save_path)

# Display the image with bounding boxes around detected text
for (bbox, text, prob) in results:
    # Unpack the bounding box
    (top_left, top_right, bottom_right, bottom_left) = bbox
    top_left = tuple([int(val) for val in top_left])
    bottom_right = tuple([int(val) for val in bottom_right])

    # Draw the bounding box and text on the image
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    cv2.putText(image, text, (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Show the result with detected text
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# Print the recognized text
for res in results:
    print(f"Detected Text: {res[1]} with confidence {res[2]:.2f}")
