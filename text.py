import easyocr
import cv2
import matplotlib.pyplot as plt

# Initialize EasyOCR reader for English language (you can add more languages if needed)
reader = easyocr.Reader(['en'])

# Load the image of the license plate
image_path = 'demo.jpg'
image = cv2.imread(image_path)

# Use EasyOCR to read text from the image
results = reader.readtext(image_path)

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