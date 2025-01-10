import cv2
import pytesseract
import numpy as np

# Specify the path to Tesseract-OCR if not added to PATH
# Example for Windows:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image_for_ocr(image):
    """Preprocess the image for OCR in challenging lighting conditions."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Apply bilateral filtering to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(enhanced, 11, 17, 17)

    # Thresholding for better separation of regions
    _, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh

def detect_number_plate(image, debug=False):
    """Detect the number plate region and return the cropped plate."""
    # Preprocess the image
    preprocessed_image = preprocess_image_for_ocr(image)

    # Apply edge detection
    edged = cv2.Canny(preprocessed_image, 50, 200)

    if debug:
        cv2.imshow("Edges", edged)
        cv2.waitKey(0)

    # Find contours in the edged image
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    for contour in contours:
        # Approximate the contour to simplify its shape
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # Check for a rectangle with four corners
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            # Typical aspect ratio and size constraints for number plates
            if 2 < aspect_ratio < 6 and 50 < w < 1000 and 20 < h < 300:
                if debug:
                    cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)
                    cv2.imshow("Detected Plate", image)
                    cv2.waitKey(0)
                return image[y:y + h, x:x + w], (x, y, w, h)

    return None, None

def recognize_number_plate(image_path, debug=False):
    """Recognize text from the detected number plate."""
    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Failed to load the image. Please check the file path.")
        return

    # Detect the number plate
    cropped_number_plate, bbox = detect_number_plate(image, debug=debug)

    if cropped_number_plate is None:
        print("Number plate not detected. Trying entire image as fallback.")
        cropped_number_plate = preprocess_image_for_ocr(image)

    # Ensure the image is grayscale before further processing
    if len(cropped_number_plate.shape) == 3:  # Check if it's a color image
        cropped_number_plate = cv2.cvtColor(cropped_number_plate, cv2.COLOR_BGR2GRAY)

    # Threshold the image for OCR
    _, cropped_number_plate = cv2.threshold(cropped_number_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if debug:
        cv2.imshow("Cropped Plate for OCR", cropped_number_plate)
        cv2.waitKey(0)

    # Use Tesseract OCR to extract text
    custom_config = r'--oem 3 --psm 8'
    extracted_text = pytesseract.image_to_string(cropped_number_plate, config=custom_config).strip()

    # Extract alphanumeric characters and format them
    if extracted_text:
        formatted_text = ''.join(filter(str.isalnum, extracted_text))
    else:
        formatted_text = "Not Detected"

    # Print the extracted text to the console
    print("Extracted Number Plate Text:", extracted_text)
    print("Formatted Number Plate:", formatted_text)

    if bbox:
        x, y, w, h = bbox
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw a blue bounding box
        cv2.putText(image, f"detected: {formatted_text}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


    # Save the output image with annotations
    output_path = "output_with_number_plate.jpeg"
    cv2.imwrite(output_path, image)
    print(f"Output saved as '{output_path}' with detected text: {formatted_text}")

# Test the function with the uploaded image
image_path = r"C:\Users\username\OneDrive\Desktop\Green-number-plates.jpg"
recognize_number_plate(image_path, debug=True)
