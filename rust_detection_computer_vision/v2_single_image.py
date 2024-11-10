import cv2
import numpy as np

def detect_corrosion(image_path):
    """
    Detect corroded iron parts in the image.
    
    Args:
        image_path (str): Path to the input image.
    
    Returns:
        None
    """
    # Read the image
    image = cv2.imread(image_path)

    # Convert to grayscale and apply thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Edge detection (Canny)
    edges = cv2.Canny(thresh, 50, 150)

    # Contour detection
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Color filtering (HSV) for rust-colored areas
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_rust = np.array([0, 100, 100])
    upper_rust = np.array([20, 255, 255])
    rust_mask = cv2.inRange(hsv, lower_rust, upper_rust)

    # Combine edge detection and color filtering results
    corroded_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if cv2.mean(rust_mask[y:y+h, x:x+w])[0] > 50:  # Threshold for rust-colored area
            corroded_contours.append(contour)

    # Draw contours around corroded regions
    corrosion_image = image.copy()
    cv2.drawContours(corrosion_image, corroded_contours, -1, (0, 255, 0), 2)

    # Display original image and corrosion detection output
    cv2.imshow('Original Image', image)
    cv2.imshow('Corrosion Detection', corrosion_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = 'corrosion_image.jpg'  # Replace with your image path
detect_corrosion(image_path)
