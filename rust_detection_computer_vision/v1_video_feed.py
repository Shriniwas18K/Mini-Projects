import cv2
import numpy as np

def detect_corrosion(frame):
    """
    Detect corroded iron parts in the frame.
    
    Returns:
        List of corroded region contours
    """
    # Convert to grayscale and apply thresholding
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Edge detection (Canny)
    edges = cv2.Canny(thresh, 50, 150)

    # Contour detection
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Color filtering (HSV) for rust-colored areas
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_rust = np.array([0, 100, 100])
    upper_rust = np.array([20, 255, 255])
    rust_mask = cv2.inRange(hsv, lower_rust, upper_rust)

    # Combine edge detection and color filtering results
    corroded_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if cv2.mean(rust_mask[y:y+h, x:x+w])[0] > 50:  # Threshold for rust-colored area
            corroded_contours.append(contour)

    return corroded_contours

def capture_and_detect():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        corroded_contours = detect_corrosion(frame)

        # Draw contours around corroded regions
        corrosion_frame = frame.copy()
        cv2.drawContours(corrosion_frame, corroded_contours, -1, (0, 255, 0), 2)

        # Display original feed and corrosion detection output
        cv2.imshow('Original Feed', frame)
        cv2.imshow('Corrosion Detection', corrosion_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_detect()
