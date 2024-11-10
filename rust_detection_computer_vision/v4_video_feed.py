import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

# Image Preprocessing
# ------------------------

def balance_lighting(image):
    """
    Enhance image lighting using CLAHE and gamma correction.

    Args:
        image (numpy array): Input image.

    Returns:
        numpy array: Image with balanced lighting.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_clahe = clahe.apply(v)
    hsv_clahe = cv2.merge([hsv[:, :, 0], hsv[:, :, 1], v_clahe])
    image_clahe = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)
    gamma = 1.5
    image_gamma = np.power(image_clahe / 255.0, gamma) * 255.0
    image_gamma = image_gamma.astype(np.uint8)
    return image_gamma


# Corrosion Detection
# ----------------------

def create_rust_mask(image):
    """
    Create a mask for rust-colored areas (HSV).

    Args:
        image (numpy array): Input image.

    Returns:
        numpy array: Rust mask.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_rust = np.array([0, 100, 100])
    upper_rust = np.array([20, 255, 255])
    return cv2.inRange(hsv, lower_rust, upper_rust)


def apply_thresholding(image):
    """
    Convert image to grayscale, apply Gaussian adaptive thresholding, and convert back to RGB.

    Args:
        image (numpy array): Input image (RGB or BGR).

    Returns:
        numpy array: Thresholded image (RGB, 3-channel).
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return cv2.cvtColor(thresh_image, cv2.COLOR_GRAY2RGB)


# Main Function
# ----------------

class CorrosionDetector:
    def __init__(self, video_capture):
        self.video_capture = video_capture
        self.root = tk.Tk()
        self.root.title("Corrosion Detector")
        self.frame = tk.Frame(self.root)
        self.frame.pack(fill="both", expand=True)

    def detect_corrosion(self):
        ret, frame = self.video_capture.read()
        if not ret:
            return

        image_balanced = balance_lighting(frame)
        rust_mask = create_rust_mask(image_balanced)
        thresh_image = apply_thresholding(image_balanced)

        # Display results
        images = [image_balanced, rust_mask, thresh_image]
        self.show_in_gui(images)

        self.root.after(10, self.detect_corrosion)

    def show_in_gui(self, images):
        for widget in self.frame.winfo_children():
            widget.destroy()

        for i, image in enumerate(images):
            resized_image = cv2.resize(image, None, fx=0.3, fy=0.3)
            tk_image = ImageTk.PhotoImage(Image.fromarray(resized_image))

            # Create a label to display the image
            label = tk.Label(self.frame, image=tk_image)
            label.image = tk_image  # Keep a reference to prevent garbage collection

            # Arrange the images horizontally
            label.pack(side="left", padx=10, pady=10)

    def run(self):
        self.detect_corrosion()
        self.root.mainloop()


if __name__ == "__main__":
    video_capture = cv2.VideoCapture(0)  # Use camera index 0
    detector = CorrosionDetector(video_capture)
    detector.run()
