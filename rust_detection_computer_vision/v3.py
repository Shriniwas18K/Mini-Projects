import cv2
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk

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

def detect_corrosion(image_path):
    """
    Detect corroded iron parts in the image.

    Args:
        image_path (str): Path to the input image.
    """
    # Read the image
    image = cv2.imread(image_path)

    # Balance lighting conditions
    image_balanced = balance_lighting(image)

    # Create rust mask
    rust_mask = create_rust_mask(image_balanced)

    # Create threshold image
    thresh_image = apply_thresholding(image_balanced)

    # Display results
    # cv2.imshow('Rust Mask', rust_mask)
    # cv2.imshow('Gaussian Adaptive Thresholding', thresh_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    show_in_gui([image_balanced,rust_mask,thresh_image])

def show_in_gui(images):
        root = tk.Tk()
        root.title("Image Viewer")

        # Create a frame to hold the images
        frame = tk.Frame(root)
        frame.pack(fill="both", expand=True)
    
        for i, image in enumerate(images):
            resized_image = cv2.resize(image, None, fx=0.3, fy=0.3)
            tk_image = ImageTk.PhotoImage(Image.fromarray(resized_image))
    
            # Create a label to display the image
            label = tk.Label(frame, image=tk_image)
            label.image = tk_image  # Keep a reference to prevent garbage collection
    
            # Arrange the images horizontally
            label.pack(side="left", padx=10, pady=10)
        root.mainloop()
        
if __name__ == "__main__":
    image_path = 'corrosion_image.jpg'  # Replace with your image path
    detect_corrosion(image_path)
