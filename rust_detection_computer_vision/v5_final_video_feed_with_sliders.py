import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np


# Image Preprocessing
# ------------------------

def balance_lighting(image, clip_limit, tile_grid_size, gamma):
    """
    Enhance image lighting using CLAHE and gamma correction.

    Args:
        image (numpy array): Input image.
        clip_limit (float): CLAHE clip limit.
        tile_grid_size (tuple): CLAHE tile grid size.
        gamma (float): Gamma correction value.

    Returns:
        numpy array: Image with balanced lighting.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    v_clahe = clahe.apply(v)
    hsv_clahe = cv2.merge([hsv[:, :, 0], hsv[:, :, 1], v_clahe])
    image_clahe = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)
    image_gamma = np.power(image_clahe / 255.0, gamma) * 255.0
    image_gamma = image_gamma.astype(np.uint8)
    return image_gamma


# Corrosion Detection
# ----------------------

def create_rust_mask(image, lower_rust, upper_rust):
    """
    Create a mask for rust-colored areas (HSV).

    Args:
        image (numpy array): Input image.
        lower_rust (numpy array): Lower HSV range for rust.
        upper_rust (numpy array): Upper HSV range for rust.

    Returns:
        numpy array: Rust mask.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, lower_rust, upper_rust)


def apply_thresholding(image, block_size):
    """
    Convert image to grayscale, apply Gaussian adaptive thresholding, and convert back to RGB.

    Args:
        image (numpy array): Input image (RGB or BGR).
        block_size (int): Thresholding block size.

    Returns:
        numpy array: Thresholded image (RGB, 3-channel).
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 2)
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

        # Create top frame for title and submission text
        top_frame = tk.Frame(self.root)
        top_frame.pack(fill="x")

        title_label = tk.Label(top_frame, text="Mini Project Submission for CV", font=("Arial", 24))
        title_label.pack(pady=20)

        submission_label = tk.Label(top_frame, text="Rust Detection in Live Feed", font=("Arial", 18))
        submission_label.pack(pady=10)
        
        # Initialize parameters
        self.clip_limit = 2.0
        self.tile_grid_size = (8, 8)
        self.gamma = 1.5
        self.lower_rust = np.array([0, 100, 100])
        self.upper_rust = np.array([20, 255, 255])
        self.block_size = 11

        # Create sliders
        slider_frame = tk.Frame(self.root)
        slider_frame.pack(fill="x")

        tk.Label(slider_frame, text="CLAHE Clip Limit").pack(side="left")
        self.clip_limit_slider = tk.Scale(slider_frame, from_=0.0, to=10.0, resolution=0.1, orient="horizontal")
        self.clip_limit_slider.set(self.clip_limit)
        self.clip_limit_slider.pack(side="left")

        tk.Label(slider_frame, text="CLAHE Tile Grid Size X").pack(side="left")
        self.tile_grid_size_slider_x = tk.Scale(slider_frame, from_=1, to=32, resolution=1, orient="horizontal")
        self.tile_grid_size_slider_x.set(self.tile_grid_size[0])
        self.tile_grid_size_slider_x.pack(side="left")

        tk.Label(slider_frame, text="CLAHE Tile Grid Size Y").pack(side="left")
        self.tile_grid_size_slider_y = tk.Scale(slider_frame, from_=1, to=32, resolution=1, orient="horizontal")
        self.tile_grid_size_slider_y.set(self.tile_grid_size[1])
        self.tile_grid_size_slider_y.pack(side="left")

        tk.Label(slider_frame, text="Gamma").pack(side="left")
        self.gamma_slider = tk.Scale(slider_frame, from_=0.1, to=5.0, resolution=0.1, orient="horizontal")
        self.gamma_slider.set(self.gamma)
        self.gamma_slider.pack(side="left")

        slider_frame = tk.Frame(self.root)
        slider_frame.pack(fill="x")

        tk.Label(slider_frame, text="Lower Rust Hue").pack(side="left")
        self.lower_rust_hue_slider = tk.Scale(slider_frame, from_=0, to=180, resolution=1, orient="horizontal")
        self.lower_rust_hue_slider.set(self.lower_rust[0])
        self.lower_rust_hue_slider.pack(side="left")

        tk.Label(slider_frame, text="Lower Rust Saturation").pack(side="left")
        self.lower_rust_saturation_slider = tk.Scale(slider_frame, from_=0, to=255, resolution=1, orient="horizontal")
        self.lower_rust_saturation_slider.set(self.lower_rust[1])
        self.lower_rust_saturation_slider.pack(side="left")

        tk.Label(slider_frame, text="Lower Rust Value").pack(side="left")
        self.lower_rust_value_slider = tk.Scale(slider_frame, from_=0, to=255, resolution=1, orient="horizontal")
        self.lower_rust_value_slider.set(self.lower_rust[2])
        self.lower_rust_value_slider.pack(side="left")

        slider_frame = tk.Frame(self.root)
        slider_frame.pack(fill="x")

        tk.Label(slider_frame, text="Upper Rust Hue").pack(side="left")
        self.upper_rust_hue_slider = tk.Scale(slider_frame, from_=0, to=180, resolution=1, orient="horizontal")
        self.upper_rust_hue_slider.set(self.upper_rust[0])
        self.upper_rust_hue_slider.pack(side="left")

        tk.Label(slider_frame, text="Upper Rust Saturation").pack(side="left")
        self.upper_rust_saturation_slider = tk.Scale(slider_frame, from_=0, to=255, resolution=1, orient="horizontal")
        self.upper_rust_saturation_slider.set(self.upper_rust[1])
        self.upper_rust_saturation_slider.pack(side="left")

        tk.Label(slider_frame, text="Upper Rust Value").pack(side="left")
        self.upper_rust_value_slider = tk.Scale(slider_frame, from_=0, to=255, resolution=1, orient="horizontal")
        self.upper_rust_value_slider.set(self.upper_rust[2])
        self.upper_rust_value_slider.pack(side="left")

        tk.Label(slider_frame, text="Thresholding Block Size").pack(side="left")
        self.block_size_slider = tk.Scale(slider_frame, from_=1, to=31, resolution=2, orient="horizontal")
        self.block_size_slider.set(self.block_size)
        self.block_size_slider.pack(side="left")

    def detect_corrosion(self):
        ret, frame = self.video_capture.read()
        if not ret:
            return

        self.clip_limit = float(self.clip_limit_slider.get())
        self.tile_grid_size = (int(self.tile_grid_size_slider_x.get()), int(self.tile_grid_size_slider_y.get()))
        self.gamma = float(self.gamma_slider.get())
        self.lower_rust = np.array([int(self.lower_rust_hue_slider.get()), int(self.lower_rust_saturation_slider.get()), int(self.lower_rust_value_slider.get())])
        self.upper_rust = np.array([int(self.upper_rust_hue_slider.get()), int(self.upper_rust_saturation_slider.get()), int(self.upper_rust_value_slider.get())])
        self.block_size = int(self.block_size_slider.get())

        image_balanced = balance_lighting(frame, self.clip_limit, self.tile_grid_size, self.gamma)
        rust_mask = create_rust_mask(image_balanced, self.lower_rust, self.upper_rust)
        thresh_image = apply_thresholding(image_balanced, self.block_size)

        # Display results
        images = [frame, image_balanced, rust_mask, thresh_image]
        self.show_in_gui(images)

        self.root.after(10, self.detect_corrosion)

    def show_in_gui(self, images):
        for widget in self.frame.winfo_children():
            widget.destroy()

        for i, image in enumerate(images):
            resized_image = cv2.resize(image, None, fx=0.3, fy=0.3)
            image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            tk_image = ImageTk.PhotoImage(Image.fromarray(image_rgb))

            # Create a label to display the image
            label = tk.Label(self.frame, image=tk_image)
            label.image = tk_image  # Keep a reference to prevent garbage collection

            # Arrange the images horizontally
            label.pack(side="left", padx=10, pady=10)

    def run(self):
        self.detect_corrosion()
        self.root.mainloop()


if __name__ == "__main__":
    video_capture = cv2.VideoCapture(0)                                             
    detector = CorrosionDetector(video_capture)
    detector.run()
