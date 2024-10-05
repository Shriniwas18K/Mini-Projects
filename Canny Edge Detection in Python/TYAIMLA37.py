import numpy as np
import warnings
from typing import Tuple
from PIL import Image
warnings.filterwarnings("ignore")

'''This module was developed as submission for FA1 Computer Vision
   for the topic canny edge detection and gaussian high pass filtering.
   Currently this has been formulated as ImageOperations library with
   driver code at bottom, it majorly uses numpy for all operations
   owing to computational efficiency using C low level routines.It 
   currently supports only canny edge detection and gaussian,butterworth,
   ideal high pass and low pass filtering methods 
   
   Author : Shriniwas Kulkarni (122B1C037) (TYAIMLA37)
   Date of starting : 17 Sept 2024
   Date of completion : 20 Sept 2024
   Dependencies : Python >=3.12.0 , Pillow >=10.4.0 , Numpy >=2.1.
   
   Recommendations : Pls hover over function names using ide's intellisense
   to see docs of function or help(ImageOperations.<function_name>) in shell
'''


class ImageOperations:
    """
    A class for performing various image operations.\n\n

    This class provides a collection of static methods for image processing tasks,
    including image filtering, edge detection, and image transformation.\n

    Methods:\n
    --------
    save_image(img, filename) : Save a NumPy array as an image file.\n
    convert_rgb_to_grayscale(img, method) : Convert an RGB image to grayscale.\n
    __generate_kernel__(kernel_name, kernel_size, kernel_coefficient) : Generate a kernel for image filtering.\n
    gaussian_blur(img, size, sigma) : Apply a Gaussian blur to an image.\n
    __sobel_gradient_magnitude_and_orientation_approximation__(img, axis, size) : Approximate gradient magnitude and orientation using Sobel operators.
    __apply_non_max_suppression__(magnitude, orientation) : Apply non-maximum suppression to gradient magnitude.\n
    __apply_edge_tracking_by_hysteresis__(magnitude, low_threshold, high_threshold) : Apply edge tracking by hysteresis.\n
    canny_edge_detection(image, low_threshold, high_threshold, gaussian_kernel_size, sobel_kernel_size, sigma) : Apply Canny edge detection algorithm.\n
    frequency_domain_filtering(img,filter,threshold,mode) : Apply gaussian,buterworth,ideal low and high pass frequency domain filtering methods.\n
    \n
    Notes:\n
    -----
    All methods are static, so they can be called without creating an instance of the class.\n
    """
    @staticmethod
    def save_image( img: np.ndarray, filename: str) -> None:
        """
        Save a NumPy array as an image file.\n\n

        Args:\n
        - img (np.ndarray): The image data to be saved.\n
        - filename (str): The filename for the saved image.\n
        \n
        Returns:\n
        - None\n
        """
        Image.fromarray(img).save(filename)

    @staticmethod
    def convert_rgb_to_grayscale( img: np.ndarray, method: str) -> np.ndarray:
        """
        Convert an RGB image to grayscale using the specified methods.\n
        Methods respectively refer to :\n
        1] avg : Taking average of three channel values of pixel intensities.\n
        2] lum : Apply luminosity method.\n
        \n
        Args:\n
        - img (np.ndarray): Input RGB image.\n
        - method (str): Grayscale conversion method ('avg' or 'lum').\n
        \n
        Returns:\n
        - np.ndarray: Grayscale image.\n
        """
        __gray__ = np.zeros(shape=(img.shape[0], img.shape[1]), dtype=np.uint8)
        __height__, __width__ = img.shape[0],img.shape[1]

        match method:
            case "avg":
                for __row__ in range(__height__):
                    for __col__ in range(__width__):
                        __gray__[__row__][__col__] = np.average(img[__row__][__col__])
                return __gray__

            case "lum":
                for __row__ in range(__height__):
                    for __col__ in range(__width__):
                        __R__, __G__, __B__ = img[__row__][__col__]
                        __gray__[__row__][__col__] = 0.299 * __R__ + 0.587 * __G__ + 0.114 * __B__
                return __gray__
            case _:
                raise ValueError("Invalid method. Supported methods are 'avg' and 'lum' only.")       
    
    @staticmethod
    def __generate_kernel__( kernel_name: str, kernel_size: Tuple[int, int], kernel_coefficient: float,
        ) -> np.ndarray:
        '''
        Generates a kernel based on the specified name and parameters.\n
        \n
        Args:\n
        - kernel_name (str): Name of the kernel ('gauss' or 'sobel').\n
        - size (Tuple[int, int]): Size of the kernel.\n
        - kernel_coefficient (float): Sigma for Gaussian kernel, axis value (0 or 1) for Sobel kernel.\n
        \n
        Returns:\n
        - np.ndarray: Generated kernel.\n
        '''
        match kernel_name:
            case "gauss":
            # Define Gaussian kernel function
                def __gaussian__(x: float, y: float, sigma: float) -> float:
                    return (1 / (2 * np.pi * sigma ** 2)) * np.exp(
                        -((x - (kernel_size[0] - 1) / 2) ** 2 + (y - (kernel_size[1] - 1) / 2) ** 2) / (2 * sigma ** 2)
                    )

            # Generate Gaussian kernel
                kernel = np.fromfunction(
                    lambda x, y: __gaussian__(x=x, y=y, sigma=kernel_coefficient),
                    shape=kernel_size,
                    dtype=np.float32
                )
                return kernel / np.sum(kernel)

            case "sobel":
                # Validate Sobel kernel coefficient
                if kernel_coefficient not in [0, 1]:
                    raise ValueError("Sobel kernel coefficient must be 0 (x-axis) or 1 (y-axis).")

                # Define Sobel kernel for 3x3 size
                sobel_kernels = {
                    0: np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
                    1: np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
                }

                # Check if size is supported
                if kernel_size != (3, 3):
                    raise NotImplementedError("Sobel kernel is only implemented for 3x3 size.")

                return sobel_kernels[kernel_coefficient]

            case _:
                raise NotImplementedError("Only Gaussian and Sobel kernels are supported.")

    @staticmethod
    def gaussian_blur( img: np.ndarray, size: Tuple[int, int], sigma: float) -> np.ndarray:
        """
        Applies a Gaussian blur to the input image.\n
        \n
        Args:\n
        - img (np.ndarray): Grayscale Input image.\n
        - size (Tuple[int, int]): Size of the Gaussian kernel.\n
        - sigma (float): Standard deviation of the Gaussian distribution.\n
        \n
        Returns:\n
        - np.ndarray: Grayscale Blurred image.\n
        """
        
        # Check if the input image is grayscale
        if len(img.shape) != 2:
            raise ValueError("Input image must be a single-channel grayscale image.")
        
        # Ensure the kernel size is odd for symmetry
        if size[0] % 2 == 0:
            size = (size[0] + 1, size[1] + 1)

        # Generate the Gaussian kernel
        gaussian_kernel =ImageOperations.__generate_kernel__(
            kernel_name="gauss",
            kernel_size=size,
            kernel_coefficient=sigma
        )


        # Get the half-size of the kernel for padding
        k_half = size[0] // 2

        # Create an output image with the same dimensions as the input
        output = np.zeros_like(img)

        # Apply the Gaussian blur by convolving the image with the kernel
        for i in range(k_half, img.shape[0] - k_half):
            for j in range(k_half, img.shape[1] - k_half):
                output[i, j] = np.sum(
                    img[i - k_half:i + k_half + 1, j - k_half:j + k_half + 1] 
                    * gaussian_kernel
                )

        # Return the blurred image without padding
        return output[k_half:img.shape[0] - k_half, k_half:img.shape[1] - k_half]
   
    @staticmethod
    def __sobel_gradient_magnitude_and_orientation_approximation__( img: np.ndarray, size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Approximate first-order derivative of pixel intensities using Sobel operator.\n
        \n
        Args:\n
        - img (np.ndarray): Input image (grayscale, single-channel).\n
        - size (Tuple[int, int]): Kernel size (default: (3, 3)).\n
        \n
        Returns:\n
        - magnitude (np.ndarray): Gradient magnitude.\n
        - orientation (np.ndarray): Gradient orientation (in radians).\n
        \n
        Raises:\n
        - ValueError: If input image is not a single-channel grayscale image.\n
        """

        # Validate input image
        if len(img.shape) != 2:
            raise ValueError("Input image must be a single-channel grayscale image.")

        # Define Sobel kernels (3x3)
        sobel_x = ImageOperations.__generate_kernel__("sobel", size, 0)
        sobel_y = ImageOperations.__generate_kernel__("sobel", size, 1)

        # Get image dimensions
        rows, cols = img.shape

        # Initialize arrays for gradient magnitude and orientation
        Gx = np.zeros_like(img, dtype=np.float64)
        Gy = np.zeros_like(img, dtype=np.float64)

        # Compute gradient using Sobel operators
        half_size = size[0] // 2
        for i in range(half_size, rows - half_size):
            for j in range(half_size, cols - half_size):
                window = img[i - half_size:i + half_size + 1, j - half_size:j + half_size + 1]
                Gx[i, j] = np.sum(window * sobel_x)
                Gy[i, j] = np.sum(window * sobel_y)

        # Compute gradient magnitude and orientation
        magnitude = np.sqrt(Gx ** 2 + Gy ** 2)
        orientation = np.arctan2(Gy, Gx)

        return magnitude, orientation
    
    @staticmethod
    def __apply_non_max_suppression__(magnitude: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        """
        Apply non-maximum suppression to the gradient magnitude.\n
        \n
        This will thin the edges by keeping only the local maxima.\n
        \n
        Args:\n
        - magnitude (np.ndarray): Gradient magnitude.\n
        - orientation (np.ndarray): Gradient orientation.\n
        \n
        Returns:\n
        - suppressed_magnitude (np.ndarray): Suppressed gradient magnitude.\n
        \n
        Raises:\n
        - ValueError: If magnitude and orientation are not the same shape.\n
        """

        # Validate input shapes
        if magnitude.shape != orientation.shape:
            raise ValueError("Magnitude and orientation must be the same shape.")

        # Initialize suppressed magnitude array
        suppressed_magnitude = np.copy(magnitude)
        rows, cols = magnitude.shape

        # Apply non-maximum suppression
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                angle = orientation[i, j]
                q = [0, 0]

            # Determine neighboring pixels based on gradient orientation
                if (-np.pi/8 <= angle < np.pi/8) or (7*np.pi/8 <= angle):
                    q[0] = magnitude[i, j+1]
                    q[1] = magnitude[i, j-1]
                elif (np.pi/8 <= angle < 3*np.pi/8):
                    q[0] = magnitude[i+1, j+1]
                    q[1] = magnitude[i-1, j-1]
                elif (3*np.pi/8 <= angle < 5*np.pi/8):
                    q[0] = magnitude[i+1, j]
                    q[1] = magnitude[i-1, j]
                else:
                    q[0] = magnitude[i-1, j+1]
                    q[1] = magnitude[i+1, j-1]

                # Suppress non-maximum pixels
                if magnitude[i, j] < max(q[0], q[1]):
                    suppressed_magnitude[i, j] = 0
        
        return suppressed_magnitude

    @staticmethod
    def __apply_edge_tracking_by_hysteresis__(magnitude: np.ndarray, low_threshold: float, high_threshold: float) -> np.ndarray:
        """
        Apply edge tracking by hysteresis to detect strong and weak edges.\n
        \n
        Args:\n
        - magnitude (np.ndarray): Gradient magnitude.\n
        - low_threshold (float): Low threshold for weak edges.\n
        - high_threshold (float): High threshold for strong edges.\n
        \n
        Returns:\n
        - edge_map (np.ndarray): Edge map with strong edges (255) and weak edges connected to strong edges (255).\n
        """

        # Get image dimensions
        rows, cols = magnitude.shape

        # Initialize edge map
        edge_map = np.zeros((rows, cols), dtype=np.uint8)

        # Find strong and weak edges
        strong_edge_i, strong_edge_j = np.where(magnitude >= high_threshold)
        weak_edge_i, weak_edge_j = np.where((magnitude >= low_threshold) & (magnitude < high_threshold))

        # Mark strong edges as white (255)
        edge_map[strong_edge_i, strong_edge_j] = 255

        # Mark weak edges as white if they are connected to strong edges
        for i, j in zip(weak_edge_i, weak_edge_j):
            if (edge_map[max(0, i-1):min(rows, i+2), max(0, j-1):min(cols, j+2)] == 255).any():
                edge_map[i, j] = 255

        return edge_map

    @staticmethod
    def canny_edge_detection(
        image: np.ndarray, 
        low_threshold: float, 
        high_threshold: float, 
        gaussian_kernel_size: Tuple[int,int] = (5, 5),
        sobel_kernel_size: Tuple[int,int] = (3,3),
        sigma:float=1
    ) -> np.ndarray:
        """
        Apply Canny edge detection algorithm.\n
        \n
        Args:\n
        - image (np.ndarray): Input image.\n
        - low_threshold (float): Low threshold for weak edges.\n
        - high_threshold (float): High threshold for strong edges.\n
        - gaussian_kernel_size (Tuple[int,int], optional): Gaussian kernel \
          size for noise reduction. Defaults to (5,5).\n
        - sobel_kernel_size (Tuple[int,int], optional): Sobel kernel size for \
          gradient computation. Defaults to (3,3).\n
        - sigma (float): Standard deviation for Gaussian kernel.Defaults to 1.\n
        \n
        Returns:\n
        - edge_map (np.ndarray): Edge map with detected edges.\n
        """

        # Apply Gaussian filter for noise reduction
        print(f"Applying Gaussian filter with kernel size {gaussian_kernel_size}...")
        blurred_image = ImageOperations.gaussian_blur(image, gaussian_kernel_size,sigma)
        ImageOperations.save_image(blurred_image, "1-blurred.jfif")

        # Compute gradient magnitude and orientation
        print(f"Computing gradient magnitude and orientation with Sobel kernel size {sobel_kernel_size}...")
        gradient_magnitude, gradient_orientation = ImageOperations.__sobel_gradient_magnitude_and_orientation_approximation__(blurred_image, sobel_kernel_size)

        # Apply non-maximum suppression
        print("Applying non-maximum suppression...")
        non_max_suppressed = ImageOperations.__apply_non_max_suppression__(gradient_magnitude, gradient_orientation)
        ImageOperations.save_image(non_max_suppressed, "2-non_max_suppressed.tiff")

        # Apply edge tracking by hysteresis
        print(f"Applying edge tracking by hysteresis with low threshold {low_threshold}, high threshold {high_threshold}...")
        edge_map = ImageOperations.__apply_edge_tracking_by_hysteresis__(non_max_suppressed, low_threshold, high_threshold)
        ImageOperations.save_image(edge_map,"3-edge_map.jpeg")
        return edge_map

    @staticmethod
    def frequency_domain_filter(
        img:np.ndarray,
        filter:str,
        threshold:float,
        mode:str,
        n:int=1
    )->np.ndarray:
        '''
        Apply frequency domain fitlering methods.\n
        Args:\n
            img (float) : numpy 2D-array of image in spatial domain(grayscale only).\n
            filter (str) : 'gaussian' , 'butterworth' or 'ideal'.\n
            threshold (float) : \n
                i)  sigma(standard deviation) for gaussian.\n
                ii) r0 (cutoff frequency) for butterworth and ideal.\n
            mode (str) : 'high' or 'low' pass filters.\n
            n (int) : filter order , arg given only in case of butterworth filter.\n
        \n
        Returns: \n
            Image array with applied frequency domain filter.\n
        Raises:\n
            ValueError if any other filters or modes are given.\n
        '''
        # Convert image to frequency domain
        # Each pixel intensity gets replaced with F(u,v)
        F:np.ndarray[np.complex128]=np.fft.fft2(img)
        # Shift the origin of the image at center
        F=np.fft.fftshift(F)
        # we will calculate D(u,v) for each coordinate
        D:np.ndarray[np.floating]=np.zeros_like(img)
        # to reduce redudant comparisons of match case statements
        # inside nested for loops we require such long code
        # its still more efficient than comparing 
        # O(number_of_pixels*number_of_filters*number_of_modes)
        # times match case statements inside nested for loops
        # i.e. we reduced that many comparisons by writing below
        # redudant lines of code as directly nested for loops
        # start execution once mode and filter is finalized
        H:np.ndarray[np.complex128]=np.full_like(F,fill_value=0,dtype=np.complex128)
        match(filter):
            case 'gaussian':
                sigma:float=threshold
                match(mode):
                    case 'high':
                        for row in range(img.shape[0]):
                            for col in range(img.shape[1]):
                                u=np.real(F[row][col])
                                v=np.imag(F[row][col])
                                D[row][col]=np.sqrt(u**2+v**2)
                                H[row][col]=1-np.exp(
                                    (-D[row][col]**2)/
                                    (2*(sigma**2))
                                )
                    case 'low':
                        for row in range(img.shape[0]):
                            for col in range(img.shape[1]):
                                u=np.real(F[row][col])
                                v=np.imag(F[row][col])
                                D[row][col]=np.sqrt(u**2+v**2)
                                H[row][col]=np.exp(
                                    (-D[row][col]**2)/
                                    (2*(sigma**2))
                                )
                    case _:
                        raise ValueError('only two modes of filtering are available\
                                          in frequency domain,high pass filtering and\
                                          low pass filtering')
        
            case 'butterworth':
                r0:float=threshold
                match(mode):
                    case 'high':
                        for row in range(img.shape[0]):
                            for col in range(img.shape[1]):
                                u=np.real(F[row][col])
                                v=np.imag(F[row][col])
                                D[row][col]=np.sqrt(u**2+v**2)
                                H[row][col]=1/(1+(
                                    r0/(D[row][col]+0.00000000001)
                                )**(2*n))
                    case 'low':
                        for row in range(img.shape[0]):
                            for col in range(img.shape[1]):
                                u=np.real(F[row][col])
                                v=np.imag(F[row][col])
                                D[row][col]=np.sqrt(u**2+v**2)
                                H[row][col]=1/(1+(
                                    D[row][col]/r0
                                )**(2*n))
                    case _:
                        raise ValueError('only two modes of filtering are available\
                                          in frequency domain,high pass filtering and\
                                          low pass filtering')
            case 'ideal':
                r0:float=threshold
                match(mode):
                    case 'high':
                        for row in range(img.shape[0]):
                            for col in range(img.shape[1]):
                                u=np.real(F[row][col])
                                v=np.imag(F[row][col])
                                D[row][col]=np.sqrt(u**2+v**2)
                                if D[row][col]<=r0:
                                    H[row][col]=np._ComplexValue(real=0,imag=0)
                                else:
                                    H[row][col]=np._ComplexValue(real=1,imag=0)
                    case 'low':
                        for row in range(img.shape[0]):
                            for col in range(img.shape[1]):
                                u=np.real(F[row][col])
                                v=np.imag(F[row][col])
                                D[row][col]=np.sqrt(u**2+v**2)
                                if D[row][col]<=r0:
                                    H[row][col]=np._ComplexValue(real=1,imag=0)
                                else:
                                    H[row][col]=np._ComplexValue(real=0,imag=0)
                                    
                    case _:
                        raise ValueError('only two modes of filtering are available\
                                          in frequency domain,high pass filtering and\
                                          low pass filtering')    
            case _:
                raise ValueError("only three filters 'gaussian','butterworth',and \
                                 'ideal' are available in this version of library in \
                                 frequency domain filtering")
            
        # apply origin reshifting
        fitlered_image_freq=H*F
        # apply inverse Fourier transform
        IF=np.fft.ifftshift(fitlered_image_freq)
        IFrecenteredorigin=np.fft.ifft2(IF)
        return Image.fromarray(np.abs(IFrecenteredorigin))

# Driver code
def main():
    path_to_image = "fruits.jfif"
    img=Image.open(path_to_image)
    img=np.array(img)
    print('Converting input RGB image to grayscale')
    img=ImageOperations.convert_rgb_to_grayscale(img,method='lum')
    # Define the PARAMETERS here
    low_threshold = 30 # Low_threshold value for hysteresis
    high_threshold = 100 # High_threshold value for hysteresis

    print('Applying Canny Edge Detection on input grayscale image')
    # Apply Canny edge detection
    edge_image = ImageOperations.canny_edge_detection(img, low_threshold, high_threshold)
    print('Completed Canny Edge detection operation')
    # below  were some examples and notable seen results of frequency domain fitlering
    # See the image with below gaussian,threshold(sigma)=10,mode='high'
    # See the image with below butterworth,threshold(cutoff frequency)=80,mode='low'
    print('Applying Gaussian high pass filtering on input grayscale image')
    gaussian_hpf_applied_img=ImageOperations.frequency_domain_filter(img,filter='gaussian',threshold=110,mode='high')
    # resulting image needs to be saved in .tiff format due to floating point concerns
    gaussian_hpf_applied_img.save('gaussian_hpf_applied_img.tiff')
    print('Completed Gaussian high pass filtering operation')
    print('Check out results')

main()