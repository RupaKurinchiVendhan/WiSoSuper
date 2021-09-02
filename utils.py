from PIL import Image
import cv2
import numpy as np
import matplotlib.image as mpimg

def read_image(path):
    '''Read image and return the image propertis.
    Parameters:
    path (string): Image path

    Returns:
    numpy.ndarray: Image exists in "path"
    list: Image size
    tuple: Image dimension (number of rows and columns)
    '''
    img = cv2.imread(path)
    size = img.shape
    dimension = (size[0], size[1])
    return img, size, dimension

def rotate(image_path, degrees_to_rotate):
    """
    Rotate the given photo the amount of given degreesk, show it and save it
    @param image_path: The path to the image to edit
    @param degrees_to_rotate: The number of degrees to rotate the image
    """
    image_obj = Image.open(image_path)
    rotated_image = image_obj.rotate(degrees_to_rotate)
    rotated_image.save(image_path)

def flip_image(image_path):
    """
    Flip or mirror the image
    @param image_path: The path to the image to edit
    """
    image_obj = Image.open(image_path)
    rotated_image = image_obj.transpose(Image.FLIP_LEFT_RIGHT)
    rotated_image.save(image_path)

def normalize_image(image):
    """
    Normalize image arary to contain values between -1 adn +1.
    Parameters:
        image: numpy array representing image
    """
    return (image - np.amin(image)) / (np.amax(image) - np.amin(image))

def normalize_2d(matrix):
    norm = np.linalg.norm(matrix)
    matrix = matrix/norm  # normalized matrix
    return matrix

def read(img_path):
    img = Image.open(img_path).convert('L')
    img.save('greyscale.png')
    image1 = mpimg.imread("greyscale.png")
    image1 = normalize_2d(image1)
    return image1

def rescale_linear(array, new_min, new_max):
    """Rescale an arrary linearly."""
    minimum, maximum = np.min(array), np.max(array)
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b