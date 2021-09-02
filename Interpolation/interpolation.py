import cv2
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, floor, log10
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import pandas as pd
from comparison.metrics import *

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

def image_change_scale(img, dimension, scale=100, interpolation=cv2.INTER_LINEAR):
    '''Resize image to a specificall scale of original image.
    Parameters:
    img (numpy.ndarray): Original image
    dimension (tuple): Original image dimension
    scale (int): Multiply the size of the original image

    Returns:
    numpy.ndarray: Resized image
    '''
    scale /= 100
    new_dimension = (int(dimension[1]*scale), int(dimension[0]*scale))
    resized_img = cv2.resize(img, new_dimension, interpolation=interpolation)

    return resized_img


def W(x):
    '''Weight function that return weight for each distance point
    Parameters:
    x (float): Distance from destination point

    Returns:
    float: Weight
    '''
    a = -0.5
    pos_x = abs(x)
    if -1 <= abs(x) <= 1:
        return ((a+2)*(pos_x**3)) - ((a+3)*(pos_x**2)) + 1
    elif 1 < abs(x) < 2 or -2 < x < -1:
        return ((a * (pos_x**3)) - (5*a*(pos_x**2)) + (8 * a * pos_x) - 4*a)
    else:
        return 0
    
def bicubic_interpolation(img, dimension):
    '''Bicubic interpolation method to convert small size image to original size image
    Parameters:
    img (numpy.ndarray): Small image
    dimension (tuple): resizing image dimension

    Returns:
    numpy.ndarray: Resized image
    '''
    nrows = dimension[0]
    ncols = dimension[1]

    output = np.zeros((nrows, ncols, img.shape[2]), np.uint8)
    for c in range(img.shape[2]):
        for i in range(nrows):
            for j in range(ncols):
                xm = (i + 0.5) * (img.shape[0]/dimension[0]) - 0.5
                ym = (j + 0.5) * (img.shape[1]/dimension[1]) - 0.5

                xi = floor(xm)
                yi = floor(ym)

                u = xm - xi
                v = ym - yi

                out = 0
                for n in range(-1, 3):
                    for m in range(-1, 3):
                        if ((xi + n < 0) or (xi + n >= img.shape[1]) or (yi + m < 0) or (yi + m >= img.shape[0])):
                            continue

                        out += (img[xi+n, yi+m, c] * (W(u - n) * W(v - m)))

                output[i, j, c] = np.clip(out, 0, 255)

    return output