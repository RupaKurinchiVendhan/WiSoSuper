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
        

def bilinear_interpolation(image, dimension):
    '''Bilinear interpolation method to convert small image to original image
    Parameters:
    img (numpy.ndarray): Small image
    dimension (tuple): resizing image dimension

    Returns:
    numpy.ndarray: Resized image
    '''
    height = image.shape[0]
    width = image.shape[1]

    scale_x = (width)/(dimension[1])
    scale_y = (height)/(dimension[0])

    new_image = np.zeros((dimension[0], dimension[1], image.shape[2]))

    for k in range(3):
        for i in range(dimension[0]):
            for j in range(dimension[1]):
                x = (j+0.5) * (scale_x) - 0.5
                y = (i+0.5) * (scale_y) - 0.5

                x_int = int(x)
                y_int = int(y)

                # Prevent crossing
                x_int = min(x_int, width-2)
                y_int = min(y_int, height-2)

                x_diff = x - x_int
                y_diff = y - y_int

                a = image[y_int, x_int, k]
                b = image[y_int, x_int+1, k]
                c = image[y_int+1, x_int, k]
                d = image[y_int+1, x_int+1, k]

                pixel = a*(1-x_diff)*(1-y_diff) + b*(x_diff) * \
                    (1-y_diff) + c*(1-x_diff) * (y_diff) + d*x_diff*y_diff

                new_image[i, j, k] = pixel.astype(np.uint8)

    return new_image


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

def main(filename):
    images_list = {}

    # Read Image
    img, size, dimension = read_image("./input/"+filename)
    print(f"Image size is: {size}")
    images_list['Original Image'] = img

    # Change Image Size
    scale_percent = 20  # percent of original image size
    resized_img = image_change_scale(img, dimension, scale_percent)
    print(f"Smalled Image size is: {resized_img.shape}")
    images_list['Smalled Image'] = resized_img

    fig, axs = plt.subplots(1, 3)
    fig.suptitle('Interpolation Baseline', fontsize=16)

    # Change image to original size using bilinear interpolation
    bil_img = image_change_scale(
        resized_img, dimension, interpolation=cv2.INTER_LINEAR)
    images_list['Bilinear Interpolation'] = bil_img

    bil_img_algo = bilinear_interpolation(resized_img, dimension)
    bil_img_algo = Image.fromarray(bil_img_algo.astype('uint8')).convert('RGB')

    # Change image to original size using bicubic interpolation
    # cubic_img_algo = bicubic(resized_img, 4, -1/4)
    cubic_img_algo = bicubic_interpolation(resized_img, dimension)
    cubic_img_algo = Image.fromarray(
        cubic_img_algo.astype('uint8')).convert('RGB')

    axs[0].set_title("Original")
    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    axs[1].set_title("Bicubic")
    axs[1].imshow(cv2.cvtColor(np.array(cubic_img_algo), cv2.COLOR_BGR2RGB))

    axs[2].set_title("Bilinear")
    axs[2].imshow(cv2.cvtColor(np.array(bil_img_algo), cv2.COLOR_BGR2RGB))
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Bilinear Error Analysis
    print("------- Bilinear Metrics -------")
    gray_bil = cv2.cvtColor(np.array(bil_img_algo), cv2.COLOR_BGR2GRAY)
    # Calculate and print the PSNR value
    psnr_bil = psnr(img, np.array(bil_img_algo))
    print(f"PSNR: {psnr_bil}")
    # Calculate and print the SSIM value
    ssim_bil = ssim(gray_img, gray_bil)
    print(f"SSIM: {ssim_bil}")
    # Calculate and print the MSE value
    mse_bil = mse(gray_img, gray_bil)
    print(f"MSE: {mse_bil}")
    # Calculate and print the MAE value
    mae_bil = mae(img, np.array(bil_img_algo))
    print(f"MAE: {mae_bil}")

    # Bicubic Error Analysis
    print("------- Bicubic Metrics -------")
    gray_cub = cv2.cvtColor(np.array(cubic_img_algo), cv2.COLOR_BGR2GRAY)
    # Calculate and print the PSNR value
    psnr_cub = psnr(img, np.array(cubic_img_algo))
    print(f"PSNR: {psnr_cub}")
    # Calculate and print the SSIM value
    ssim_cub = ssim(gray_img, gray_cub)
    print(f"SSIM: {ssim_cub}")
    # Calculate and print the MSE value
    mse_cub = mse(gray_img, gray_cub)
    print(f"MSE: {mse_cub}")
    # Calculate and print the MAE value
    mae_cub = mae(img, np.array(cubic_img_algo))
    print(f"MAE: {mae_cub}")

    # save image
    cv2.imwrite('./output/bicubic/'+filename, np.array(cubic_img_algo))
    cv2.imwrite('./output/bilinear/'+filename, np.array(bil_img_algo))

    # plt.grid()
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return np.array(bil_img_algo), np.array(cubic_img_algo)