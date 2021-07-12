import cv2
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, log10
from PIL import Image


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


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def main():
    images_list = {}

    # Read Image
    # img, size, dimension = read_image("./butterfly.png")
    img, size, dimension = read_image("./test.png")
    print(f"Image size is: {size}")
    images_list['Original Image'] = img

    # Change Image Size
    scale_percent = 25  # percent of original image size
    resized_img = image_change_scale(img, dimension, scale_percent)
    print(f"Smalled Image size is: {resized_img.shape}")
    images_list['Smalled Image'] = resized_img

    fig, axs = plt.subplots(1, 2)
    fig.suptitle('Interpolation Baseline', fontsize=16)

    # Change image to original size using bilinear interpolation
    bil_img = image_change_scale(
        resized_img, dimension, interpolation=cv2.INTER_LINEAR)
    images_list['Bilinear Interpolation'] = bil_img

    bil_img_algo = bilinear_interpolation(resized_img, dimension)
    bil_img_algo = Image.fromarray(bil_img_algo.astype('uint8')).convert('RGB')

    axs[0].set_title("Original")
    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    axs[1].set_title("Bilinear")
    axs[1].imshow(cv2.cvtColor(np.array(bil_img_algo), cv2.COLOR_BGR2RGB))

    # Calculate and print the PSNR value
    psnr = PSNR(img, np.array(bil_img_algo))
    print(f"PSNR value is {psnr} dB")

    # plt.grid()
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()