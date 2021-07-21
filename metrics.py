import cv2
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, floor, log10
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_absolute_error as mae
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

def psnr(imageA, imageB):
    mse = np.mean((imageA - imageB) ** 2)
    # MSE is zero means no noise is present in the signal and PSNR has no importance.
    if(mse == 0):  
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def psnr2(imageA, imageB):
    return -10. * np.log10(np.mean(np.square(imageA - imageB)))


def mse(imageA, imageB):
 # the 'Mean Squared Error' between the two images is the sum of the squared difference between the two images
 mse_error = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
 mse_error /= float(imageA.shape[0] * imageA.shape[1])
 # return the MSE. The lower the error, the more "similar" the two images are.
 return mse_error


def mae(imageA, imageB):
    mae = np.sum(np.absolute((imageB.astype("float") - imageA.astype("float"))))
    mae /= float(imageA.shape[0] * imageA.shape[1] * 255)
    if (mae < 0):
        return mae * -1
    else:
        return mae

def main(filenameA, gan, bil, cub):
    imgA, sizeA, _ = read_image("./input/"+filenameA)
    print(f"Image Size is: {sizeA}")
    imgB, _, _ = read_image("./input/"+gan)
    imgC, _, _ = read_image("./input/"+bil)
    imgD, _, _ = read_image("./input/"+cub)

    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
    grayC = cv2.cvtColor(imgC, cv2.COLOR_BGR2GRAY)
    grayD = cv2.cvtColor(imgD, cv2.COLOR_BGR2GRAY)

    print("------- GANs Metrics -------")
    # Calculate and print the PSNR value
    psnr_val = psnr(imgA, imgB)
    print(f"PSNR: {psnr_val}")
    # Calculate and print the SSIM value
    ssim_val = ssim(grayA, grayB)
    print(f"SSIM: {ssim_val}")
    # Calculate and print the MSE value
    mse_val = mse(grayA, grayB)
    print(f"MSE: {mse_val}")
    # Calculate and print the MAE value
    mae_val = mae(imgA, imgB)
    print(f"MAE: {mae_val}")

    print("------- Bilinear Metrics -------")
    # Calculate and print the PSNR value
    psnr_val = psnr(imgA, imgC)
    print(f"PSNR: {psnr_val}")
    # Calculate and print the SSIM value
    ssim_val = ssim(grayA, grayC)
    print(f"SSIM: {ssim_val}")
    # Calculate and print the MSE value
    mse_val = mse(grayA, grayC)
    print(f"MSE: {mse_val}")
    # Calculate and print the MAE value
    mae_val = mae(imgA, imgC)
    print(f"MAE: {mae_val}")

    print("------- Bicubic Metrics -------")
    # Calculate and print the PSNR value
    psnr_val = psnr(imgA, imgD)
    print(f"PSNR: {psnr_val}")
    # Calculate and print the SSIM value
    ssim_val = ssim(grayA, grayD)
    print(f"SSIM: {ssim_val}")
    # Calculate and print the MSE value
    mse_val = mse(grayA, grayD)
    print(f"MSE: {mse_val}")
    # Calculate and print the MAE value
    mae_val = mae(imgA, imgD)
    print(f"MAE: {mae_val}")

    fig, axs = plt.subplots(1, 4)
    fig.suptitle('Comparison', fontsize=16)

    axs[0].set_title("Ground Truth")
    axs[0].imshow(cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB))

    axs[1].set_title("GANs")
    axs[1].imshow(cv2.cvtColor(np.array(imgB), cv2.COLOR_BGR2RGB))

    axs[2].set_title("Bicubic")
    axs[2].imshow(cv2.cvtColor(np.array(imgD), cv2.COLOR_BGR2RGB))

    axs[3].set_title("Bilinear")
    axs[3].imshow(cv2.cvtColor(np.array(imgC), cv2.COLOR_BGR2RGB))

    plt.show()

if __name__ == "__main__":
    main("paper/wind paper real.png", "paper/wind paper gan.png", "paper/wind paper real bil.png", "paper/wind paper real cub.png")