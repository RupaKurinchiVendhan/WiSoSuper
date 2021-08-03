import numpy as np
from math import sqrt, floor, log10
from skimage.metrics import structural_similarity as SSIM
from sklearn.metrics import mean_absolute_error as mae # not using this currently, but can replace my implementation if needed


def psnr(imageA, imageB):
    mse = np.mean((imageA - imageB) ** 2)
    # MSE is zero means no noise is present in the signal and PSNR has no importance.
    if(mse == 0):  
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

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

def ssim(imageA, imageB):
    SSIM(imageA, imageB, multichannel=True)