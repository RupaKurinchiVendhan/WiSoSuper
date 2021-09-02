import numpy as np
from math import sqrt, floor, log10
from skimage.metrics import structural_similarity as SSIM
import lpips
import pylab
import matplotlib.image as mpimg
from PIL import Image
import scipy.stats as stats
from scipy.stats import entropy

# # # Linearly calibrated models (LPIPS)
loss_fn = lpips.LPIPS(net='vgg', spatial=False) # Can also set net = 'squeeze' or 'vgg'

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
 mse_error /= float(imageA.shape[0] * imageA.shape[1] * 255 )
 mse_error /= (np.mean((imageA.astype("float"))))**2
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
    return SSIM(imageA, imageB, multichannel=True)

def perceptual_dist(imageA_path, imageB_path, use_gpu=False, spatial=False):

    # # Linearly calibrated models (LPIPS)
    # loss_fn = lpips.LPIPS(net='vgg', spatial=False) 
    # # loss_fn = lpips.LPIPS(net='alex', spatial=spatial, lpips=False) # Can also set net = 'squeeze' or 'vgg'

    ## Example usage with images
    ex_ref = lpips.im2tensor(lpips.load_image(imageA_path))
    ex_p0 = lpips.im2tensor(lpips.load_image(imageB_path))
    if(use_gpu):
        ex_ref = ex_ref.cuda()
        ex_p0 = ex_p0.cuda()

    ex_d0 = loss_fn.forward(ex_ref,ex_p0)

    if not spatial:
        return ex_d0.mean().item()
        # print('Distances: (%.3f)'%(ex_d0, ex_d1))
    else:
        # Visualize a spatially-varying distance map between ex_p0 and ex_ref
        pylab.imshow(ex_d0[0,0,...].data.cpu().numpy())
        pylab.show()
        return ex_d0.mean()
        # print('Distances: (%.3f, %.3f)'%(ex_d0.mean(), ex_d1.mean()))            # The mean distance is approximately the same as the non-spatial distance

def split_and_average(img):
    img_red = img[:,:,2]
    img_green = img[:,:,1]
    img_blue = img[:,:,0]

    img_avg = np.zeros(img_red.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_avg[i][j] = (img_red[i][j]+img_green[i][j]+img_blue[i][j])/3
    return img_avg

def power_spectrum(img_path):
    img = Image.open(img_path).convert('L')
    img.save('greyscale.png')

    image = mpimg.imread("greyscale.png")
    npix = image.shape[0]
    # img = cv2.imread(img_path)

    # image = split_and_average(img)

    npix = image.shape[0]
    fourier_image = np.fft.fftn(image)
    fourier_amplitudes = np.abs(fourier_image)**2

    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)

    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    kbins = np.arange(0.5, npix//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                        statistic = "mean",
                                        bins = kbins)
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    return kvals, Abins

def rescale_linear(array, new_min, new_max):
    """Rescale an arrary linearly."""
    minimum, maximum = np.min(array), np.max(array)
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b

def energy_spectrum(img_path):
    img = Image.open(img_path).convert('L')
    img.save('greyscale.png')

    image = mpimg.imread("greyscale.png")

    
    # img = cv2.imread(img_path)

    # image = split_and_average(img)

    # npix = image.shape[0]
    # ampls = abs(np.fft.fftn(image))/npix
    # ek = ampls**2
    # ek = np.fft.fftshift(ek)
    # ek = ek.flatten()

    # kfreq = np.fft.fftfreq(npix) * npix
    # kfreq2D = np.meshgrid(kfreq, kfreq)
    # knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)

    # knrm = knrm.flatten()
    
    # kbins = np.arange(0.5, npix//2+1, 1.)
    # kvals = 0.5*(kbins[1:] + kbins[:-1])
    
    # ek, _, _ = stats.binned_statistic(knrm, ek,
    #                                     statistic = "mean",
    #                                     bins = kbins)
    
    # ek *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)

    npix = image.shape[0]
    fourier_image = np.fft.fftn(image)
    fourier_amplitudes = np.abs(fourier_image)**2
    fourier_amplitudes = np.fft.fftshift(fourier_amplitudes)

    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)

    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    kbins = np.arange(0.5, npix//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                        statistic = "mean",
                                        bins = kbins)
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)

    return kvals, Abins

def js(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
   # normalize
    p /= p.sum()
    q /= q.sum()
    m = (p + q) / 2
    return (entropy(p, m) + entropy(q, m)) / 2