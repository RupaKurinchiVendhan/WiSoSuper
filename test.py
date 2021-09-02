from numpy.lib.function_base import interp
from PhIREGAN.PhIREGANs import *
from EDSR.model import get_generator
import EDSR.utils
import EDSR.test
from comparison.metrics import *
from utils import *
from Interpolation.interpolation import *
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.image as mpimg
import scipy.stats as stats

class Tester:
    DEFAULT_TIMESTEPS = []
    COMPONENTS = {'wind': {'ua':1, 'va':1}, 'solar': {'dni':0, 'dhi':1}}

    def __init__(self, timesteps=None):
        self.gan_metrics = {'PSNR': [], 'SSIM': [], 'MSE': [], 'MAE': []}
        self.bilinear_metrics = {'PSNR': [], 'SSIM': [], 'MSE': [], 'MAE': []}
        self.bicubic_metrics = {'PSNR': [], 'SSIM': [], 'MSE': [], 'MAE': []}
        self.edsr_metrics = {'PSNR': [], 'SSIM': [], 'MSE': [], 'MAE': []}
        self.timesteps = timesteps if timesteps is not None else self.DEFAULT_TIMESTEPS

    def interpolate(self, gt_HR, cub, bil):
        img, size, dimension = read_image(gt_HR)

        # Change Image Size
        scale_percent = 20  # percent of original image size
        resized_img = image_change_scale(img, dimension, scale_percent)

        # Change image to original size using bilinear interpolation
        bil_img = image_change_scale(
            resized_img, dimension, interpolation=cv2.INTER_LINEAR)
        bil_img_algo = bilinear_interpolation(resized_img, dimension)
        bil_img_algo = Image.fromarray(bil_img_algo.astype('uint8')).convert('RGB')

        # Change image to original size using bicubic interpolation
        cubic_img_algo = bicubic_interpolation(resized_img, dimension)
        cubic_img_algo = Image.fromarray( 
            cubic_img_algo.astype('uint8')).convert('RGB')

        # Save output
        cv2.imwrite(cub, np.array(cubic_img_algo))
        cv2.imwrite(bil, np.array(bil_img_algo))

        return np.array(bil_img_algo), np.array(cubic_img_algo)
                    
    def compare_output_helper(self, data_type, component, timestep, i, plot=False):
        gt_HR = "PhIREGAN/{data_type} test/{data_type} images/{data_type}/HR/{component}_{timestep}_{i}.png".format(data_type=data_type, component=component, timestep=timestep, i=i)
        gt_LR = "PhIREGAN/{data_type} test/{data_type} images/{data_type}/LR/{component}_{timestep}_{i}.png".format(data_type=data_type, component=component, timestep=timestep, i=i)
        gans = "PhIREGAN/{data_type} test/gans images/gans_{component}_{timestep}_{i}.png".format(data_type=data_type, component=component, timestep=timestep, i=i)
        cub = "PhIREGAN/{data_type} test/bicubic/bicubic_{component}_{timestep}_{i}.png".format(data_type=data_type, component=component, timestep=timestep, i=i)
        bil = "PhIREGAN/{data_type} test/bilinear/bilinear{component}_{timestep}_{i}.png".format(data_type=data_type, component=component, timestep=timestep, i=i)
        # edsr = "EDSR.png"
        # print(gt_HR)
        # print("PhIREGAN\wind test\wind images\wind\HR\ua_3461_0.png")
        imgA, _, _ = read_image(gt_HR)
        imgB, _, _ = read_image(gans)
        imgC, _, _ = read_image(bil)
        imgD, _, _ = read_image(cub)
        # imgC, imgD = self.interpolate(gt_HR, cub, bil)
        # imgE = read_image(edsr)

        gan_psnr_val = psnr(imgA, imgB)
        gan_ssim_val = ssim(imgA, imgB)
        gan_mse_val = mse(imgA, imgB)
        gan_mae_val = mae(imgA, imgB)
        
        self.gan_metrics['PSNR'].append(gan_psnr_val)
        self.gan_metrics['SSIM'].append(gan_ssim_val)
        self.gan_metrics['MSE'].append(gan_mse_val)
        self.gan_metrics['MAE'].append(gan_mae_val)

        bil_psnr_val = psnr(imgA, imgC)
        bil_ssim_val = ssim(imgA, imgC)
        bil_mse_val = mse(imgA, imgC)
        bil_mae_val = mae(imgA, imgC)
        
        self.bilinear_metrics['PSNR'].append(bil_psnr_val)
        self.bilinear_metrics['SSIM'].append(bil_ssim_val)
        self.bilinear_metrics['MSE'].append(bil_mse_val)
        self.bilinear_metrics['MAE'].append(bil_mae_val)

        cub_psnr_val = psnr(imgA, imgD)
        cub_ssim_val = ssim(imgA, imgD)
        cub_mse_val = mse(imgA, imgD)
        cub_mae_val = mae(imgA, imgD)
        
        self.bicubic_metrics['PSNR'].append(cub_psnr_val)
        self.bicubic_metrics['SSIM'].append(cub_ssim_val)
        self.bicubic_metrics['MSE'].append(cub_mse_val)
        self.bicubic_metrics['MAE'].append(cub_mae_val)

        # edsr_psnr_val = psnr(imgA, imgE)
        # edsr_ssim_val = ssim(imgA, imgE)
        # edsr_mse_val = mse(imgA, imgE)
        # edsr_mae_val = mae(imgA, imgE)
        
        # self.edsr_metrics['PSNR'].append(edsr_psnr_val)
        # self.edsr_metrics['SSIM'].append(edsr_ssim_val)
        # self.edsr_metrics['MSE'].append(edsr_mse_val)
        # self.edsr_metrics['MAE'].append(edsr_mae_val)

        if plot == True:
            fig, axs = plt.subplots(1, 5)

            axs[0].set_title("Ground Truth")
            axs[0].imshow(cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB))

            axs[1].set_title("GANs")
            axs[1].imshow(cv2.cvtColor(np.array(imgB), cv2.COLOR_BGR2RGB))

            axs[2].set_title("Bicubic")
            axs[2].imshow(cv2.cvtColor(np.array(imgD), cv2.COLOR_BGR2RGB))

            axs[3].set_title("Bilinear")
            axs[3].imshow(cv2.cvtColor(np.array(imgC), cv2.COLOR_BGR2RGB))

            # axs[4].set_title("EDSR")
            # axs[4].imshow(cv2.cvtColor(np.array(imgE), cv2.COLOR_BGR2RGB))

            plt.show()

            imgs = [gans, bil, cub, gt_LR, gt_HR] # ADD EDSR
            for i in imgs:
                img = Image.open(i).convert('L')
                img.save('greyscale.png')

                image = mpimg.imread("greyscale.png")
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

                plt.loglog(kvals, Abins, label=i[0:-4])
                plt.xlabel("$k$")
                plt.ylabel("$P(k)$")
                plt.tight_layout()
                plt.legend()
            plt.title("Radially-Averaged Power Spectrum")
            plt.show()

    def compare_output(self, data_type, component):
        for timestep in self.timesteps:
            for i in range(256):
                self.compare_output_helper(data_type, component, timestep, i)
            
if __name__ == '__main__':
    test_wind_timesteps = [3461]
    data_type = 'wind'
    component = 'ua'
    test = Tester(test_wind_timesteps)
    # test.gan()
    # test.save_output()
    test.compare_output(data_type=data_type, component=component)

    print("------- GANs Metrics -------")
    # Calculate and print the PSNR value
    gan_psnr_val = np.mean(test.gan_metrics['PSNR'])
    print(f"PSNR: {gan_psnr_val}")
    # Calculate and print the SSIM value
    gan_ssim_val = np.mean(test.gan_metrics['SSIM'])
    print(f"SSIM: {gan_ssim_val}")
    # Calculate and print the MSE value
    gan_mse_val = np.mean(test.gan_metrics['MSE'])
    print(f"MSE: {gan_mse_val}")
    # Calculate and print the MAE value
    gan_mae_val = np.mean(test.gan_metrics['MAE'])
    print(f"MAE: {gan_mae_val}")

    print("------- Bilinear Metrics -------")
    # Calculate and print the PSNR value
    bil_psnr_val = np.mean(test.bilinear_metrics['PSNR'])
    print(f"PSNR: {bil_psnr_val}")
    # Calculate and print the SSIM value
    bil_ssim_val = np.mean(test.bilinear_metrics['SSIM'])
    print(f"SSIM: {bil_ssim_val}")
    # Calculate and print the MSE value
    bil_mse_val = np.mean(test.bilinear_metrics['MSE'])
    print(f"MSE: {bil_mse_val}")
    # Calculate and print the MAE value
    bil_mae_val = np.mean(test.bilinear_metrics['MAE'])
    print(f"MAE: {bil_mae_val}")

    print("------- Bicubic Metrics -------")
    # Calculate and print the PSNR value
    cub_psnr_val = np.mean(test.bicubic_metrics['PSNR'])
    print(f"PSNR: {cub_psnr_val}")
    # Calculate and print the SSIM value
    cub_ssim_val = np.mean(test.bicubic_metrics['SSIM'])
    print(f"SSIM: {cub_ssim_val}")
    # Calculate and print the MSE value
    cub_mse_val = np.mean(test.bicubic_metrics['MSE'])
    print(f"MSE: {cub_mse_val}")
    # Calculate and print the MAE value
    cub_mae_val = np.mean(test.bicubic_metrics['MAE'])
    print(f"MAE: {cub_mae_val}")

    # print("------- EDSR Metrics -------")
    # # Calculate and print the PSNR value
    # edsr_psnr_val = np.mean(test.edsr_metrics['PSNR'])
    # print(f"PSNR: {edsr_psnr_val}")
    # # Calculate and print the SSIM value
    # edsr_ssim_val = np.mean(test.edsr_metrics['SSIM'])
    # print(f"SSIM: {edsr_ssim_val}")
    # # Calculate and print the MSE value
    # edsr_mse_val = np.mean(test.edsr_metrics['MSE'])
    # print(f"MSE: {edsr_mse_val}")
    # # Calculate and print the MAE value
    # edsr_mae_val = np.mean(test.edsr_metrics['MAE'])
    # print(f"MAE: {edsr_mae_val}")

    # Save output for further analysis
    gans_df = pd.DataFrame(test.gan_metrics) 
    gans_df.to_csv('gan'+data_type+'.csv')

    bilinear_df = pd.DataFrame(test.bilinear_metrics)
    bilinear_df.to_csv('bilinear'+data_type+'.csv')

    bicubic_df = pd.DataFrame(test.bicubic_metrics)
    bicubic_df.to_csv('bicubic'+data_type+'.csv')

    # edsr_df = pd.DataFrame(test.edsr_metrics)
    # edsr_df.to_csv('edsr'+data_type+'.csv')