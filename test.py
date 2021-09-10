from PhIREGAN.PhIREGANs import *
from metrics import *
from utils import *
from Interpolation.interpolation import *
import cv2
import numpy as np
from PIL import Image

class Tester:
    DEFAULT_TIMESTEPS = []
    COMPONENTS = {'wind': {'ua':1, 'va':1}, 'solar': {'dni':0, 'dhi':1}}

    def __init__(self, timesteps=None):
        self.phiregan_metrics = {'PSNR': [], 'SSIM': [], 'MSE': [], 'MAE': []}
        self.cnn_metrics = {'PSNR': [], 'SSIM': [], 'MSE': [], 'MAE': []}
        self.bicubic_metrics = {'PSNR': [], 'SSIM': [], 'MSE': [], 'MAE': []}
        self.edsr_metrics = {'PSNR': [], 'SSIM': [], 'MSE': [], 'MAE': []}
        self.timesteps = timesteps if timesteps is not None else self.DEFAULT_TIMESTEPS

    def interpolate(self, HR_dir, cub):
        for gt_HR in os.listdir(HR_dir):
            img, _, dimension = read_image(gt_HR)

            # Change Image Size
            scale_percent = 20  # percent of original image size
            resized_img = image_change_scale(img, dimension, scale_percent)

            # Change image to original size using bicubic interpolation
            cubic_img_algo = bicubic_interpolation(resized_img, dimension)
            cubic_img_algo = Image.fromarray( 
                cubic_img_algo.astype('uint8')).convert('RGB')

            # Save output
            cv2.imwrite(cub+gt_HR, np.array(cubic_img_algo))

    def report_metrics(self):
        print("------- GANs Metrics -------")
        # Calculate and print the PSNR value
        gan_psnr_val = np.mean(self.gan_metrics['PSNR'])
        print(f"PSNR: {gan_psnr_val}")
        # Calculate and print the SSIM value
        gan_ssim_val = np.mean(self.gan_metrics['SSIM'])
        print(f"SSIM: {gan_ssim_val}")
        # Calculate and print the MSE value
        gan_mse_val = np.mean(self.gan_metrics['MSE'])
        print(f"MSE: {gan_mse_val}")
        # Calculate and print the MAE value
        gan_mae_val = np.mean(self.gan_metrics['MAE'])
        print(f"MAE: {gan_mae_val}")

        print("------- SR CNN Metrics -------")
        # Calculate and print the PSNR value
        cnn_psnr_val = np.mean(self.cnn_metrics['PSNR'])
        print(f"PSNR: {cnn_psnr_val}")
        # Calculate and print the SSIM value
        cnn_ssim_val = np.mean(self.cnn_metrics['SSIM'])
        print(f"SSIM: {cnn_ssim_val}")
        # Calculate and print the MSE value
        cnn_mse_val = np.mean(self.cnn_metrics['MSE'])
        print(f"MSE: {cnn_mse_val}")
        # Calculate and print the MAE value
        cnn_mae_val = np.mean(self.cnn_metrics['MAE'])
        print(f"MAE: {cnn_mae_val}")

        print("------- Bicubic Metrics -------")
        # Calculate and print the PSNR value
        cub_psnr_val = np.mean(self.bicubic_metrics['PSNR'])
        print(f"PSNR: {cub_psnr_val}")
        # Calculate and print the SSIM value
        cub_ssim_val = np.mean(self.bicubic_metrics['SSIM'])
        print(f"SSIM: {cub_ssim_val}")
        # Calculate and print the MSE value
        cub_mse_val = np.mean(self.bicubic_metrics['MSE'])
        print(f"MSE: {cub_mse_val}")
        # Calculate and print the MAE value
        cub_mae_val = np.mean(self.bicubic_metrics['MAE'])
        print(f"MAE: {cub_mae_val}")

        print("------- EDSR Metrics -------")
        # Calculate and print the PSNR value
        edsr_psnr_val = np.mean(self.edsr_metrics['PSNR'])
        print(f"PSNR: {edsr_psnr_val}")
        # Calculate and print the SSIM value
        edsr_ssim_val = np.mean(self.edsr_metrics['SSIM'])
        print(f"SSIM: {edsr_ssim_val}")
        # Calculate and print the MSE value
        edsr_mse_val = np.mean(self.edsr_metrics['MSE'])
        print(f"MSE: {edsr_mse_val}")
        # Calculate and print the MAE value
        edsr_mae_val = np.mean(self.edsr_metrics['MAE'])
        print(f"MAE: {edsr_mae_val}")

        print("------- ESRGAN Metrics -------")
        # Calculate and print the PSNR value
        esrgan_psnr_val = np.mean(self.esrgan_metrics['PSNR'])
        print(f"PSNR: {esrgan_psnr_val}")
        # Calculate and print the SSIM value
        esrgan_ssim_val = np.mean(self.esrgan_metrics['SSIM'])
        print(f"SSIM: {esrgan_ssim_val}")
        # Calculate and print the MSE value
        esrgan_mse_val = np.mean(self.esrgan_metrics['MSE'])
        print(f"MSE: {esrgan_mse_val}")
        # Calculate and print the MAE value
        esrgan_mae_val = np.mean(self.esrgan_metrics['MAE'])
        print(f"MAE: {esrgan_mae_val}")

    def compare_output_helper(self, data_type, component, timestep, i):
        gt_HR = "output/{data_type} test/{data_type} images/{data_type}/HR/{component}_{timestep}_{i}.png".format(data_type=data_type, component=component, timestep=timestep, i=i)
        phiregan = "output/{data_type} test/phiregan images/phiregan_{component}_{timestep}_{i}.png".format(data_type=data_type, component=component, timestep=timestep, i=i)
        cub = "output/{data_type} test/bicubic/{component}_{timestep}_{i}.png".format(data_type=data_type, component=component, timestep=timestep, i=i)
        edsr = "output/{data_type} test/edsr/sr_output/{component}_{timestep}_{i}.png".format(data_type=data_type, component=component, timestep=timestep, i=i)
        esrgan = "output/{data_type} test/esrgan/inference_result/{component}_{timestep}_{i}.png".format(data_type=data_type, component=component, timestep=timestep, i=i)
        cnn = "output/{data_type} test/cnn images/cnn_{component}_{timestep}_{i}.png".format(data_type=data_type, component=component, timestep=timestep, i=i)
        
        imgA, _, _ = read_image(gt_HR)
        imgB, _, _ = read_image(phiregan)
        imgC, _, _ = read_image(cnn)
        imgD, _, _ = read_image(cub)
        imgE, _, _ = read_image(edsr)
        imgF, _, _ = read_image(esrgan)

        phiregan_psnr_val = psnr(imgA, imgB)
        phiregan_ssim_val = ssim(imgA, imgB)
        phiregan_mse_val = mse(imgA, imgB)
        phiregan_mae_val = mae(imgA, imgB)
        
        self.phiregan_metrics['PSNR'].append(phiregan_psnr_val)
        self.phiregan_metrics['SSIM'].append(phiregan_ssim_val)
        self.phiregan_metrics['MSE'].append(phiregan_mse_val)
        self.phiregan_metrics['MAE'].append(phiregan_mae_val)

        cnn_psnr_val = psnr(imgA, imgC)
        cnn_ssim_val = ssim(imgA, imgC)
        cnn_mse_val = mse(imgA, imgC)
        cnn_mae_val = mae(imgA, imgC)
        
        self.cnn_metrics['PSNR'].append(cnn_psnr_val)
        self.cnn_metrics['SSIM'].append(cnn_ssim_val)
        self.cnn_metrics['MSE'].append(cnn_mse_val)
        self.cnn_metrics['MAE'].append(cnn_mae_val)

        cub_psnr_val = psnr(imgA, imgD)
        cub_ssim_val = ssim(imgA, imgD)
        cub_mse_val = mse(imgA, imgD)
        cub_mae_val = mae(imgA, imgD)
        
        self.bicubic_metrics['PSNR'].append(cub_psnr_val)
        self.bicubic_metrics['SSIM'].append(cub_ssim_val)
        self.bicubic_metrics['MSE'].append(cub_mse_val)
        self.bicubic_metrics['MAE'].append(cub_mae_val)

        edsr_psnr_val = psnr(imgA, imgE)
        edsr_ssim_val = ssim(imgA, imgE)
        edsr_mse_val = mse(imgA, imgE)
        edsr_mae_val = mae(imgA, imgE)
        
        self.edsr_metrics['PSNR'].append(edsr_psnr_val)
        self.edsr_metrics['SSIM'].append(edsr_ssim_val)
        self.edsr_metrics['MSE'].append(edsr_mse_val)
        self.edsr_metrics['MAE'].append(edsr_mae_val)

        esrgan_psnr_val = psnr(imgA, imgF)
        esrgan_ssim_val = ssim(imgA, imgF)
        esrgan_mse_val = mse(imgA, imgF)
        esrgan_mae_val = mae(imgA, imgF)
        
        self.esrgan_metrics['PSNR'].append(esrgan_psnr_val)
        self.esrgan_metrics['SSIM'].append(esrgan_ssim_val)
        self.esrgan_metrics['MSE'].append(esrgan_mse_val)
        self.esrgan_metrics['MAE'].append(esrgan_mae_val)

    def compare_output(self, data_type, component):
        for timestep in self.timesteps:
            for i in range(256):
                self.compare_output_helper(data_type, component, timestep, i)
        self.report_metrics()

if __name__ == '__main__':
    test_wind_timesteps = [3461]
    test = Tester(test_wind_timesteps)
    test.interpolate("output/{data_type} test/{data_type} images/{data_type}/HR/", "output/{data_type} test/bicubic/")
    test.compare_output(data_type='wind', component=None)