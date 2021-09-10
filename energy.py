'''
@author: Rupa Kurinchi-Vendhan
The following code offers a method for generating a kinetic energy spectrum, in a manner similar to generating a power spectrum.
For an official implementation of how to create a plot using turbulent flow statistics as in the paper, refer to this repository:
https://github.com/b-fg/Energy_spectra/blob/master/ek.py.
'''

import matplotlib.image as mpimg
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from PIL import Image
from PhIREGAN.PhIREGANs import *
from utils import *
import scipy.stats as stats
from Interpolation.interpolation import *
import os

Energy_Spectrum = {'Ground Truth':  {'x':[], 'y':[]}, 'LR Input': {'x':[], 'y':[]}, 'PhIREGAN': {'x':[], 'y':[]}, 'EDSR': {'x':[], 'y':[]}, 'ESRGAN': {'x':[], 'y':[]}, 'SR CNN': {'x':[], 'y':[]}, 'Bicubic': {'x':[], 'y':[]}}
COMPONENTS = {'wind': {'ua':1, 'va':1}, 'solar': {'dni':0, 'dhi':1}}

plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', size=16) # controls default text size
plt.rc('axes', titlesize=14) # fontsize of the title
plt.rc('axes', labelsize=14) # fontsize of the x and y labels
plt.rc('xtick', labelsize=14) # fontsize of the x tick labels
plt.rc('ytick', labelsize=14) # fontsize of the y tick labels
plt.rc('legend', fontsize=14) # fontsize of the legend


def energy_spectrum(img_path, min, max):
    img = Image.open(img_path).convert('L')
    img.save('greyscale.png')
    image = mpimg.imread("greyscale.png")
    image = rescale_linear(image, min, max)

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
    kvals = (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                        statistic = "mean",
                                        bins = kbins)
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)

    return kvals, Abins

def compare_output_helper(data_type, component, timestep, i):
    gt_HR = "PhIREGAN/{data_type} test/{data_type} images/{data_type}/HR/{component}_{timestep}_{i}.png".format(data_type=data_type, component=component, timestep=timestep, i=i)
    gt_HR_arr = "PhIREGAN/{data_type} test/{data_type} arrays/{data_type}/HR/{component}_{timestep}_{i}.png".format(data_type=data_type, component=component, timestep=timestep, i=i)
    gt_LR = "PhIREGAN/{data_type} test/LR/LR/{component}_{timestep}_{i}.png".format(data_type=data_type, component=component, timestep=timestep, i=i)
    phiregan = "PhIREGAN/{data_type} test/gans images/gans_{component}_{timestep}_{i}.png".format(data_type=data_type, component=component, timestep=timestep, i=i)
    cub = "PhIREGAN/{data_type} test/bicubic/bicubic_{component}_{timestep}_{i}.png".format(data_type=data_type, component=component, timestep=timestep, i=i)
    edsr = "PhIREGAN/{data_type} test/edsr/sr_output/{component}_{timestep}_{i}.png".format(data_type=data_type, component=component, timestep=timestep, i=i)
    cnn = "PhIREGAN/{data_type} test/cnns images/cnns_{component}_{timestep}_{i}.png".format(data_type=data_type, component=component, timestep=timestep, i=i)
    esrgan = "PhIREGAN/{data_type} test/esrgan/inference_result/{component}_{timestep}_{i}.png".format(data_type=data_type, component=component, timestep=timestep, i=i)

    min, max = np.min(np.load(gt_HR_arr)), np.max(np.load(gt_HR_arr))

    if os.path.isfile(cub) and os.path.isfile(edsr) and os.path.isfile(phiregan):        
        HR_kvals2, HR_ek = energy_spectrum(gt_HR, min, max)
        Energy_Spectrum['Ground Truth']['x'].append(HR_kvals2)
        Energy_Spectrum['Ground Truth']['y'].append(HR_ek)

        LR_kvals2, LR_ek = energy_spectrum(gt_LR, min, max)
        Energy_Spectrum['LR Input']['x'].append(LR_kvals2)
        Energy_Spectrum['LR Input']['y'].append(LR_ek)

        gan_kvals2, gan_EK = energy_spectrum(phiregan, min, max)

        Energy_Spectrum['PhIREGAN']['x'].append(gan_kvals2)
        Energy_Spectrum['PhIREGAN']['y'].append(gan_EK)

        cnn_kvals2, cnn_EK = energy_spectrum(cnn, min, max)

        Energy_Spectrum['SR CNN']['x'].append(cnn_kvals2)
        Energy_Spectrum['SR CNN']['y'].append(cnn_EK)

        cub_kvals2, cub_EK = energy_spectrum(cub, min, max)

        Energy_Spectrum['Bicubic']['x'].append(cub_kvals2)
        Energy_Spectrum['Bicubic']['y'].append(cub_EK)

        edsr_kvals2, edsr_EK = energy_spectrum(edsr, min, max)

        Energy_Spectrum['EDSR']['x'].append(edsr_kvals2)
        Energy_Spectrum['EDSR']['y'].append(edsr_EK)

        esrgan_kvals2, esrgan_EK = energy_spectrum(esrgan, min, max)

        Energy_Spectrum['ESRGAN']['x'].append(esrgan_kvals2)
        Energy_Spectrum['ESRGAN']['y'].append(esrgan_EK)

def plot_energy_spectra():
    colors = {'Ground Truth': 'black', 'LR Input': 'pink', 'PhIREGAN': 'tab:blue', 'EDSR': 'tab:orange', 'ESRGAN': 'tab:green', 'SR CNN': 'tab:red', 'Bicubic': 'tab:purple'}
    for model in Energy_Spectrum:
        k = np.flip(np.mean(Energy_Spectrum[model]['x'], axis=0))
        E = np.mean(Energy_Spectrum[model]['y'], axis=0) / 10000
        plt.loglog(k, E, color=colors[model], label=model)
    plt.xlabel("k (wavenumber)")
    plt.ylabel("Kinetic Energy")
    plt.tight_layout()
    plt.title("Energy Spectrum")
    plt.legend()
    plt.savefig("wind_spectrum.png", dpi=1000, transparent=True, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    test_wind_timesteps = [2889]
    data_type = 'wind'
    component = None
    for comp in COMPONENTS[data_type]:
            for timestep in test_wind_timesteps:
                for i in range(256):
                    compare_output_helper(data_type, comp, timestep, i)
    plot_energy_spectra()