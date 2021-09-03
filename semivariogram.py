import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy.lib.function_base import interp
from skgstat.models import variogram
from PhIREGAN.PhIREGANs import *
from utils import *
from Interpolation.interpolation import *
import os
import skgstat as skg

plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', size=16) # controls default text size
plt.rc('axes', titlesize=14) # fontsize of the title
plt.rc('axes', labelsize=14) # fontsize of the x and y labels
plt.rc('xtick', labelsize=14) # fontsize of the x tick labels
plt.rc('ytick', labelsize=14) # fontsize of the y tick labels
plt.rc('legend', fontsize=14) # fontsize of the legend


VARIOGRAM = {'Ground Truth':  {'x':[], 'y':[]}, 'LR Input':  {'x':[], 'y':[]}, 'PhIREGAN': {'x':[], 'y':[]}, 'EDSR': {'x':[], 'y':[]}, 'ESRGAN': {'x':[], 'y':[]}, 'SR CNN': {'x':[], 'y':[]}, 'Bicubic': {'x':[], 'y':[]}}


def compare_output_helper(data_type, component, timestep, i, plot=False):
    gt_HR = "output/{data_type} test/{data_type} images/{data_type}/HR/{component}_{timestep}_{i}.png".format(data_type=data_type, component=component, timestep=timestep, i=i)
    phiregan = "output/{data_type} test/phiregan images/phiregan_{component}_{timestep}_{i}.png".format(data_type=data_type, component=component, timestep=timestep, i=i)
    cub = "output/{data_type} test/bicubic5/bicubic_{component}_{timestep}_{i}.png".format(data_type=data_type, component=component, timestep=timestep, i=i)
    edsr = "output/{data_type} test/sr_output_solar/{component}_{timestep}_{i}.png".format(data_type=data_type, component=component, timestep=timestep, i=i)
    cnn = "output/{data_type} test/cnn images/cnn_{component}_{timestep}_{i}.png".format(data_type=data_type, component=component, timestep=timestep, i=i)
    esrgan = "output/{data_type} test/esrgan/inference_result/{component}_{timestep}_{i}.png".format(data_type=data_type, component=component, timestep=timestep, i=i)

    if os.path.isfile(cub) and os.path.isfile(edsr) and os.path.isfile(phiregan):
        imgs = {'Ground Truth': gt_HR, 'PhIREGAN': phiregan, 'EDSR': edsr, 'ESRGAN': esrgan, 'SR CNN': cnn, 'Bicubic': edsr}
        for i in imgs:
            img = Image.open(imgs[i]).convert('L')
            img.save('grey.png')
            image = mpimg.imread("grey.png")
            coords = np.arange(4, 500, 0.0496)[:10000]
            image = np.ravel(image)
            V = skg.Variogram(coords, image, normalize=True)
            _bins = V.bins
            _exp = V.experimental
            x = np.linspace(4, np.nanmax(_bins), 100)
        
            # apply the model
            y = V.transform(x)
        
            # handle the relative experimental variogram
            if V.normalized:
                _bins /= np.nanmax(_bins)
                y /= np.max(_exp)
                _exp /= np.nanmax(_exp)
            VARIOGRAM[i]['x'].append(x)
            VARIOGRAM[i]['y'].append(y)

def plot_variogram(component):
    colors = {'Ground Truth': 'black', 'PhIREGAN': 'tab:blue', 'EDSR': 'tab:orange', 'ESRGAN': 'tab:green', 'SR CNN': 'tab:red', 'Bicubic': 'tab:purple'}
    for model in VARIOGRAM:
        x = np.mean(VARIOGRAM[model]['x'], axis=0)
        y = np.mean(VARIOGRAM[model]['y'], axis=0)
        plt.plot(x, y, color=colors[model], label=model)
    plt.yscale('log')
    plt.xlabel("Lag Distance (km)")
    plt.ylabel("Normalized Variance")
    plt.title(component.upper()+" Semivariogram")
    plt.legend()
    plt.savefig(component+"_semivariogram.png", dpi=1000, transparent=True, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    component='dhi'
    test_solar_timesteps = [2322]
    for timestep in test_solar_timesteps:
            for i in range(256):
                compare_output_helper(data_type='solar', component=component, timestep=timestep, i=i)
    plot_variogram()