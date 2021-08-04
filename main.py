from numpy.lib.function_base import interp
from PhIREGAN.PhIREGANs import *
import EDSR.data
from EDSR.model import get_generator
import EDSR.utils
from EDSR.test import *
from EDSR.pretrain import *
from comparison.metrics import *
from comparison.util import *
from Interpolation.interpolation import *
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import pandas as pd
import matplotlib.image as mpimg
import scipy.stats as stats

def phiregan(data_type, data_dir, mode):
    components = {'wind': {'ua':1, 'va':1}, 'solar': {'dni':0, 'dhi':1}}
    for filename in os.listdir(data_dir):
        print("Loading data")
        timestep = filename[filename.index('_')+1:]
        timestep = timestep[:timestep.index('_')]
        if 'LR' in filename:
            LR_data_path = data_dir+'/'+filename
            HR_data_path = LR_data_path.replace('LR', 'HR')
        if 'HR' in filename:
            HR_data_path = data_dir+'/'+filename
            LR_data_path = HR_data_path.replace('HR', 'LR')
        print("Loading model")
        model_path = 'PhIREGAN/models/{data_type}_mr-hr/trained_gan/gan'.format(data_type=data_type)
        r = [5]
        if data_type=='wind':
            mu_sig=[[0.7684, -0.4575], [5.02455, 5.9017]]
        elif data_type=='solar':
            mu_sig = [[344.3262, 113.7444], [386.9283, 117.9627]]

        phiregans = PhIREGANs(component=component, mu_sig=mu_sig)
        if mode == 'train':
            model_dir = phiregans.pretrain(r=r,
                                        LR_data_path=LR_data_path,
                                        HR_data_path=HR_data_path,
                                        model_path=model_path,
                                        batch_size=1)

            model_dir = phiregans.train(r=r,
                                        LR_data_path=LR_data_path,
                                        HR_data_path=HR_data_path,
                                        model_path=model_dir,
                                        batch_size=1)
        sr_data = phiregans.test(r=r,
                    data_path=LR_data_path,
                    model_path=model_dir,
                    batch_size=1, timestep=timestep)          
        for i in range (256):
            for component in components[data_type]:
                f = "PhIREGAN/{data_type} test/gans images/gans_{component}_{timestep}_{i}.png".format(data_type=data_type, component=component, timestep=timestep, i=i)
                data = sr_data[i, :, :, components[data_type][component]]
                vmin, vmax = np.min(sr_data[:,:,:,components[data_type][component]]), np.max(sr_data[:,:,:,components[data_type][component]])   
                plt.imsave(f, data, vmin=vmin, vmax=vmax, origin='lower', format="png")
                rotate(f, 180)
                flip_image(f)

def edsr(model_path, data_dir, valid_dir, save_dir, ext, mode, resume=None, cuda=None):
    if mode == 'train':
        pretrain('edsr', data_dir, valid_dir, ext, ext, resume, cuda)
    global sess
    sess = tf.Session()
    model = get_generator('edsr', is_train=False)
    print("Loading model")
    model.load_weights(model_path)
    sr_from_folder(model, valid_dir+'/LR', save_dir, ext)

def esrgan(model_path, data_dir, valid_dir, save_dir, ext, mode, resume=None, cuda=None):
    if mode == 'train':
        pretrain('esrgan', data_dir, valid_dir, ext, ext, resume, cuda)
    global sess
    sess = tf.Session()
    model = get_generator('edsr', is_train=False)
    print("Loading model")
    model.load_weights(model_path)
    sr_from_folder(model, valid_dir+'/LR', save_dir, ext)

def main():
    parser = argparse.ArgumentParser(description='Generate SR images')
    parser.add_argument('--model', required=True, type=str, help='Model architecture')
    parser.add_argument('--model_path', required=True, type=str, help='Path to a model')
    parser.add_argument('--mode', required=True, type=str, help='Train or test')    
    parser.add_argument('--data_type', required=True, type=str, help='Wind or solar')
    parser.add_argument('--data_dir', type=str, default=None, help='Path to training images/tfrecords')
    parser.add_argument('--valid_dir', type=str, required=True, help='Path to test images/tfrecords')
    parser.add_argument('--ext', type=str, default='.png', help='Image extension')
    parser.add_argument('--save_dir', type=str, required=True, help='Folder to save SR images')    
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint file')
    parser.add_argument('--cuda', type=str, default=None, help='A list of gpus')
    args = parser.parse_args()

    if args.cuda is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    if (args.model == 'phiregan'):
        phiregan(args.data_type, args.mode)
    elif (args.model == 'edsr'):
        edsr(args.model_path, args.data_dir, args.valid_dir, args.save_dir, args.ext, args.mode, args.resume, args.cuda)
    elif (args.model == 'esrgan'):
        esrgan(args.model_path, args.data_dir, args.valid_dir, args.save_dir, args.ext, args.mode, args.resume, args.cuda)


if __name__ == '__main__':
    main()