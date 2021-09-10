from PhIREGAN.PhIREGANs import *
from EDSR.model import get_generator
from EDSR.test import *
from EDSR.train import *
from ESRGAN.train import *
from ESRGAN.inference import *
from metrics import *
from utils import *
from Interpolation.interpolation import *
import os

def phiregan(data_type, data_dir, mode):
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

        phiregans = PhIREGANs(data_type=data_type, mu_sig=mu_sig)
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
        if mode == 'test':
            phiregans.test(r=r,
                    data_path=LR_data_path,
                    model_path=model_path,
                    data_type=data_type,
                    model_type='phiregan',
                    timestep=timestep, 
                    batch_size=1)


def srcnn(data_type, data_dir, mode):
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
        model_path = 'PhIREGAN/models/{data_type}_mr-hr/trained_cnn/cnn'.format(data_type=data_type)
        r = [5]
        if data_type=='wind':
            mu_sig=[[0.7684, -0.4575], [5.02455, 5.9017]]
        elif data_type=='solar':
            mu_sig = [[344.3262, 113.7444], [386.9283, 117.9627]]

        phiregans = PhIREGANs(data_type=data_type, mu_sig=mu_sig)
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
        if mode == 'test':
            phiregans.test(r=r,
                    data_path=LR_data_path,
                    model_path=model_dir,
                    batch_size=1, 
                    data_type=data_type,
                    model_type='cnn',
                    timestep=timestep) 
        

def edsr(mode, data_dir=None, ext='.png', valid_dir=None, save_dir=None, model_path=None, resume=None, cuda=None):
    if mode == 'train':
        train(data_dir, valid_dir, ext, ext, resume=resume, cuda=cuda)
    if mode == 'test':
        global sess
        sess = tf.Session()
        model = get_generator('edsr', is_train=False)
        print("Loading model")
        model.load_weights(model_path)
        sr_from_folder(model, data_dir+'/LR', save_dir, ext)


def esrgan(mode, data_dir, save_dir, pretrain_generator=True):
    if mode == 'train':
        pretrain(data_dir=data_dir, pretrain_generator=pretrain_generator)
    if mode == 'test':
        test(valid_dir=data_dir, save_dir=save_dir)


def main():
    parser = argparse.ArgumentParser(description='Generate SR images')
    parser.add_argument('--model', required=True, type=str, help='Model architecture')
    parser.add_argument('--model_path', required=False, type=str, help='Path to a model')
    parser.add_argument('--mode', required=True, type=str, help='Train or test')    
    parser.add_argument('--data_type', required=True, type=str, help='Wind or solar')
    parser.add_argument('--data_dir', type=str, default=None, help='Path to training images/tfrecords')
    parser.add_argument('--valid_dir', type=str, default=None, help='Path to validation images/tfrecords')
    parser.add_argument('--ext', type=str, default='.png', help='Image extension')
    parser.add_argument('--save_dir', type=str, required=False, help='Folder to save SR images')    
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint file')
    parser.add_argument('--cuda', type=str, default=None, help='A list of gpus')
    args = parser.parse_args()

    if args.cuda is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    if (args.model == 'phiregan'):
        phiregan(args.data_type, args.data_dir, args.mode)
    elif (args.model == 'srcnn'):
        srcnn(args.data_type, args.data_dir, args.mode)
    elif (args.model == 'edsr'):
        edsr(args.mode, args.data_dir, args.ext, args.valid_dir, args.save_dir, args.model_path, args.resume, args.cuda)
    elif (args.model == 'esrgan'):
        esrgan(args.mode, args.data_dir, args.save_dir)


if __name__ == '__main__':
    main()