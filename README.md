<div align="center">
# SuperEnergyBench: Evaluating Super-Resolution Models on US Wind and Solar Data

Caltech SURF Project for 2020-2021

Rupa Kurinchi-Vendhan, Lucien Werner, Björn Lütjens, Ritwik Gupta

## Abstract
As the United States constructs additional renewable wind and solar energy power plants, policy makers in charge of operational decision making, scheduling, and resource allocation are faced with challenges introduced by the variability in spatial resolution in solar irradiance and wind speeds. Physics-based short-term forecasting models predict wind speeds and solar irradiance fields at coarse resolutions. Thus, machine learning-based super-resolution methods have been developed to provide higher fidelity for decision making. We generate a machine-learning ready dataset of wind and solar data from NREL databases. Additionally, we present a benchmark of super-resolution methods against this data. In addition to simple interpolation methods, we investigate three machine learning methods: the [physics-informed resolution-enhancing generative adversarial network (PhIREGAN)](https://www.pnas.org/content/117/29/16805) model, the [enhanced super-resolution generative adversarial network (ESRGAN)](http://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Wang_ESRGAN_Enhanced_Super-Resolution_Generative_Adversarial_Networks_ECCVW_2018_paper.pdf), and an [enhanced deep super-resolution (EDSR)](https://arxiv.org/abs/1707.02921) network.

<div align="left">

## Dataset
The dataset used for this project is available through the [data.ipynb](https://github.com/RupaKurinchiVendhan/SuperEnergyBench/blob/main/data.ipynb) notebook. This file also contains instructions for generating your own machine learning-ready dataset, with flexibility to change the parameters inputted to NREL's WIND Toolkit and NSRDB.

Quick Links:

[wind data](https://www.filefactory.com/file/77ekv9v0ssx4/wind.zip)

## Training
To train the PhIREGAN, ESRGAN, or EDSR models to achieve 5x super-resolution (SR), use the following commands.

**PhIREGAN**

`main.py --model=phiregan --mode=train --data_dir=path/to/train/data --data_type=data_type`

Here, `data` should be a folder of TFRecords. The `data_type` can either be `wind` or `solar`. To train the SR CNN, replace `phiregan` with `srcnn`. 

**ESRGAN**

`main.py --model=esrgan --mode=train --data_dir=path/to/train/data`

As opposed to the PhIREGAN, the data directories for EDSR and ESRGAN must be a folder with HR and LR subdirectories of images. 

**EDSR**

`main.py --model=edsr --mode=train --data_dir=path/to/train/data --valid_dir=path/to/valid/data --cuda=0`

## Testing
Once you have trained your models, use the commands below to run them on test data.

**PhIREGAN**

`main.py --model=phiregan --mode=test --data_dir=path/to/test/data --data_type=data_type`

To test the SR CNN, replace `phiregan` with `srcnn`.

**ESRGAN**

`main.py --model=esrgan --mode=test --data_dir=path/to/test/data --save_dir=path/to/save`

Here, `save_dir` is the directory where SR outputs should be saved.

**EDSR**

`main.py --model=edsr --mode=test --data_dir=path/to/test/data --save_dir=path/to/save --model_path=path/to/model --cuda=0`

**Comparison**

To calculate the standard image quality metric values for the SR outputs of each model, use [test.py](https://github.com/RupaKurinchiVendhan/SuperEnergyBench/blob/main/test.py). Alternatively, you can generate kinetic energy spectra using [energy.py](https://github.com/RupaKurinchiVendhan/SuperEnergyBench/blob/main/energy.py) and normalized semivariograms using [semivariogram.py](https://github.com/RupaKurinchiVendhan/SuperEnergyBench/blob/main/energy.py).