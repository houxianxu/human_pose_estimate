import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from IPython.core.display import clear_output
import sys
from my_scripts.regressor_trainer import NetworkTrainer
from models.LinearNet_flic import LinearNet_flic
from my_scripts.transform import Transform

# %matplotlib inline
# plt.rcParams['figure.figsize'] = (15.0, 8.0) # set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'

# # for auto-reloading extenrnal modules
# # see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
# %load_ext autoreload
# %autoreload 2


# set up fransform parameters to find the bounding box, crop and resize the images
# transfrom parameters
pad_inf = 1  # infimum scale
pad_sup = 1.5  # supremum scale
size = 100  # resize size
shift = 3  # radomly shift the bounding box
lcn = True  # local contrast normalization
trans = Transform(pad_inf, pad_sup, size=size, shift=shift, lcn=lcn)


# set up the model and trainer for training

data_dir = 'data/FLIC-full' # store image 
data_info_file = '%s/train_joints.csv' % data_dir # store corrdinates of joints for training data
model = LinearNet_flic(size)
trainer = NetworkTrainer(data_dir=data_dir, data_info_file=data_info_file,
                         model=model, trans=trans)

if __name__ == '__main__':
    best_model, train_loss_history, val_loss_history = trainer.train_bgd()
