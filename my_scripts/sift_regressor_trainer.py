import numpy as np
import logging
import time
import os
import cv2 as cv
import copy
import cPickle as pickle
from chainer import optimizers, cuda
from multiprocessing import Process, Queue
from progressbar import ProgressBar
from my_scripts.draw_loss import draw_loss_from_list, draw_batch_loss
reload(logging)  # for ipython notebook

