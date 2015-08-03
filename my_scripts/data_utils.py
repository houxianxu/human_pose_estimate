import numpy as np
import cv2 as cv
from progressbar import ProgressBar


def load_flic_naive(data_dir, train_fn, test_fn, trans=None, width=720,
                    height=480, channel=3):
    """
    Load original flic data naive
    Try to load all the images and take too much memeory
    And it can't work in my laptop

    Inputs:
        - data_dir: directory to store images
        - train_fn: a csv file including train image name
                    and coordinates of joints
        - test_fn: a csv file including test image name
                    and coordinates of joints
        - trans: a class to transform image
        - width: the width of an image
        - height: the height of an image
        - channel: the channel of an image

    Returns:
        - X_train: an array of shape (N_train, 3, height, width)
        - Y_train: an array of shape (N_test, 3, height, width)
        - X_test: an array of shape (N_train, size * 2)
        - Y_test: an array of shape (N_test, num_joints * 2)
    """
    # load file name list
    train_fn = '%s/%s' % (data_dir, train_fn)
    test_fn = '%s/%s' % (data_dir, test_fn)
    train_dl = np.array([l.strip() for l in open(train_fn).readlines()])
    test_dl = np.array([l.strip() for l in open(test_fn).readlines()])

    N_train = train_dl.shape[0]
    N_test = test_dl.shape[0]
    N_train = 10000
    num_joints = len(train_dl[0].strip().split(',')) / 2

    if trans is not None:  # transform images
        height = trans.size
        width = trans.size

    X_train = np.zeros((N_train, channel, height, width))
    Y_train = np.zeros((N_train, num_joints * 2))
    X_test = np.zeros((N_test, channel, height, width))
    Y_test = np.zeros((N_test, num_joints * 2))

    if trans is not None:
        # load train image
        pbar = ProgressBar(N_train)
        for i in xrange(N_train):
            img_info = train_dl[i].split(',')
            img_fn = '%s/images/%s' % (data_dir, img_info[0])
            img = cv.imread(img_fn)
            joints = np.asarray([int(float(p)) for p in img_info[1:]])
            x_data_trans, joints_trans = trans.transform(img_fn, img, joints)
            X_train[i] = x_data_trans.transpose((2, 0, 1))
            Y_train[i] = joints_trans
            pbar.update(i + 1)
        pbar.finish()

        # load test image
        pbar = ProgressBar(N_test)
        for i in xrange(N_test):
            img_info = test_dl[i].split(',')
            img_fn = '%s/images/%s' % (data_dir, img_info[0])
            img = cv.imread(img_fn)
            joints = np.asarray([int(float(p)) for p in img_info[1:]])
            x_data_trans, joints_trans = trans.transform(img_fn, img, joints)
            X_test[i] = x_data_trans.transpose((2, 0, 1))
            Y_test[i] = joints_trans
            pbar.update(i)
        pbar.finish()

    else:
        pbar = ProgressBar(N_train)
        for i in xrange(N_train):
            img_info = train_dl[i].split(',')
            img_fn = '%s/images/%s' % (data_dir, img_info[0])
            img = cv.imread(img_fn)
            joints = np.asarray([int(float(p)) for p in img_info[1:]])
            X_train[i] = img.transpose((2, 0, 1))
            Y_train[i] = joints
            pbar.update(i + 1)
        pbar.finish()

        # load test image
        pbar = ProgressBar(N_test)
        for i in xrange(N_test):
            img_info = test_dl[i].split(',')
            img_fn = '%s/images/%s' % (data_dir, img_info[0])
            img = cv.imread(img_fn)
            joints = np.asarray([int(float(p)) for p in img_info[1:]])
            X_test[i] = img.transpose((2, 0, 1))
            Y_test[i] = joints
            pbar.update(i)
        pbar.finish()
    return X_train, Y_train, X_test, Y_test

