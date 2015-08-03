#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
from os.path import basename, splitext
import cv2 as cv
import numpy as np
import glob
import matplotlib.pyplot as plt


def draw_limb(img, joints, i, j, color):
    cv.line(img, joints[i], joints[j], (255, 255, 255),
            thickness=2, lineType=cv.CV_AA)
    cv.line(img, joints[i], joints[j], color,
            thickness=1, lineType=cv.CV_AA)

    return img


def draw_joints(img, joints, line=True, text_scale=0.5):
    h, w, c = img.shape

    if line:
        # left hand to left elbow
        img = draw_limb(img, joints, 0, 1, (0, 255, 0))
        img = draw_limb(img, joints, 1, 2, (0, 255, 0))
        img = draw_limb(img, joints, 4, 5, (0, 255, 0))
        img = draw_limb(img, joints, 5, 6, (0, 255, 0))
        img = draw_limb(img, joints, 2, 4, (255, 0, 0))
        neck = tuple((np.array(joints[2]) + np.array(joints[4])) / 2)
        joints.append(neck)
        img = draw_limb(img, joints, 3, 7, (255, 0, 0))
        joints.pop()

    # all joint points
    for j, joint in enumerate(joints):
        cv.circle(img, joint, 5, (0, 0, 255), -1)
        cv.circle(img, joint, 3, (0, 255, 0), -1)
        cv.putText(img, '%d' % j, joint, cv.FONT_HERSHEY_SIMPLEX, text_scale,
                   (0, 0, 0), thickness=3, lineType=cv.CV_AA)
        cv.putText(img, '%d' % j, joint, cv.FONT_HERSHEY_SIMPLEX, text_scale,
                   (255, 255, 255), thickness=1, lineType=cv.CV_AA)

    return img


def show_ori_trans_pred_joints(N, data_info, data_dir, trans, model):
    """
    Show the original, trainsformed and predicted joints

    Inputs:
    - N: (int) number of images to show
    - data_info: An (array) containing image name and joints
    - data_dir: A (string) representing the image directory
    - trans: An (object) trainsforming image
    - model: used to predict joints
    """
    size = 220
    ori_imgs = np.zeros((N, 480, 720, 3))
    tran_imgs = np.zeros((N, size, size, 3))
    pred_imgs = np.zeros((N, size, size, 3))
    num_bounding_box = 5

    for i, x in enumerate(data_info):
        x = x.strip().split(',')
        img_file_name = '%s/images/%s' % (data_dir, x[0])
        ori_joints = np.array([int(float(p)) for p in x[1:]])
        ori_img = cv.imread(img_file_name)
        # draw joints
        draw_ori_joints = zip(ori_joints[0::2], ori_joints[1::2])
        ori_img_joints = draw_joints(ori_img.copy(), draw_ori_joints)

        # get bounding box
        for j in xrange(num_bounding_box):
            trans._img = ori_img
            trans._joints = ori_joints
            x, y, w, h = trans.crop()
            # draw bouding box
            cv.rectangle(ori_img_joints, (x, y), (x + w, y+h), (0, 255, 255), thickness=3)
        ori_imgs[i] = ori_img_joints

        # tranformed image
        tran_img, tran_joints = trans.transform(img_file_name, ori_img.copy(), ori_joints.copy())
        tran_img_rev, tran_joints_rev = trans.revert(tran_img.copy(), tran_joints)
        draw_tran_joints = [tuple(p) for p in tran_joints_rev] # must tuple in opencv
        tran_imgs[i] = draw_joints(tran_img_rev.copy(), draw_tran_joints)

        # predict image
        pred_img = tran_img.transpose((2, 0, 1))[np.newaxis, :]  # need transpose and add a new dimenstion
        _, pred_joints_data = model.forward(pred_img, tran_joints, train=False)
        pred_joints = pred_joints_data.data[0]  # just one image
        _, pred_joints_rev = trans.revert(tran_img.copy(), pred_joints)
        draw_pred_joints = [tuple(p) for p in pred_joints_rev]
        pred_imgs[i] = draw_joints(tran_img_rev.copy(), draw_pred_joints)

    for y in xrange(3):
        for i in xrange(N):
            plt_idx = y * N + i + 1
            plt.subplot(3, N, plt_idx)
            if y == 0:  # show original image
                b,g,r = cv.split(ori_imgs[i])  # get b,g,r
            elif y == 1:  # transform image
                b,g,r = cv.split(tran_imgs[i])
            else:  # predict joints
                b,g,r = cv.split(pred_imgs[i])
            rgb_img = cv.merge([r,g,b])   
            plt.imshow(rgb_img.astype('uint8'))
            plt.axis('off')
    plt.show()    

if __name__ == '__main__':
    out_dir = 'data/FLIC-full/test_images'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    joints_csv = csv.reader(open('data/FLIC-full/test_joints.csv'))
    for line in joints_csv:
        img_fn = 'data/FLIC-full/images/%s' % line[0]
        img = cv.imread(img_fn)

        joints = [int(float(j)) for j in line[1:]]
        joints = zip(joints[0::2], joints[1::2])

        draw = draw_joints(img, joints)
        cv.imwrite('%s/%s' % (out_dir, basename(img_fn)), draw)

        print img_fn



