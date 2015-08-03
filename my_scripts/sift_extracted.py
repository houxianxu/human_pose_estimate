import numpy as np
import cv2 as cv


def extract_sift_one_img(img_array, descriptor=False, draw_kp=False):
    """
    Extract sift feature from image

    Inputs:
    - img_array: (array) represents an image
    - descriptor: (logic) if True, compute descriptor
    - draw_kp: (logic) if True, draw keypoints

    Returns:
    - kp: (list) of keypoint of sift feature
    - des: (array) of (N, 128) as descriptor of keypoint
    - kp_img: (array) represent original image with keypoints
    """
    gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT()

    kp, des, kp_img = None, None, None
    if descriptor:
        kp, des = sift.detectAndCompute(gray, None)
    else:
        kp = sift.detect(gray, None)

    if draw_kp:
        kp_img = cv.drawKeypoints(img_array, kp)

    return kp, des, kp_img
