import time
import numpy as np
import cv2 as cv
import copy
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
    train_dl = train_dl[0:1000]
    test_dl = test_dl[0:100]
    N_train = train_dl.shape[0]
    N_test = test_dl.shape[0]
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


class SIFT_BOW(object):
    def __init__(self, K=200):
        self.K = K

    def build_bow_get_sift_feature(self, data_dir, train_fn, test_fn, trans=None):
        """
        Load original flic data and extract SIFT with bag of words

        Inputs:
            - data_dir: directory to store images
            - train_fn: a csv file including train image name
                        and coordinates of joints
            - test_fn: a csv file including test image name
                        and coordinates of joints
            - trans: a class to transform image

        Returns:
            - X_train: an array of shape (N_train, K*128)
            - Y_train: an array of shape (N_test, num_joints * 2)
            - X_test: an array of shape (N_train, K*128)
            - Y_test: an array of shape (N_test, num_joints * 2)
        """
        # load file name list
        train_fn = '%s/%s' % (data_dir, train_fn)
        test_fn = '%s/%s' % (data_dir, test_fn)
        train_dl = np.array([l.strip() for l in open(train_fn).readlines()])
        test_dl = np.array([l.strip() for l in open(test_fn).readlines()])
        # train_dl = train_dl[0:100]
        # test_dl = test_dl[0:100]
        N_train = train_dl.shape[0]
        N_test = test_dl.shape[0]

        X_train = None
        Y_train = None
        X_test = None
        Y_test = None

        # set up Bag of Word
        bow_train, bow_extract = self.get_bow(self.K)

        pbar = ProgressBar(N_train)
        for i, x in enumerate(train_dl):
            x = x.strip().split(',')
            img_file_name = '%s/images/%s' % (data_dir, x[0])
            ori_img = cv.imread(img_file_name)
            ori_joints = np.array([int(float(p)) for p in x[1:]])
            # transform, i.e., cropping
            tran_img, _ = trans.transform(img_file_name, ori_img.copy(), ori_joints.copy())
            # compute SIFT
            _, des, _ = self.extract_sift_one_img(tran_img, descriptor=True)
            if len(des) > 100:
                mask = np.random.choice(len(des), 100)
                bow_train.add(des[mask])
            else:
                bow_train.add(des)
            pbar.update(i + 1)
        pbar.finish()

        # compute vocabulary
        print 'start to clustering SIFT features ... '
        start_time = time.time()
        vocabulary = bow_train.cluster()
        elapsed_time = time.time() - start_time
        bow_extract.setVocabulary(vocabulary)
        print 'It take {} sec for clustering with {} images'.format(
              elapsed_time, N_train)

        # extract tranining data from bow_extract
        X_train, Y_train = self.extract_feature_from_bow(train_dl, data_dir,
                                                    bow_extract, trans)
        X_test, Y_test = self.extract_feature_from_bow(test_dl, data_dir,
                                                  bow_extract, trans)

        return X_train, Y_train, X_test, Y_test

    def extract_sift_one_img(self, img_array, descriptor=False, draw_kp=False):
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

    def get_bow(self, K):
        """
        Get Bag-Of-Words trainer and extractor
        """
        extract = cv.DescriptorExtractor_create('SIFT')
        flann_params = dict(algorithm=0, trees=5)
        matcher = cv.FlannBasedMatcher(flann_params, {})
        bow_train = cv.BOWKMeansTrainer(K)
        bow_extract = cv.BOWImgDescriptorExtractor(extract, matcher)

        return bow_train, bow_extract

    def extract_feature_from_bow(self, data_dl, data_dir, bow_extract, trans):
        """
        Extract feature based on bow_extract
        """
        N = data_dl.shape[0]
        num_joints = len(data_dl[0].strip().split(',')) / 2
        X = np.zeros((N, self.K))
        Y = np.zeros((N, num_joints * 2))
        pbar = ProgressBar(N)
        for i, x in enumerate(data_dl):
            x = x.strip().split(',')
            img_file_name = '%s/images/%s' % (data_dir, x[0])
            ori_img = cv.imread(img_file_name)
            ori_joints = np.array([int(float(p)) for p in x[1:]])
            # transform, i.e., cropping
            tran_img, tran_joints = trans.transform(img_file_name, ori_img.copy(), ori_joints.copy())
            # compute SIFT
            kp, des, _ = self.extract_sift_one_img(tran_img, descriptor=True)
            img_feature = bow_extract.compute(tran_img, kp)
            X[i] = img_feature
            Y[i] = tran_joints
            pbar.update(i + 1)
        pbar.finish()
        return X, Y


class Patch_Extraction(object):
    """
    Extract several batch information of images.
    """
    def __init__(self, patch_size=30):
        # store the extracted patch ratio
        self.patch_size = patch_size

    def build_patch_feature(self, data_dir, train_fn, test_fn, trans=None):
        """
        Load original images and transform, and extract patches.

        Inputs:
            - data_dir: directory to store images
            - train_fn: a csv file including train image name
                        and coordinates of joints
            - test_fn: a csv file including test image name
                        and coordinates of joints
            - trans: a class to transform image

        Returns:
            - X_train: an array of shape (N_train, patch_size*patch_size*3)
            - Y_train: an array of shape (N_test, num_joints * 2)
            - X_test: an array of shape (N_train, patch_size*patch_size*3)
            - Y_test: an array of shape (N_test, num_joints * 2)
        """
        train_fn = '%s/%s' % (data_dir, train_fn)
        test_fn = '%s/%s' % (data_dir, test_fn)
        train_dl = np.array([l.strip() for l in open(train_fn).readlines()])
        test_dl = np.array([l.strip() for l in open(test_fn).readlines()])
        # train_dl = train_dl[0:100]
        # test_dl = test_dl[0:100]

        X_train = None
        Y_train = None
        X_test = None
        Y_test = None

        X_train, Y_train = self.extract_imgs(train_dl, data_dir, trans)
        X_test, Y_test = self.extract_imgs(test_dl, data_dir, trans)

        return X_train, Y_train, X_test, Y_test

    def extract_imgs(self, data_dl, data_dir, trans):
        N = data_dl.shape[0]
        num_joints = len(data_dl[0].strip().split(',')) / 2
        X = np.zeros((N, self.patch_size * self.patch_size * 3 * num_joints))
        y = np.zeros((N, num_joints * 2))

        pbar = ProgressBar(N)
        for i, x in enumerate(data_dl):
            x = x.strip().split(',')
            img_file_name = '%s/images/%s' % (data_dir, x[0])
            ori_img = cv.imread(img_file_name)
            ori_joints = np.array([int(float(p)) for p in x[1:]])
            # transform, i.e., cropping
            tran_img, tran_joints = trans.transform(img_file_name, ori_img.copy(), ori_joints.copy())
            # revert to get the original coordinates of joints
            _, rev_tran_joints = trans.revert(tran_img.copy(), tran_joints.copy())
            patch_array = self.extract_one_img(tran_img, rev_tran_joints)[0]
            X[i] = patch_array.flatten()
            y[i] = tran_joints  # store the normalized joints
            pbar.update(i)
        pbar.finish()

        return X, y

    def extract_one_img(self, img, joints):
        """
        Inputs:
        - img: (array) represents an image
        - joints: a (list) of tuple of joints

        Output:
        - patch_array: a (array) array represents all patches
        - patch_info: a (list) store the patch corrdinates, for plot
        """
        patch_size = self.patch_size
        patch_array = []
        patch_info = []
        for i, joint in enumerate(joints):
            patch_results = self.extract_one_joint(img.copy(), joint, patch_size)
            patch_array.append(patch_results[0])
            patch_info.append(patch_results[1])
        return np.asarray(patch_array), patch_info

    def extract_one_joint(self, img, joint, batch_size):
        """
        Extract one patch image from one joint.
        """
        x, y = joint
        y_min = y - batch_size / 2
        y_max = y + batch_size / 2
        x_min = x - batch_size / 2
        x_max = x + batch_size / 2

        # clipping to make sure within the original image
        y_min = int(np.clip(y_min, 0, img.shape[0] - 1))
        y_max = int(np.clip(y_max, 0, img.shape[0] - 1))
        x_min = int(np.clip(x_min, 0, img.shape[1] - 1))
        x_max = int(np.clip(x_max, 0, img.shape[1] - 1))

        # cropping
        patch = img[y_min:y_max + 1, x_min:x_max + 1]
        if patch.shape != (batch_size, batch_size, 3):
            # in case the cropping is outside of boundary
            patch = cv.resize(patch, (batch_size, batch_size),
                              interpolation=cv.INTER_NEAREST)
        patch_info = x_min, x_max, y_min, y_max
        return patch, patch_info



