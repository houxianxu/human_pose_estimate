import cv2 as cv
import numpy as np


class Transform(object):

    def __init__(self, pad_inf, pad_sup, size, shift, lcn,
                 channel=3, num_joints=7):
        """
        Inputs:
        - pad_inf: (a float) infimum for padding size when cropping
        - pad_sup: (a float) supremum for padding size when cropping
        - size: (an int) resize size
        - shift: (an int) slide an image when cropping
        - lcn: if True perform local contrast normalization, otherwise don't
        """
        self.padding = [pad_inf, pad_sup]
        self.size = size
        self.shift = shift
        self.lcn = lcn
        self.channel = channel
        self._img = None  # initial in self.transfrom
        self._joints = None  # initialize in self.transform
        self.num_joints = num_joints  # initialize in self.transform

    def transform(self, img_name, x_data, joints, result_dir=None):
        """
        Inputs:
        - img_name: A string, i.e., the name the image name.
        - x_data: An array of shape (height, width, 3) to represent a image.
        - joints: A int vector storing coordinates of joints.

        Returns:
        - self.img: Transformed x_data
        - self.joints: Transformed joints

        Others:
        save the crop paramters to local file.
        """
        self._img = x_data
        self._joints = joints
        self.num_joints = len(joints) / 2
        if hasattr(self, 'padding') and hasattr(self, 'shift'):
            crop_param = self.crop()
        # if hasattr(self, 'flip'):
        #     self.fliplr()
        if hasattr(self, 'size'):
            self.resize()
        if hasattr(self, 'lcn') and self.lcn:
            self.contrast()

        # joint pos centerization
        h, w, c = self._img.shape
        center_pt = np.array([w / 2, h / 2], dtype=np.float32)  # x,y order
        joints = zip(self._joints[0::2], self._joints[1::2])
        joints = np.array(joints, dtype=np.float32) - center_pt
        joints[:, 0] /= w
        joints[:, 1] /= h
        self.joints = joints.flatten()

        # save the crop coordinates
        if result_dir is not None:
            self.save(img_name, crop_param, result_dir)
        return self._img, self.joints

    def crop(self):
        """
        image cropping
        Return the copy params x, y, w, h
        """
        joints = self._joints.reshape((len(self._joints) / 2, 2))

        x, y, w, h = cv.boundingRect(np.asarray([joints.tolist()]))

        # bounding rect extending
        inf, sup = self.padding
        r = sup - inf
        pad_w_r = np.random.rand() * r + inf  # inf~sup
        pad_h_r = np.random.rand() * r + inf  # inf~sup

        x -= (w * pad_w_r - w) / 2
        y -= (h * pad_h_r - h) / 2
        w *= pad_w_r
        h *= pad_h_r

        # shifting
        x += np.random.rand() * self.shift * 2 - self.shift
        y += np.random.rand() * self.shift * 2 - self.shift

        # clipping
        x, y, w, h = [int(z) for z in [x, y, w, h]]
        x = np.clip(x, 0, self._img.shape[1] - 1)
        y = np.clip(y, 0, self._img.shape[0] - 1)
        w = np.clip(w, 1, self._img.shape[1] - (x + 1))
        h = np.clip(h, 1, self._img.shape[0] - (y + 1))
        self._img = self._img[y:y + h, x:x + w]

        # joint shifting
        joints = np.asarray([(j[0] - x, j[1] - y) for j in joints])
        self._joints = joints.flatten()

        return x, y, w, h

    def resize(self):
        if not isinstance(self.size, int):
            raise Exception('self.size should be int')
        orig_h, orig_w, _ = self._img.shape
        self._joints[0::2] = self._joints[0::2] / float(orig_w) * self.size
        self._joints[1::2] = self._joints[1::2] / float(orig_h) * self.size
        self._img = cv.resize(self._img, (self.size, self.size),
                              interpolation=cv.INTER_NEAREST)

    def contrast(self):
        if self.lcn:
            if not self._img.dtype == np.float32:
                self._img = self._img.astype(np.float32)
            # local contrast normalization
            for ch in range(self._img.shape[2]):
                im = self._img[:, :, ch]
                im = (im - np.mean(im)) / \
                    (np.std(im) + np.finfo(np.float32).eps)
                self._img[:, :, ch] = im

    def fliplr(self):
        # it is bad to use it, it can't be reverted back.
        if np.random.randint(2) == 1 and self.flip is True:
            self._img = np.fliplr(self._img)
            self._joints[0::2] = self._img.shape[1] - self._joints[0:: 2]
            joints = zip(self._joints[0::2], self._joints[1::2])

            # shoulder
            joints[2], joints[4] = joints[4], joints[2]
            # elbow
            joints[1], joints[5] = joints[5], joints[1]
            # wrist
            joints[0], joints[6] = joints[6], joints[0]

            self._joints = np.array(joints).flatten()

    def revert(self, img, joints):
        """
        Revert RGB from 0-1 to 255
        Revert joints from 0-1 to 255
        """
        h, w, c = img.shape
        center_pt = np.array([w / 2, h / 2])
        joints = np.array(zip(joints[0::2], joints[1::2]))  # x,y order
        joints[:, 0] *= w
        joints[:, 1] *= h
        joints += center_pt
        joints = joints.astype(np.int32)

        if hasattr(self, 'lcn') and self.lcn:
            img -= img.min()
            img /= img.max()
            img *= 255
            img = img.astype(np.uint8)

        return img, joints

    def save(self, img_name, crop_param, result_dir):
        """
        Save crop paramters
        """
        param_str = img_name + ',' + ','.join([str(i) for i in crop_param]) + '\n'
        file_name = '{}/crop_param.csv'.format(result_dir)
        with open(file_name, 'a') as param_file:
            param_file.write(param_str)

    def revert_back_last_level(self, new_coord, img_name, param_dict):
        """
        Compute the last level coordinates.
        param_dict is a dictionary {img_name: x, y, w, h}
        """
        new_joint_x, new_joint_y = new_coord
        orig_h = param_dict[img_name][3]
        orig_w = param_dict[img_name][2]

        # revert resize part
        re_joint_x = float(new_joint_x) / self.size * orig_w
        re_joint_y = float(new_joint_y) / self.size * orig_h

        # add crop padding
        crop_x = param_dict[img_name][0]
        crop_y = param_dict[img_name][1]

        re_joint_x += crop_x
        re_joint_y += crop_y
        return re_joint_x, re_joint_y



