# Load the FLIC label data and  write into csv file
from __future__ import print_function
import numpy as np
from scipy.io import loadmat


def get_joint_list(joints):
    head = np.asarray(joints['reye']) + \
        np.asarray(joints['leye']) + \
        np.asarray(joints['nose'])
    head /= 3
    del joints['reye']
    del joints['leye']
    del joints['nose']
    joints['head'] = head.tolist()
    joint_pos = [joints['lwri']]
    joint_pos.append(joints['lelb'])
    joint_pos.append(joints['lsho'])
    joint_pos.append(joints['head'])
    joint_pos.append(joints['rsho'])
    joint_pos.append(joints['relb'])
    joint_pos.append(joints['rwri'])

    return np.array(joint_pos).flatten()


def load_images_names_and_joints():
    """
    Load the FLIC label data and write into csv files

    Outputs
    -------
    data/FLIC-full/train_joints.csv
    data/FLIC-full/test_joints.csv
    """
    training_indices = loadmat('data/FLIC-full/tr_plus_indices.mat')
    training_indices = training_indices['tr_plus_indices'].flatten()

    examples = loadmat('data/FLIC-full/examples.mat')
    examples = examples['examples'][0]
    joint_ids = ['lsho', 'lelb', 'lwri', 'rsho', 'relb', 'rwri', 'lhip',
                 'lkne', 'lank', 'rhip', 'rkne', 'rank', 'leye', 'reye',
                 'lear', 'rear', 'nose', 'msho', 'mhip', 'mear', 'mtorso',
                 'mluarm', 'mruarm', 'mllarm', 'mrlarm', 'mluleg', 'mruleg',
                 'mllleg', 'mrlleg']

    fp_train = open('data/FLIC-full/train_joints.csv', 'w')
    fp_test = open('data/FLIC-full/test_joints.csv', 'w')
    for i, example in enumerate(examples):
        joint = example[2].T
        joint = dict(zip(joint_ids, joint))
        fname = example[3][0]
        joint = get_joint_list(joint)
        msg = '{},{}'.format(fname, ','.join([str(j) for j in joint.tolist()]))

        if i in training_indices:
            print(msg, file=fp_train)
        else:
            print(msg, file=fp_test)

if __name__ == '__main__':
    load_images_names_and_joints()
