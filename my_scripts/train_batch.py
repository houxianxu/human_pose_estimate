from my_scripts.transform import Transform
import cPickle as pickle
import argparse
import os
from data_utils import Patch_Extraction
from my_scripts.trainer import Trainer
from models.LinearNet_flic_patch import LinearNet_flic_patch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='sgd',
                        help='mode sgd or bgd')
    args = parser.parse_args()
    # set up fransform parameters to find bounding box, crop and resize images
    # transfrom parameters
    pad_inf = 1.5  # infimum scale
    pad_sup = 2.0  # supremum scale
    size = 100  # resize size
    shift = 3  # radomly shift the bounding box
    lcn = True  # local contrast normalization
    trans = Transform(pad_inf, pad_sup, size=size, shift=shift, lcn=lcn)

    # set up the model and trainer for training
    data_dir = 'data/FLIC-full'  # store image
    # store corrdinates of joints for training datas
    train_fn = 'train_joints.csv'
    test_fn = 'test_joints.csv'

    patch_size = 30
    print 'Start to compute patch features  ...'
    patch_extraction = Patch_Extraction(patch_size)
    X_train, Y_train, X_test, Y_test = patch_extraction.build_patch_feature(
                                       data_dir, train_fn, test_fn, trans)
    print 'Finish computing patch feature extraction'

    print X_train.shape, Y_train.shape, X_test.shape, Y_test.shape
    model = LinearNet_flic_patch(patch_size)
    linerTrainer = Trainer(model=model, num_epochs=1000)
    results = linerTrainer.train(X_train, Y_train)
    # # best_model, train_loss_batch_history,
    # # train_loss_epoch_history, val_loss_epoch_history
    results_file = 'results/batch_trained_resutls.chainer'
    with open(results_file, 'wb', pickle.HIGHEST_PROTOCOL) as pickle_file:
        pickle.dump(results, pickle_file)

