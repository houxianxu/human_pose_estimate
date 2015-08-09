from my_scripts.transform import Transform
import cPickle as pickle
import argparse
import os
from data_utils import SIFT_BOW
from my_scripts.trainer import Trainer
from models.LinearNet_flic_sift import LinearNet_flic_sift


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
    lcn = False  # local contrast normalization
    trans = Transform(pad_inf, pad_sup, size=size, shift=shift, lcn=lcn)

    # set up the model and trainer for training
    data_dir = 'data/FLIC-full'  # store image
    # store corrdinates of joints for training datas
    train_fn = 'train_joints.csv'
    test_fn = 'test_joints.csv'

    sift_results = 'results/sift_results.sift'
    num_cluster = 200
    if os.path.exists(sift_results):
        print 'SIFT features are already existed'
        with open(sift_results, 'rb') as pickle_file:
            X_train, Y_train, X_test, Y_test = pickle.load(pickle_file)
    else:
        print 'Start to compute SIFT BOW ...'
        sift_bow = SIFT_BOW(num_cluster)
        X_train, Y_train, X_test, Y_test = sift_bow.build_bow_get_sift_feature(data_dir, train_fn, test_fn, trans=trans)
        sift_results = 'results/sift_results.sift'
        with open(sift_results, 'wb', pickle.HIGHEST_PROTOCOL) as pickle_file:
            pickle.dump((X_train, Y_train, X_test, Y_test), pickle_file)

    print 'Finish computing SIFT feature extraction'

    print X_train.shape, Y_train.shape, X_test.shape, Y_test.shape
    model = LinearNet_flic_sift(num_cluster)
    linerTrainer = Trainer(model=model, num_epochs=500)
    results = linerTrainer.train(X_train, Y_train)
    # best_model, train_loss_batch_history,
    # train_loss_epoch_history, val_loss_epoch_history
    results_file = 'results/sift_trained_resutls.chainer'
    with open(results_file, 'wb', pickle.HIGHEST_PROTOCOL) as pickle_file:
        pickle.dump(results, pickle_file)

