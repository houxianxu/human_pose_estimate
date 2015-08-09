from my_scripts.regressor_trainer import NetworkTrainer
from models.LinearNet_flic import LinearNet_flic
from my_scripts.transform import Transform
import cPickle as pickle
import argparse

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
    data_info_file = '%s/train_joints.csv' % data_dir
    model = LinearNet_flic(size)
    trainer = NetworkTrainer(data_dir=data_dir, data_info_file=data_info_file,
                             model=model, num_epochs=100, trans=trans)
    if args.mode == 'sgd':
        results = trainer.train()
        results_file = 'results/sgd_trained_resutls.chainer'
    elif args.mode == 'bgd':
        results = trainer.train_bgd()
        results_file = 'results/bgd_trained_resutls.chainer'
    else:
        raise Exception('Unrecognized mode type "%s"' % args.mode)
    # best_model, train_loss_batch_history,
    # train_loss_epoch_history, val_loss_epoch_history
    with open(results_file, 'wb', pickle.HIGHEST_PROTOCOL) as pickle_file:
        pickle.dump(results, pickle_file)
