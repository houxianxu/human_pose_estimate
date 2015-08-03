from __future__ import print_function
import argparse
import logging
import time
import os
import imp
import shutil
import numpy as np
from chainer import optimizers, cuda
from transform import Transform
import cPickle as pickle
from draw_loss import draw_loss_curve
from progressbar import ProgressBar
from multiprocessing import Process, Queue
reload(logging)  # for ipython notebook


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/AlexNet_flic.py',
                        help='model definition file in models dir')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--prefix', type=str, default='AlexNet_flic')
    parser.add_argument('--snapshot', type=int, default=10)
    parser.add_argument('--datadir', type=str, default='data/FLIC-full')
    parser.add_argument('--channel', type=int, default=3)
    parser.add_argument('--flip', type=int, default=0,
                        help='flip left and right for data augmentation')
    parser.add_argument('--size', type=int, default=220,
                        help='resizing')
    parser.add_argument('--crop_pad_inf', type=float, default=1.5,
                        help='random number infimum for padding size when cropping')
    parser.add_argument('--crop_pad_sup', type=float, default=2.0,
                        help='random number supremum for padding size when cropping')
    parser.add_argument('--shift', type=int, default=5,
                        help='slide an image when cropping')
    parser.add_argument('--lcn', type=int, default=1,
                        help='local contrast normalization for data augmentation')
    parser.add_argument('--joint_num', type=int, default=7)
    parser.add_argument('--fname_index', type=int, default=0,
                        help='the index of image file name in a csv line')
    parser.add_argument('--joint_index', type=int, default=1,
                        help='the start index of joint values in a csv line')
    parser.add_argument('--restart_from', type=str, default=None,
                        help='*.chainermodel file path to restart from')
    parser.add_argument('--epoch_offset', type=int, default=0,
                        help='set greater than 0 if you restart from a chainermodel pickle')
    parser.add_argument('--opt', type=str, default='AdaGrad',
                        choices=['AdaGrad', 'MomentumSGD', 'Adam'])
    parser.add_argument('--level', type=int, default=0,
                        help='the cascade level')
    parser.add_argument('--param', type=str, default=None, 
                        help='the trained model file')
    parser.add_argument('--past_level_model', type=str, default='models/LinearNet_flic.py',
                        help='last model used in previous level')
    parser.add_argument('--pred_joint', type=int, default=0,
                        help='single joint for prediction')
    args = parser.parse_args()

    return args


def load_model(args):
    """
    load trained model from last level
    """
    model_fn = os.path.basename(args.past_level_model)
    model_name = model_fn.split('.')[0]
    module = imp.load_source(model_fn.split('.')[0], args.past_level_model)
    model = pickle.load(open(args.param))
    return model


def creat_result_dir(args):
    """
    create result directory and log file
    """
    if args.restart_from is None:
        result_dir = 'results/' + os.path.basename(args.model).split('.')[0]
        result_dir += '_' + time.strftime('%Y-%m-%d_%H-%M-%S_')
        result_dir += str(time.time()).replace('.', '')
        result_dir += '_level_{}'.format(args.level)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        log_fn = '%s/log.txt' % result_dir
        logging.basicConfig(filename=log_fn, level=logging.DEBUG)
        logging.info(args)
    else:
        result_dir = '.'
        log_fn = 'log.txt'
        logging.basicConfig(filename=log_fn, level=logging.DEBUG)
        logging.info(args)

    return log_fn, result_dir


def get_model_optimizer(result_dir, args):
    """
    Load model to be trained and optimizer
    """
    model_fn = os.path.basename(args.model)
    model_name = model_fn.split('.')[0]
    module = imp.load_source(model_name, args.model)
    Net = getattr(module, model_name)

    dst = '%s/%s' % (result_dir, model_fn)
    if not os.path.exists(dst):  # copy model to result
        shutil.copy(args.model, dst)

    # copy train.py file to result
    dst = '%s/%s' % (result_dir, os.path.basename(__file__))
    if not os.path.exists(dst):
        shutil.copy(__file__, dst)

    # prepare the model
    # for GPU
    model = Net()
    if args.restart_from is not None:
        if args.gpu >= 0:
            cuda.init(args.gpu)
        model = pickle.load(open(args.restart_from, 'rb'))
    if args.gpu >= 0:
        cuda.init(args.gpu)
        model.to_gpu()

    # prepare optimizer
    if args.opt == 'AdaGrad':
        optimizer = optimizers.AdaGrad(lr=0.0005)
    elif args.opt == 'MomentumSGD':
        optimizer = optimizers.MomentumSGD(lr=0.0005, momentum=0.9)
    elif args.opt == 'Adam':
        optimizer = optimizers.Adam()
    else:
        raise Exception('No optimizer is selected')
    optimizer.setup(model.collect_parameters())

    return model, optimizer


def load_dataset(args):
    """
    Load image name and correspoinding labels

    Outputs
    -------
    A tuple of
        - train_dl: a list of file name and labels for training data
        - test_dl: a list of file name and labels for test data
    """
    train_fn = '%s/train_joints.csv' % args.datadir
    test_fn = '%s/test_joints.csv' % args.datadir
    train_dl = np.array([l.strip() for l in open(train_fn).readlines()])
    test_dl = np.array([l.strip() for l in open(test_fn).readlines()])

    return train_dl, test_dl


def load_data(stage, trans, args, input_q, data_q):
    """
    Used for multiprocess to load and transform data
    """
    c = args.channel
    s = args.size  # output image size
    d = args.joint_num * 2  # number of labels (x, y)

    while True:  # Listen to data in train function
        x_batch = input_q.get()
        if x_batch is None:
            break

        input_data = np.zeros((args.batchsize, c, s, s))
        label = np.zeros((args.batchsize, d))

        # data load and transformation
        train = stage == 'train'
        for i, x in enumerate(x_batch):
            x_data, t = trans.transform(x.split(','), args.datadir, train,
                                        args.fname_index, args.joint_index)
            input_data[i] = x_data.transpose((2, 0, 1))
            label[i] = t

        # decide the training level
        if args.level != 0:
            trained_model = load_model(args)
            _, preds = trained_model.forward(input_data, label)
            label_one_joint = np.zeros((args.batchsize, 2))
            # transform based on single joint
            for i, x in enumerate(x_batch):
                x_data = input_data[i].transpose((1, 2, 0))
                pred = preds.data[i]
                x_data, pred = trans.revert(x_data, pred)
                pred = pred.flatten()
                x_data, t = trans.transform(x.split(','), args.datadir, train,
                        level=args.level, sub_img=x_data,
                        label=label[i], pred=pred, pred_joint=args.pred_joint)

                input_data[i] = x_data.transpose((2, 0, 1))
                label_one_joint[i] = t

            label = label_one_joint

        data_q.put([input_data, label])


def train(train_dl, N, model, optimizer, args, input_q, data_q):
    pbar = ProgressBar(N)  # Show current iteration
    perm = np.random.permutation(N)
    sum_loss = 0

    # putting data to input_q by batch
    for i in xrange(0, N, args.batchsize):
        x_batch = train_dl[perm[i:i + args.batchsize]]
        input_q.put(x_batch)
    # Notice: at the same time, data are loaded to transformed

    # training
    for i in xrange(0, N, args.batchsize):
        input_data, label = data_q.get()  # transformed

        if args.gpu >= 0:
            input_data = cuda.to_gpu(input_data.astype(np.float32))
            label = cuda.to_gpu(label.astype(np.float32))

        optimizer.zero_grads()
        loss, pred = model.forward(input_data, label, train=True)
        # print('loss_train{}'.format(loss.data))
        loss.backward()
        optimizer.update()

        sum_loss += float(cuda.to_cpu(loss.data)) * args.batchsize
        pbar.update(i + args.batchsize if (i + args.batchsize) < N else N)
    pbar.update(N)
    return sum_loss


def test(test_dl, test_N, model, args, input_q, data_q):
    """
    Test using test datasets
    """
    pbar = ProgressBar(test_N)
    sum_loss = 0

    # putting all data
    for i in xrange(0, test_N, args.batchsize):
        x_batch = test_dl[i: i + args.batchsize]
        input_q.put(x_batch)

    for i in xrange(0, test_N, args.batchsize):
        input_data, label = data_q.get()

        if args.gpu >= 0:
            input_data = cuda.to_gpu(input_data.astype(np.float32))
            label = cuda.to_gpu(label.astype(np.float32))

        loss, pred = model.forward(input_data, label, train=False)
        # print('loss_test{}'.format(loss.data))

        sum_loss += float(cuda.to_cpu(loss.data)) * args.batchsize
        pbar.update(i + args.batchsize if (i + args.batchsize) < test_N
                    else test_N)
    return sum_loss


def get_log_msg(stage, epoch, sum_loss, N, args, st):
    """
    Build log info for each epoch
    """
    msg = 'epoch:{:02d}\t{} mean loss={}\t elapsed time={} sec.'.format(
        epoch + args.epoch_offset,
        stage,
        sum_loss / N,
        time.time() - st)
    return msg


def main():
    """
    Load data and training model
    """
    args = add_arguments()
    log_fn, result_dir = creat_result_dir(args)

    # creat model and optimizer
    model, optimizer = get_model_optimizer(result_dir, args)

    # load data label
    train_dl, test_dl = load_dataset(args)
    # train_dl = train_dl[0:len(train_dl)/10]
    N = len(train_dl)
    print(N)
    N_test = len(test_dl)

    # Set data transformation
    trans = Transform(padding=[args.crop_pad_inf, args.crop_pad_sup],
                      flip=bool(args.flip),
                      size=args.size,
                      shift=args.shift,
                      lcn=bool(args.lcn),
                      result_dir=result_dir)

    # Add info into log file
    logging.info('# of training data:{}'.format(N))
    logging.info('# of test data:{}'.format(N_test))
    logging.info(time.strftime('%Y-%m-%d_%H-%M-%S'))
    logging.info('start training...')

    # learning loop
    n_epoch = args.epoch
    for epoch in xrange(1, n_epoch + 1):
        # using Multiprocess to speed up and save memory
        # loading data while training
        # load data process
        input_q = Queue()  # original data
        data_q = Queue()  # transformed data
        data_loader = Process(target=load_data,
                              args=('train', trans, args, input_q, data_q))
        data_loader.start()

        # really train
        st = time.time()
        sum_loss = train(train_dl, N, model, optimizer,
                         args, input_q, data_q)

        msg = get_log_msg('training', epoch, sum_loss, N, args, st)
        logging.info(msg)
        print('\n%s' % msg)

        # quit data loading process (all data are loaded in train)
        input_q.put(None)
        data_loader.join()

        # start loading test data
        input_q = Queue()
        data_q = Queue()
        data_loader = Process(target=load_data,
                              args=('test', trans, args, input_q, data_q))
        data_loader.start()
        # validation
        st = time.time()
        sum_loss = test(test_dl, N_test, model, args, input_q, data_q)
        msg = get_log_msg('test', epoch, sum_loss, N_test, args, st)
        logging.info(msg)
        print('\n%s' % msg)

        # quit data loading process
        input_q.put(None)
        data_loader.join()

        if epoch == 1 or epoch % args.snapshot == 0:
            model_fn = '%s/%s_epoch_%d.chainermodel' % (
                result_dir, args.prefix, epoch + args.epoch_offset)
            pickle.dump(model, open(model_fn, 'wb'), -1)
        draw_loss_curve(log_fn, '%s/log.png' % result_dir)

    # store the last epoch
    model_fn = '%s/%s_epoch_%d.chainermodel' % (
                result_dir, args.prefix, epoch + args.epoch_offset)
    pickle.dump(model, open(model_fn, 'wb'), -1)

    logging.info(time.strftime('%Y-%m-%d_%H-%M-%S'))
    logging.info('end training...')

if __name__ == '__main__':
    main()
