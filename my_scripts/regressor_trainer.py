import numpy as np
import logging
import time
import os
import cv2 as cv
import copy
import cPickle as pickle
from chainer import optimizers, cuda
from multiprocessing import Process, Queue
from progressbar import ProgressBar
from my_scripts.draw_loss import draw_loss_from_list, draw_batch_loss
reload(logging)  # for ipython notebook


class NetworkTrainer(object):
    """Performs SGD on a model"""
    def __init__(self, data_dir=None, data_info_file=None, model=None, gpu=-1,
                 val_rate=0.2, reg=0.0, dropout=1.0, learning_rate=1e-4,
                 momentum=0, update='SGD', weight_decay=0.0001, num_iters=50,
                 sample_batches=True, batch_size=32, num_epochs=100,
                 trans=None, snapshot=5, verbose=True):
        """
        Inputs:
        - data_dir: The directory storing images
        - data_info_file: A csv file storing image name and joints
        - model: A regressor or classifier defined by chainer, can be called:
          loss, prediction = model.forward(X, y, reg, dropout)
          loss can be used to compute gradient.
        - gpu: if >= 0 then use gpu, otherwise use cpu
        - val_rate: The percent of validation data
        - reg: Regularization strength.
        - dropout: Amount of dropout to use
        - learning_rate: Initial learning rate to use
        - momentum: For momentum update
        - update: The update rule to use in chainer.
          One of 'AdaGrad', 'MomentumSGD', 'Adam', 'RMSprop' or 'SGD'
        - weight_decay: weight decay percent for L2 Regularization
        - num_iters: number or iteration in BGD
        - sample_batches: If True, then use stochastic gradient decent;
          otherwise use batch gradient decent.
        - batch_size: The number of training samples to use in each update.
        - num_epochs: The number of epochs to take over the training data.
        - trans: A Transform object to preprocess image data.
        - verbose: If True, print status of each epoch.
        """
        self.params = locals()  # store all the parameters
        [setattr(self, key, value) for key, value in self.params.iteritems()]
        self.optimizer = None  # initialize in self.train()

    def train(self):
        """
        Optimize the parameters of a model to minimize the loss.
        Loss and trained models are stored in local file.
        Returns a tuple of:
        - best_model: The model with the lowest validation error.
        - train_loss_history: A list containing the value of loss
          at each iteration (update) during training.
        - val_loss_history: A list containing the value of loss
          at each epoch based on trained model.
        Others:
        - A logging file to store losses.
        - Some model files to store trained model at each epoch
        """

        # prepare the training and validation datasets
        data_info_file = self.data_info_file
        data_info = np.array([l.strip() for l in open(data_info_file).readlines()])
        # data_info = data_info[0:1000]
        N = len(data_info)
        print (N)
        N_val = int(N * self.val_rate)
        N_train = N - N_val
        perm = np.random.permutation(N)
        train_info = data_info[perm[0: N_train]]
        val_info = data_info[perm[N_train:]]
        print 'Number of training data %d' % N_train
        print 'Number of validation data %d' % N_val
        # create log file and result directory
        log_file, result_dir = self.creat_result_dir_logfile()
        print(self.params)
        logging.info(self.params)  # log the params for training

        # get the optimizer
        self.optimizer = self.get_optimizer()
        lowest_val_loss = float('inf')
        best_model = None
        self.train_loss_batch_history = []
        train_loss_epoch_history = []
        val_loss_epoch_history = []


        # Add info into to log file
        logging.info('# of training data: {}'.format(N_train))
        logging.info('# of validation data: {}'.format(N_val))
        logging.info('# of batch size: {}'.format(self.batch_size))
        logging.info('# of epoch: {}'.format(self.num_epochs))
        logging.info(time.strftime('%Y-%m-%d_%H-%M-%S'))
        logging.info('start training ...')

        for epoch in xrange(1, self.num_epochs + 1):
            # use multiprocess to speed up training and save memory
            # the model can be trained while the batch data is loading
            input_q = Queue()  # store data info for loading and transformation
            data_q = Queue()  # store transformed data

            # create process to load training data
            data_load_process = Process(target=self.data_load,
                                        args=(input_q, data_q, result_dir))
            data_load_process.start()
            start_time = time.time()
            self.compute_loss_model_update(train_info, input_q, data_q)
            input_q.put(None)
            data_load_process.join()

            # each epoch performs an evaluation on subset of training data
            # the same as number of validation datasets
            input_q = Queue()
            data_q = Queue()
            data_load_process = Process(target=self.data_load,
                                        args=(input_q, data_q, result_dir))
            data_load_process.start()
            train_loss_epoch = self.compute_loss(train_info, N_val, input_q, data_q)
            train_loss_epoch_history.append(train_loss_epoch)
            elapsed_time = time.time() - start_time
            log_msg = self.get_log_msg('training', train_loss_epoch,
                                       elapsed_time, epoch)
            logging.info(log_msg)
            print(log_msg)
            input_q.put(None)
            data_load_process.join()

            # each epoch perform an evaluation on validation set
            input_q = Queue()
            data_q = Queue()
            data_load_process = Process(target=self.data_load,
                                        args=(input_q, data_q, result_dir))
            data_load_process.start()
            start_time = time.time()
            val_loss_epoch = self.compute_loss(val_info, N_val, input_q, data_q)
            val_loss_epoch_history.append(val_loss_epoch)
            elapsed_time = time.time() - start_time
            log_msg = self.get_log_msg('validation',
                                       val_loss_epoch, elapsed_time, epoch)
            logging.info(log_msg)
            print(log_msg)
            if val_loss_epoch < lowest_val_loss:
                lowest_val_loss = val_loss_epoch
                best_model = copy.deepcopy(self.model)
            input_q.put(None)
            data_load_process.join()

            # draw loss curve
            draw_batch_loss(self.train_loss_batch_history, result_dir
                            + '/batch_loss.jpg')
            draw_loss_from_list(train_loss_epoch_history,
                                val_loss_epoch_history,
                                result_dir + '/val_loss.jpg')

            # store the trained model
            if epoch == 1 or epoch % self.snapshot == 0:
                model_fn = '%s/epoch_%d.chainermodel' % (result_dir, epoch)
                pickle.dump(self.model, open(model_fn, 'wb'), -1)

        logging.info(time.strftime('%Y-%m-%d_%H-%M-%S'))
        logging.info('End training')

        return (best_model, self.train_loss_batch_history,
                train_loss_epoch_history, val_loss_epoch_history)

    def creat_result_dir_logfile(self):
        """ Create a result directory and a log file """
        # get the class name of the model
        model = self.model
        model_name = model.__class__.__name__

        # create result directory
        result_dir = 'results/' + model_name
        result_dir += '_' + time.strftime('%Y-%m-%d-%H-%M-%S')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        # create log file
        log_file = '%s/log.txt' % result_dir
        open(log_file, 'a').close()
        logging.basicConfig(filename=log_file, level=logging.DEBUG)

        return log_file, result_dir

    def get_optimizer(self):
        """ Load optimizer to update model """
        update = self.update
        learning_rate = self.learning_rate
        momentum = self.momentum

        if update == 'AdaGrad':
            optimizer = optimizers.AdaGrad(lr=learning_rate)
        elif update == 'MomentumSGD':
            optimizer = optimizers.MomentumSGD(lr=learning_rate, momentum=momentum)
        elif update == 'Adam':
            optimizer = optimizers.Adam()
        elif update == 'RMSprop':
            optimizer = optimizers.RMSprop(lr=learning_rate)
        elif update == 'SGD':
            optimizer = optimizers.SGD(lr=learning_rate)
        else:
            raise Exception('Unrecognized update type "%s"' % update)
        optimizer.setup(self.model.collect_parameters())
        return optimizer

    def data_load(self, input_q, data_q, result_dir):
        """
        Used for multiprocess to load and transform data batch by batch
        """
        batch_size = self.batch_size
        trans = self.trans
        data_dir = self.data_dir

        c = trans.channel  # channel of image
        s = trans.size
        d = trans.num_joints  # number of joints

        while True:
            x_batch = input_q.get()
            # x_batch is a string containing image name and joints
            if x_batch is None:
                break

            input_data = np.zeros((batch_size, c, s, s))
            label = np.zeros((batch_size, d * 2))

            # load and transform image
            for i, x in enumerate(x_batch):
                x = x.strip().split(',')
                img_file_name = '%s/images/%s' % (data_dir, x[0])
                joints = np.array([int(float(p)) for p in x[1:]])
                x_data = cv.imread(img_file_name)
                x_data_trans, joints_trans = trans.transform(img_file_name, x_data,
                                                             joints, result_dir)
                input_data[i] = x_data_trans.transpose((2, 0, 1))
                label[i] = joints_trans

            # add to data_q
            data_q.put([input_data, label])

    def compute_loss_model_update(self, train_info,
                                  input_q, data_q):
        """
        Compute loss and perform model update
        Store the batch loss in self.train_loss_batch_history
        """
        N = train_info.shape[0]
        pbar = ProgressBar(N)

        # get some variables
        batch_size = self.batch_size

        if self.sample_batches:
            num_batch = int(np.ceil(N / batch_size))
            # put data info into input_q by batch
            # notice at the same time, data are loaded and transformed
            for i in xrange(num_batch):
                mask = np.random.choice(N, batch_size, replace=False)
                x_batch = train_info[mask]
                input_q.put(x_batch)
        else:  # full gradient descent
            input_q.put(train_info)
            num_batch = 1

        # training with batch data
        for i in xrange(num_batch):
            input_data, label = data_q.get()

            if self.gpu >= 0:  # whether to use gpu
                input_data = cuda.to_gpu(input_data.astype(np.float32))
                label = cuda.to_gpu(label.astype(np.float32))

            self.optimizer.zero_grads()  # initial gradients to zero
            loss, pred = self.model.forward(input_data, label, train=True)
            loss.backward()  # compute gradient
            self.optimizer.weight_decay(self.weight_decay)
            self.optimizer.update()
            self.train_loss_batch_history.append(float(cuda.to_cpu(loss.data)))
            update_size = (i + 1) * batch_size
            pbar.update(update_size if update_size <= N else N)
        pbar.finish()

    def get_log_msg(self, stage, mean_loss, elapsed_time, epoch):
        """
        Build log message each epoch for training and validation
        """
        msg = 'epoch{:02d}\t{} mean loss ={}\t elapsed time={} sec.'.format(
            epoch, stage, mean_loss, elapsed_time)
        return msg

    def compute_loss(self, data_info, N_val, input_q, data_q):
        N = data_info.shape[0]
        if N > N_val:  # subset the datasets
            mask = np.random.choice(N, N_val, replace=False)
            data_info = data_info[mask]
            N = N_val

        pbar = ProgressBar(N)
        batch_size = self.batch_size
        if self.sample_batches:  # d
            num_batch = int(np.ceil(N / batch_size))
            # put data info into input_q by batch
            # notice at the same time, data are loaded and transformed
            for i in xrange(num_batch):
                mask = np.random.choice(N, batch_size, replace=False)
                x_batch = data_info[mask]
                input_q.put(x_batch)
        else:  # full gradient descent
            input_q.put(data_info)
            num_batch = 1

        sum_loss = 0
        # training with batch data
        for i in xrange(num_batch):
            input_data, label = data_q.get()

            if self.gpu >= 0:  # whether to use gpu
                input_data = cuda.to_gpu(input_data.astype(np.float32))
                label = cuda.to_gpu(label.astype(np.float32))

            loss, pred = self.model.forward(input_data, label, train=False)
            sum_loss += float(cuda.to_cpu(loss.data)) * batch_size
            update_size = (i + 1) * batch_size
            pbar.update(update_size if update_size <= N else N)
        pbar.finish()

        mean_loss = sum_loss / N
        return mean_loss

    def train_bgd(self):
        """
        Optimize the parameters of a model use batch gradient decent.
        Loss and trained models are stored in local file.
        Returns a tuple of:
        - best_model: The model with the lowest validation error.
        - train_loss_history: A list containing the value of loss
          at each iteration (update) during training.
        - val_loss_history: A list containing the value of loss
          at each epoch based on trained model.
        Others:
        - A logging file to store losses.
        - Some model files to store trained model at each epoch
        """
        # prepare the training and validation datasets
        data_info_file = self.data_info_file
        data_info = np.array([l.strip() for l in open(data_info_file).readlines()])
        # data_info = data_info[0:1000]
        N = len(data_info)
        print (N)
        N_val = int(N * self.val_rate)
        N_train = N - N_val
        perm = np.random.permutation(N)
        train_info = data_info[perm[0: N_train]]
        val_info = data_info[perm[N_train:]]
        print 'Number of training data %d' % N_train
        print 'Number of validation data %d' % N_val
        # create log file and result directory
        log_file, result_dir = self.creat_result_dir_logfile()
        print(self.params)
        logging.info(self.params)  # log the params for training

        # get the optimizer
        self.optimizer = self.get_optimizer()
        lowest_val_loss = float('inf')
        best_model = None
        self.train_loss_batch_history = []
        train_loss_epoch_history = []
        val_loss_epoch_history = []

        # Add info into to log file
        logging.info('Use batch gradient decent!')
        logging.info('# of training data: {}'.format(N_train))
        logging.info('# of validation data: {}'.format(N_val))
        logging.info('# of batch gradient decent')
        logging.info('# of iteration: {}'.format(self.num_epochs))
        logging.info(time.strftime('%Y-%m-%d_%H-%M-%S'))
        logging.info('start training ...')

        for epoch in xrange(1, self.num_epochs + 1):
            start_time = time.time()
            input_data, label = self.all_data_load(train_info, result_dir)
            self.compute_loss_model_update_bgd(input_data, label, self.num_iters)

            # each epoch performs an evaluation on subset of training data
            # the same as number of validation datasets
            mask = np.random.choice(N_train, N_val, replace=False)
            train_test_data = input_data[mask]
            label_test = label[mask]
            train_loss_epoch = self.compute_loss_bgd(train_test_data,
                                                     label_test,
                                                     self.num_iters)
            train_loss_epoch_history.append(train_loss_epoch)
            elapsed_time = time.time() - start_time
            log_msg = self.get_log_msg('training', train_loss_epoch,
                                       elapsed_time, epoch)
            logging.info(log_msg)
            print(log_msg)

            # each epoch perform an evaluation on validation set
            start_time = time.time()
            input_data, label = self.all_data_load(val_info, result_dir)
            val_loss_epoch = self.compute_loss_bgd(input_data, label, self.num_iters)
            val_loss_epoch_history.append(val_loss_epoch)
            elapsed_time = time.time() - start_time
            log_msg = self.get_log_msg('validation', val_loss_epoch,
                                       elapsed_time, epoch)
            logging.info(log_msg)
            print(log_msg)

            if val_loss_epoch < lowest_val_loss:
                lowest_val_loss = val_loss_epoch
                best_model = copy.deepcopy(self.model)

            # draw loss curve
            draw_batch_loss(self.train_loss_batch_history,
                            result_dir + '/batch_loss.jpg')
            draw_loss_from_list(train_loss_epoch_history,
                                val_loss_epoch_history,
                                result_dir + '/loss.jpg')

            # store the trained model
            if epoch == 1 or epoch % self.snapshot == 0:
                model_fn = '%s/epoch_%d.chainermodel' % (result_dir, epoch)
                pickle.dump(self.model, open(model_fn, 'wb'), -1)

        logging.info(time.strftime('%Y-%m-%d_%H-%M-%S'))
        logging.info('End training')

        return (best_model, self.train_loss_batch_history,
                train_loss_epoch_history, val_loss_epoch_history)

    def all_data_load(self, data_info, result_dir):
        """
        Load all the image data.
        """
        trans = self.trans
        data_dir = self.data_dir

        c = trans.channel  # channel of image
        s = trans.size
        d = trans.num_joints  # number of joints

        N = data_info.shape[0]
        input_data = np.zeros((N, c, s, s))
        label = np.zeros((N, d * 2))

        pbar = ProgressBar(N)
        # load and transform image
        for i, x in enumerate(data_info):
            x = x.strip().split(',')
            img_file_name = '%s/images/%s' % (data_dir, x[0])
            joints = np.array([int(float(p)) for p in x[1:]])
            x_data = cv.imread(img_file_name)
            x_data_trans, joints_trans = trans.transform(img_file_name, x_data,
                                                         joints, result_dir)
            input_data[i] = x_data_trans.transpose((2, 0, 1))
            label[i] = joints_trans
            pbar.update(i)
        pbar.finish()

        return input_data, label

    def compute_loss_model_update_bgd(self, input_data, label, num_iters):
        """
        Update optimizer using batch gradient decent.
        """
        pbar = ProgressBar(num_iters)
        for it in xrange(num_iters):
            self.optimizer.zero_grads()  # initial gradients to zero
            loss, pred = self.model.forward(input_data, label, train=True)
            loss.backward()  # compute gradient
            self.optimizer.weight_decay(self.weight_decay)  # decay gradient
            self.train_loss_batch_history.append(float(cuda.to_cpu(loss.data)))
            self.optimizer.update()
            pbar.update(it)
        pbar.finish()

    def compute_loss_bgd(self, input_data, label, num_iters):
        sum_loss = 0
        for it in xrange(num_iters):
            loss, pred = self.model.forward(input_data, label, train=True)
            sum_loss += float(cuda.to_cpu(loss.data))
        mean_loss = sum_loss / num_iters
        return mean_loss

