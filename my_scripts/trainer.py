import numpy as np
import time
import logging
import os
import copy
import cPickle as pickle
from chainer import optimizers, cuda
from progressbar import ProgressBar
from my_scripts.draw_loss import draw_loss_from_list, draw_batch_loss


class Trainer(object):
    """
    Perform GD on a model, no need for data preprocessing
    """
    def __init__(self, model=None, gpu=-1, val_rate=0.2, reg=0.0, dropout=1.0,
                 learning_rate=1e-4, momentum=0, update='SGD',
                 weight_decay=0.0001, num_iters=50,  sample_batches=True,
                 batch_size=32, num_epochs=100, snapshot=5):
        self.params = locals()  # store all the parameters
        [setattr(self, key, value) for key, value in self.params.iteritems()]
        self.optimizer = None  # initialize in self.train()

    def train(self, X, y):
        """
        Optimize the parameters of a model to minimize the loss.
        Loss and trained models are stored in local file.
        Returns a tuple of:
        - best_model: The model with the lowest validation error.
        - train_loss_batch_history: A list all batch loss in all epochs
        - train_loss_epoch_history: A list containing the value of loss
          at each epoch (update) during training.
        - val_loss_epoch_history: A list containing the value of loss
          at each epoch based on trained model.
        Others:
        - A logging file to store losses.
        - Some model files to store trained model at each epoch
        """
        # prepare the training and validation datasets
        # chainer only support np.float32
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        if y.dtype != np.float32:
            y = y.astype(np.float32)

        N = X.shape[0]
        N_val = int(N * self.val_rate)
        N_train = N - N_val
        perm = np.random.permutation(N)
        X_train = X[perm[0: N_train]]
        y_train = y[perm[0: N_train]]
        X_val = X[perm[N_train:]]
        y_val = y[perm[N_train:]]
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
            start_time = time.time()
            # update model
            self.compute_loss_model_update(X_train, y_train)

            # each epoch performs an evaluation on subset of training data
            # the same as number of validation datasets
            mask = np.random.choice(N_train, N_val, replace=False)
            train_test_data = X[mask]
            label_test = y[mask]
            # train_test_data = X_train
            # label_test = y_train
            train_loss_epoch = self.compute_loss(train_test_data, label_test)
            train_loss_epoch_history.append(train_loss_epoch)
            elapsed_time = time.time() - start_time
            log_msg = self.get_log_msg('training', train_loss_epoch,
                                       elapsed_time, epoch)
            logging.info(log_msg)
            print(log_msg)

            # each epoch perform an evaluation on validation set
            start_time = time.time()
            val_loss_epoch = self.compute_loss(X_val, y_val)
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

    def compute_loss_model_update(self, X, y):
        """
        Use BGD to compute loss and update model
        """
        N = X.shape[0]
        pbar = ProgressBar(N)
        batch_size = self.batch_size
        num_batch = int(np.ceil(N / batch_size))
        # put data info into input_q by batch
        # notice at the same time, data are loaded and transformed
        for i in xrange(num_batch):
            mask = np.random.choice(N, batch_size, replace=False)
            x_batch = X[mask]
            y_batch = y[mask]
            if self.gpu >= 0:  # whether to use gpu
                x_batch = cuda.to_gpu(x_batch.astype(np.float32))
                y_batch = cuda.to_gpu(y_batch.astype(np.float32))

            self.optimizer.zero_grads()
            loss, pred = self.model.forward(x_batch, y_batch, train=True)
            loss.backward()  # compute gradient
            self.optimizer.weight_decay(self.weight_decay)
            self.optimizer.update()
            self.train_loss_batch_history.append(float(cuda.to_cpu(loss.data)))
            update_size = (i + 1) * batch_size
            pbar.update(update_size if update_size <= N else N)
        pbar.finish()        # get some variables

    def compute_loss(self, X, y):
        loss, pred = self.model.forward(X, y, train=False)
        return float(cuda.to_cpu(loss.data))

    def creat_result_dir_logfile(self):
        """ Create a result directory and a log file """
        # get the class name of the model
        model = self.model
        model_name = model.__class__.__name__

        # create result directory
        result_dir = 'results/' + model_name
        print result_dir
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

    def get_log_msg(self, stage, mean_loss, elapsed_time, epoch):
        """
        Build log message each epoch for training and validation
        """
        msg = 'epoch{:02d}\t{} mean loss ={}\t elapsed time={} sec.'.format(
            epoch, stage, mean_loss, elapsed_time)
        return msg
