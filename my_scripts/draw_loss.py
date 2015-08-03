import re
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from ggplot import *


def draw_loss_curve(logfile, outfile):
    """
    Drow loss curve based on logfile
    """
    train_loss = []
    val_loss = []
    for line in open(logfile, 'r'):
        line = line.strip()

        if not ('root:epoch' in line):
            continue
        if 'training' in line and 'inf' not in line:
            tr_loss = float(re.search(ur'loss =([0-9\.]+)', line).groups()[0])
            train_loss.append(tr_loss)
        if 'validation' in line and 'inf' not in line:
            v_loss = float(re.search(ur'loss =([0-9\.]+)', line).groups()[0])
            val_loss.append(v_loss)
            print tr_loss, v_loss

    if not len(train_loss) > 1:
        return
    draw_loss_from_list(train_loss, val_loss, outfile)


def draw_loss_from_list(train_loss_history, val_loss_history, outfile):
    """
    Draw loss curve based on train and validation loss.
    """
    # build loss dictionary
    loss_dict = {'epoch': range(1, len(train_loss_history) + 1),
                 'train loss': train_loss_history,
                 'validation loss': val_loss_history}

    df = pd.DataFrame(loss_dict)
    # long version
    long_df = pd.melt(df, id_vars='epoch',
                      value_vars=['train loss', 'validation loss'],
                      var_name='type', value_name='loss')
    gg = ggplot(long_df, aes('epoch', 'loss', color='type')) + geom_line()
    ggsave(outfile, gg)


def draw_batch_loss(loss_batch_history, outfile):
    """
    Draw loss based on batch loss history.
    """
    loss_dict = {'iteration': range(1, len(loss_batch_history) + 1),
                 'batch loss': loss_batch_history}
    df = pd.DataFrame(loss_dict)
    gg = ggplot(df, aes('iteration', 'batch loss')) + geom_line()
    ggsave(outfile, gg)

if __name__ == '__main__':
    log_file = 'results/LinearNet_flic_2015-07-27-12-14-37/log.txt'
    output_file = 'results/LinearNet_flic_2015-07-27-12-14-37/log_loss.jpg'
    draw_loss_curve(log_file, output_file)


