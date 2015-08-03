import chainer.functions as F
from chainer import Variable, FunctionSet


class LinearNet_flic_one_joint(FunctionSet):

    """
    One layer neural networks with batch Normalization
    """

    def __init__(self):
        super(LinearNet_flic_one_joint, self).__init__(
            fc=F.Linear(220 * 220 * 3, 2)  # the normalized picture size
            ) #  the leading dimension is treated as the batch dimension, 
              #  and the other dimensions are reduced to one dimension.

    def forward(self, x_data, y_data, train=True):
        x = Variable(x_data)
        t = Variable(y_data)

        h = self.fc(x)

        # print(h.data.shape)
        # print(t.data.shape)
        loss = F.mean_squared_error(t, h)

        return loss, h
