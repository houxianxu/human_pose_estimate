import chainer.functions as F
from chainer import Variable, FunctionSet


class LinearNet_flic_sift(FunctionSet):

    """
    One layer neural networks with batch Normalization
    """

    def __init__(self, size=200, num_joints=7):
        """
        - size: Input image is size of size x size
        - num_joints: number of joints to predict
        """
        super(LinearNet_flic_sift, self).__init__(
            fc=F.Linear(size, num_joints * 2)  # the normalized picture size
            ) #  the leading dimension is treated as the batch dimension, 
              #  and the other dimensions are reduced to one dimension.

    def forward(self, x_data, y_data, train=True):
        x = Variable(x_data)
        t = Variable(y_data)

        h = self.fc(x)
        loss = F.mean_squared_error(t, h)
        return loss, h
