import torch.nn as nn
from torch import Tensor
import torch


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def compute_loss(self, x: Tensor, y_t: Tensor) -> Tensor:
        '''

        :param x: batch_size, in_features
        :param y_t: in_features, batch_size
        :return:
        '''

        correlate = torch.matmul(x, y_t)  # batch_size, batch_size
        correlate = torch.exp(correlate)

        positive = torch.diag(correlate)  # batch_size
        total = torch.sum(correlate, dim=1, keepdim=False)  # batch_size

        ratio = positive / total    # batch_size
        ratio = torch.log(ratio)    # batch_size

        inv_loss = torch.sum(ratio)     # scalar
        loss = -inv_loss    # scalar

        return loss

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        '''

        :param x: batch_size, in_features
        :param y: batch_size, in_features
        :return:
        '''

        x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-8)
        y_norm = y / (y.norm(dim=1, keepdim=True) + 1e-8)

        x_norm_T = torch.transpose(x_norm, dim0=0, dim1=1)  # dim, batch
        y_norm_T = torch.transpose(y_norm, dim0=0, dim1=1)  # dim, batch

        loss_1 = self.compute_loss(x=x_norm, y_t=x_norm_T)
        loss_2 = self.compute_loss(x=x_norm, y_t=y_norm_T)
        loss_3 = self.compute_loss(x=y_norm, y_t=y_norm_T)

        return loss_1 + loss_2 + loss_3
