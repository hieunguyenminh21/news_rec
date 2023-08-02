import torch.nn as nn
import torch
from torch import Tensor
from typing import Tuple


class MaskedBatchNorm1d(nn.Module):
    """
    Layer for batch norm on masked data
    Input: batch_size, num_features (data)     batch_size, num_features (mask)
    Output: batch_size, num_features
    """
    def __init__(
            self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
            affine: bool = True, track_running_stats: bool = True
    ):
        """
        Init method
        :param num_features: number features of input
        :param eps: for numeric stable
        :param momentum: momentum for update mean and var
        :param affine: if True  -> has trainable params
        :param track_running_stats: if True -> using running mean and var
        """
        super(MaskedBatchNorm1d, self).__init__()
        self.num_features: int = num_features
        self.eps: float = eps
        self.momentum: float = momentum
        self.affine: bool = affine
        self.track_running_stats: bool = track_running_stats
        if affine:
            self.weight = nn.Parameter(
                torch.ones(size=(num_features,), dtype=torch.float)
            )
            self.bias = nn.Parameter(
                torch.zeros(size=(num_features,), dtype=torch.float)
            )
        else:
            self.weight = None
            self.bias = None
        if self.track_running_stats:
            self.register_buffer(
                'running_mean',
                torch.zeros(size=(num_features,), dtype=torch.float)
            )
            self.register_buffer(
                'running_var',
                torch.ones(size=(num_features,), dtype=torch.float)
            )
            self.register_buffer(
                'is_first_batch',
                torch.tensor(True, dtype=torch.bool)
            )
        else:
            self.running_mean = None
            self.running_var = None
            self.is_first_batch = None

    def _compute_mean_var(
            self, x: Tensor, attn_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute mean and var
        :param x: batch_size, num_features
        :param attn_mask: batch_size, num_features
        :return:
            - mean: num_features
            - var: num_features
        """
        # Compute masked mean and var
        current_mean = (
               x * attn_mask
        ).sum(dim=0) / (
               attn_mask.sum(dim=0) + 1e-10
        )    # num_features
        current_var = (
              torch.pow(
                  input=x - current_mean, exponent=2
              ) * attn_mask
        ).sum(dim=0) / (
              attn_mask.sum(dim=0) + 1e-10
        )  # num_features
        return current_mean, current_var

    def _update_running_stats(
            self, current_mean: Tensor, current_var: Tensor
    ):
        """
        Update running mean and running var
        :param current_mean: batch_size, num_features
        :param current_var: batch_size, num_features
        :return:
        """
        if not self.training:
            return
        if not self.track_running_stats:
            return
        if self.is_first_batch:
            self.running_mean = current_mean.detach()
            self.running_var = current_var.detach()
            self.is_first_batch = torch.logical_not(self.is_first_batch)
        else:
            self.running_mean = (
                (1 - self.momentum) * self.running_mean
            ) + (
                self.momentum * current_mean.detach()
            )
            self.running_var = (
                (1 - self.momentum) * self.running_var
            ) + (
                self.momentum * current_var.detach()
            )

    def _compute_norm(
            self, x: Tensor,
            current_mean: Tensor, current_var: Tensor
    ) -> Tensor:
        """
        Compute norm
        :param x: batch_size, num_features
        :param current_mean: num_features
        :param current_var: num_features
        :return: batch_size, num_features
        """
        if self.training or not self.track_running_stats:
            normed = (
                 x - current_mean.unsqueeze(dim=0)
            ) / (
                torch.sqrt(
                    current_var.unsqueeze(dim=0) + self.eps
                )
            )    # batch_size, num_features
        else:
            normed = (
                 x - self.running_mean.unsqueeze(dim=0)
            ) / (
                torch.sqrt(
                    self.running_var.unsqueeze(dim=0) + self.eps
                )
            )
        if self.affine:
            normed = normed * self.weight + self.bias
        return normed

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        """
        Compute masked batch norm
        :param x: batch_size, num_features
        :param attn_mask: batch_size, num_features
        :return: batch_size, num_features
        """
        current_mean, current_var = self._compute_mean_var(
            x=x, attn_mask=attn_mask
        )
        self._update_running_stats(
            current_mean=current_mean, current_var=current_var
        )
        normed = self._compute_norm(
            x=x, current_mean=current_mean, current_var=current_var
        )
        normed = normed * attn_mask
        return normed
