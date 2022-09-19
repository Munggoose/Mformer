import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.loss import _Loss
from torch import Tensor
import torch.fft as fft

class FFTLoss(_Loss):
    r"""

    Args:
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Target: :math:`(*)`, same shape as the input.

    Examples::

        >>> loss = nn.MSELoss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    # __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None,seq_l: int=None, reduction: str = 'mean') -> None:
        super(FFTLoss, self).__init__(size_average, reduce, reduction)
        self.seq_l = seq_l

    def forward(self, input: Tensor, target: Tensor,norm: str ='ortho') -> Tensor:
        if self.seq_l == None:
            self.seq_l = input.shape[1]
        
        input = fft.fftn(input.squeeze(),dim=1,norm = norm)
        target = fft.fftn(target.squeeze(),dim=1,norm = norm)

        input_real = input.real
        input_imag = input.imag
        target_real = target.real
        target_imag = target.imag
        loss = F.mse_loss(input_real, target_real, reduction=self.reduction) + F.mse_loss(input_imag, target_imag, reduction=self.reduction)
        return loss