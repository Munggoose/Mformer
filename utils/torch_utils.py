import torch
import torch.nn as nn



def dct(x, norm=None):
    """_summary_

    Args:
        x (Tensor): input Tensor
        norm (str, optional): normalization method  in [forward, backward, ortho]
    """

    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1,N)
    v = torch.cat([x[:, ::2],x[:, 1::2].flip([1])],dim=1)
    Vc = torch.fft.rfft( v, 1, onesided=False)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None,:] * torch.pi/ ( 2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:,:,0]*W_r - Vc[:,:,1]* W_i

    if norm == 'ortho':
        V[:,0] /= torch.sqrt(N) * 2
        V[:,1:] /= torch.sqrt(N/2) * 2
    
    V = 2 * V.view(*x_shape)

    return V



def dct1(x):
    """
    Discrete Cosine Transform, Type I
    :param x: the input signal
    :return: the DCT-I of the signal over the last dimension
    """
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])

    return torch.fft.rfft(torch.cat([x, x.flip([1])[:, 1:-1]], dim=1), 1)[:, :, 0].view(*x_shape)