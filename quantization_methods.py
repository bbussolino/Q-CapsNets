from numpy import dtype
import torch


# Round to nearest even
def round_to_nearest_inplace(x, s, n):
    """ In-place implementation of the round-to-nearest-even quantization method

        Args:
            x: input Tensor
            N: number of bits of the fractional part
        Returns:
            x: quantized Tensor
    """
    assert n > 0
    dequant_scale = s / 2**(n-1)   #  s / 2^{n-1}
    quant_scale = 1 / dequant_scale  # 2^{n-1} / s
    x.mul_(quant_scale).round_().clamp_(min=-2**(n-1), max=2**(n-1)-1).mul_(dequant_scale)


# round to nearest even 
class ClassRoundToNearest(torch.autograd.Function):
    """ Implementation of the round-to-nearest-even quantization method

        The class implements the functions to use in the forward and backward pass. For the gradient computation
        in the backward pass, the quantization operation is skipped.
    """

    @staticmethod
    def forward(ctx, input, s, n):
        assert n > 0
        dequant_scale = s / 2**(n-1)   #  s / 2^{n-1}
        quant_scale = 1 / dequant_scale  # 2^{n-1} / s
        output = (input * quant_scale).round().clamp(min=-2**(n-1), max=2**(n-1)-1) * dequant_scale
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def round_to_nearest(x, s, n):
    """ Function that applies the round-to-nearest-even class

        Args:
            x: input Tensor
            N: number of bits of the fractional part
        Returns:
            x: quantized Tensor
    """
    f = ClassRoundToNearest.apply
    x = f(x, s, n)
    return x


# stochastic rounding
def stochastic_rounding_inplace(x, s, n):
    """ In-place implementation of the stochastic rounding quantization method

        Args:
            x: input Tensor
            N: number of bits of the fractional part
        Returns:
            x: quantized Tensor
    """
    assert n > 0
    device = x.device 
    dtype = x.dtype 
    dequant_scale = s / 2**(n-1)   #  s / 2^{n-1}
    quant_scale = 1 / dequant_scale  # 2^{n-1} / s 
    
    #scaled_in = (input * quant_scale)
    x.mul_(quant_scale)
    round_temp = x.floor().clamp(min=-2**(n-1), max=2**(n-1)-1)
    prob = torch.abs(x-round_temp)
    rand_num = torch.rand_like(x)
    round_decision = torch.where(prob <= rand_num, 
                                    torch.tensor(0., dtype=dtype, device=device), 
                                    torch.tensor(1., dtype=dtype, device=device)) #* torch.sign(x) 
    x.floor_().add_(round_decision).clamp_(min=-2**(n-1), max=2**(n-1)-1).mul_(dequant_scale)


class ClassStochasticRounding(torch.autograd.Function):
    """ Implementation of the stochastic rounding quantization method

        The class implements the functions to use in the forward and backward pass. For the gradient computation
        in the backward pass, the quantization operation is skipped.
    """

    @staticmethod
    def forward(ctx, input, s, n):
        assert n > 0
        device = input.device 
        dtype = input.dtype 
        dequant_scale = s / 2**(n-1)   #  s / 2^{n-1}
        quant_scale = 1 / dequant_scale  # 2^{n-1} / s 
        
        scaled_in = (input * quant_scale)
        round_temp = scaled_in.floor()
        prob = torch.abs(scaled_in-round_temp)
        rand_num = torch.rand_like(input)
        round_decision = torch.where(prob <= rand_num, 
                                     torch.tensor(0., dtype=dtype, device=device), 
                                     torch.tensor(1., dtype=dtype, device=device)) #* torch.sign(input) 
        output = (round_temp + round_decision).clamp(min=-2**(n-1), max=2**(n-1)-1) * dequant_scale
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def stochastic_rounding(x, s, n):
    """ Function that applies the stochastic rounding class

        Args:
            x: input Tensor
            N: number of bits of the fractional part
        Returns:
            x: quantized Tensor
    """
    f = ClassStochasticRounding.apply
    x = f(x, s, n)
    return x


# Truncation
def truncation_inplace(x, s, n):
    """ In-place implementation of the truncation quantization method

        Args:
            x: input Tensor
            N: number of bits of the fractional part
        Returns:
            x: quantized Tensor
    """
    assert n > 0
    dequant_scale = s / 2**(n-1)   #  s / 2^{n-1}
    quant_scale = 1 / dequant_scale  # 2^{n-1} / s
    x.mul_(quant_scale).floor_().clamp_(min=-2**(n-1), max=2**(n-1)-1).mul_(dequant_scale)


class ClassTruncation(torch.autograd.Function):
    """ Implementation of the truncation quantization method

        The class implements the functions to use in the forward and backward pass. For the gradient computation
        in the backward pass, the quantization operation is skipped.
    """

    @staticmethod
    def forward(ctx, input, s, n):
        assert n > 0
        dequant_scale = s / 2**(n-1)   #  s / 2^{n-1}
        quant_scale = 1 / dequant_scale  # 2^{n-1} / s
        output = (input * quant_scale).floor().clamp(min=-2**(n-1), max=2**(n-1)-1) * dequant_scale
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def truncation(x, s, n):
    """ Function that applies the truncation class

        Args:
            x: input Tensor
            N: number of bits of the fractional part
        Returns:
            x: quantized Tensor
    """
    f = ClassTruncation.apply
    x = f(x, s, n)
    return x
