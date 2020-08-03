import torch


# Round to nearest even
def round_to_nearest_inplace(x, N):
    """ In-place implementation of the round-to-nearest-even quantization method

        Args:
            x: input Tensor
            N: number of bits of the fractional part
        Returns:
            x: quantized Tensor
    """
    x.mul_(2 ** N).floor_().mul_(2 ** float(-N))


class ClassRoundToNearest(torch.autograd.Function):
    """ Implementation of the round-to-nearest-even quantization method

        The class implements the functions to use in the forward and backward pass. For the gradient computation
        in the backward pass, the quantization operation is skipped.
    """

    @staticmethod
    def forward(ctx, input, N):
        ctx.N = N
        return (input * 2 ** N + 2 ** float(-N - 1)).floor() * 2 ** float(-N)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def round_to_nearest(x, N):
    """ Function that applies the round-to-nearest-even class

        Args:
            x: input Tensor
            N: number of bits of the fractional part
        Returns:
            x: quantized Tensor
    """
    f = ClassRoundToNearest.apply
    x = f(x, N)
    return x


# stochastic rounding
def stochastic_rounding_inplace(x, N):
    """ In-place implementation of the stochastic rounding quantization method

        Args:
            x: input Tensor
            N: number of bits of the fractional part
        Returns:
            x: quantized Tensor
    """
    input_old = x.clone()
    eps = 2 ** (-float(N))
    p = torch.rand_like(x)
    x.mul_(2 ** N).add_(2 ** float(-N - 1)).floor_().mul_(2 ** float(-N))
    x[p < ((input_old - x) / eps)] += eps


class ClassStochasticRounding(torch.autograd.Function):
    """ Implementation of the stochastic rounding quantization method

        The class implements the functions to use in the forward and backward pass. For the gradient computation
        in the backward pass, the quantization operation is skipped.
    """

    @staticmethod
    def forward(ctx, input, N):
        eps = 2 ** float(-N)
        p = torch.rand_like(input)
        output = (input * 2 ** N + 2 ** float(-N - 1)).floor() * float(2 ** float(-N))
        output[p < ((input - output) / eps)] = output[p < ((input - output) / eps)] + eps
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def stochastic_rounding(x, N):
    """ Function that applies the stochastic rounding class

        Args:
            x: input Tensor
            N: number of bits of the fractional part
        Returns:
            x: quantized Tensor
    """
    f = ClassStochasticRounding.apply
    x = f(x, N)
    return x


# Logarithmic rounding
def logarithmic_inplace(x, N):
    """ In-place implementation of the logarithmic quantization method

        Args:
            x: input Tensor
            N: number of bits of the fractional part
        Returns:
            x: quantized Tensor
    """
    sign = torch.sign(x)
    exponent = x.abs().add(1e-8).log2().round()
    x.fill_(2.).pow_(exponent)
    x[x < (2 ** float(-N))] *= 0
    x.clamp_(0, 2 ** 10).mul_(sign)


class ClassLogarithmic(torch.autograd.Function):
    """ Implementation of the logarithmic quantization method

        The class implements the functions to use in the forward and backward pass. For the gradient computation
        in the backward pass, the quantization operation is skipped.
    """

    @staticmethod
    def forward(ctx, input, N):
        sign = torch.sign(input)
        output = input.abs().add(1e-8).log2().round()
        output = 2 ** output
        output[output < (2 ** float(-N))] = 0
        output[output > 2 ** 10] = 2 ** 10
        output = output * sign
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def logarithmic(x, N):
    """ Function that applies the logarithmic class

        Args:
            x: input Tensor
            N: number of bits of the fractional part
        Returns:
            x: quantized Tensor
    """
    f = ClassLogarithmic.apply
    x = f(x, N)
    return x


# Truncation
def truncation_inplace(x, N):
    """ In-place implementation of the truncation quantization method

        Args:
            x: input Tensor
            N: number of bits of the fractional part
        Returns:
            x: quantized Tensor
    """
    x.mul_(2 ** N).floor_().mul_(2 ** float(-N))


class ClassTruncation(torch.autograd.Function):
    """ Implementation of the truncation quantization method

        The class implements the functions to use in the forward and backward pass. For the gradient computation
        in the backward pass, the quantization operation is skipped.
    """

    @staticmethod
    def forward(ctx, input, N):
        output = (input * 2 ** N).floor() * 2 ** float(-N)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def truncation(x, N):
    """ Function that applies the truncation class

        Args:
            x: input Tensor
            N: number of bits of the fractional part
        Returns:
            x: quantized Tensor
    """
    f = ClassTruncation.apply
    x = f(x, N)
    return x
