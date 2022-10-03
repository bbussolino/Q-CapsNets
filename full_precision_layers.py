import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from characterization_utils import CharacterizationUtils


class Conv2d_ReLU(nn.Module, CharacterizationUtils):
    """ Convolutional layer followed by ReLU activation

        Methods:
            forward: compute the output of the module given the input

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        """ Parameters:
                in_channels: number of input channels
                out_channels: number of output channels
                kernel_size: kernel size
                stride: stride
        """
        super(Conv2d_ReLU, self).__init__()
        CharacterizationUtils.__init__(self)
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride)
        self.relu = nn.ReLU()
        self.leaf = True

    def forward(self, x):
        """ forward method

            Args:
                x: input Tensor of shape [batch_size, in_channels, hi, wi]
            Returns:
                out_conv: output Tensor of shape [batch_size, out_channels, ho, wo]
        """
        if CharacterizationUtils.characterize: 
            self.update_max(x, "input")
            self.update_sqnr(x, "input")
        out_conv = self.conv(x)  # Convolution
        out_conv = self.relu(out_conv)  # Activation
        
        if CharacterizationUtils.characterize: 
            self.update_max(out_conv, "output")
            self.update_sqnr(out_conv, "output")

        return out_conv


class Conv2d_BN_ReLU(nn.Module, CharacterizationUtils):
    """ Convolutional layer followed by Batch Normalization and ReLU activation

        Methods:
            forward: compute the output of the module given the input

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1, momentum=0.99, eps=0.001):
        """ Parameters:
                in_channels: number of input channels
                out_channels: number of output channels
                kernel_size: kernel size
                stride: stride
                padding: padding for the convolution (default = 1)
                momentum: momentum for the Batch Normalization (default = 0.99)
                eps: value for numerical stability of the Batch Normalization (default = 0.001)
        """
        super(Conv2d_BN_ReLU, self).__init__()
        CharacterizationUtils.__init__(self)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels, momentum=momentum, eps=eps)
        self.relu = nn.ReLU()
        self.leaf = True

    def forward(self, x):
        """ forward method

            Args:
                x: input Tensor of shape [batch_size, in_channels, hi, wi]
            Returns:
                out_conv: output Tensor of shape [batch_size, out_channels, ho, wo]
        """
        if CharacterizationUtils.characterize: 
            self.update_max(x, "input")
            self.update_sqnr(x, "input")
        out_conv = self.conv(x)
        out_conv = self.relu(out_conv)
        out_conv = self.batchnorm(out_conv)
        if CharacterizationUtils.characterize: 
            self.update_max(out_conv, "output")
            self.update_sqnr(out_conv, "output")

        return out_conv

class DynamicRouting(CharacterizationUtils):
    def squash(self, x, dim=2):
        """ Computes the squash function of a Tensor along dimension dim

        Args:
            x: input tensor
            dim: dimension along which the squash must be performed (default = 2)

        Returns:
            A tensor squashed along dimension dim of the same shape of x """
        norm = torch.norm(x, dim=dim, keepdim=True)
        return x * norm / (1 + norm ** 2)


    def update_routing(self, votes, logits, iterations, bias):
        """ Dynamic routing algorithm   (paper: Dynamic Routing Between Capsules, Sabour et al., 2017)

        Args:
            votes  :  hat{u_{j|i}}   Tensor  [bs, ci, co, no] / [bs, ci, co, no, ho, wo]
            logits  :  b_{ij}        Tensor  [bs, ci, co]     / [bs, ci, co, ho, wo]
            iterations  :  Number of iterations of the algorithm
            bias  :  Bias term       Tensor  [co, no] / [co, no, 1, 1]

        Returns:
            activation  :  v_j       Tensor  [bs, co, no] / [bs, co, no, ho, wo]

        Other variables of the algorithm:
            votes_trans  :  equivalent to votes, but with shape [no, bs, ci, co] / [no, bs, ci, co, ho, wo]
            route  :  c_{ij}         Tensor  [bs, ci, co] / [bs, ci, co, ho, wo]
            preactivate_unrolled  :  c_{ij} * hat{u_{j|i}}  Tensor  [no, bs, ci, co] / [no, bs, ci, co, ho, wo]
            preactivate_trans  :  equivalent to preactivate_unrolled
                                    but with shape [bs, ci, co, no] / [bs, ci, co, no, ho, wo]
            preactivate  :  s_j      Tensor  [bs, co, no] / [bs, co, no, ho, wo]
            act_3d  :  equivalent to activation, but with shape [bs, 1, co, no] / [bs, 1, co, no, ho, wo]
            distances  :  \hat{u_{j|i}} \cdot v_j   Tensor [bs, ci, co] / [bs, ci, co, ho, wo]

        Meaning of the dimensions:
            bs: batch size
            ci: number of input channels / number of input capsules
            co: number of output channels / number of output capsules
            ni: dimension of input capsules
            no: dimension of output capsules
            ho/wo: height/width of the output feature maps if the capsule layer is convolutional
            """
        # Raise an error if the number of iterations is lower than 1
        if iterations < 1:
            raise ValueError('The number of iterations must be greater or equal than 1')

        # Perform different permutations depending on the number of dimensions of the vector (4 or 6)
        dimensions = len(votes.size())
        if dimensions == 4:  # [bs, ci, co, no]
            votes_trans = votes.permute(3, 0, 1, 2).contiguous()  # [no, bs, ci, co]

        else:  # [bs, ci, co, no, ho, wo]
            votes_trans = votes.permute(3, 0, 1, 2, 4, 5).contiguous()  # [no, bs, ci, co, ho, wo]
        
        for iteration in range(iterations):
            route = F.softmax(logits, dim=2)
            if CharacterizationUtils.characterize: 
                self.update_max(route, f"post_softmax_{iteration}")
                self.update_sqnr(route, f"post_softmax_{iteration}")
            
            preactivate_unrolled = route * votes_trans
            if dimensions == 4:
                preactivate_trans = preactivate_unrolled.permute(1, 2, 3, 0).contiguous()  # bs, ci, co, no
            else:
                preactivate_trans = preactivate_unrolled.permute(1, 2, 3, 0, 4, 5).contiguous()  # bs, ci, co, no, ho, wo

            preactivate = preactivate_trans.sum(dim=1) + bias  # bs, co, no, (ho, wo)
            if CharacterizationUtils.characterize: 
                self.update_max(preactivate, f"pre_squash_{iteration}")
                self.update_sqnr(preactivate, f"pre_squash_{iteration}")
                
            activation = self.squash(preactivate, dim=2)  # bs, co, no, (ho, wo)
            if CharacterizationUtils.characterize: 
                self.update_max(activation, f"output_{iteration}")
                self.update_sqnr(activation, f"output_{iteration}")

            act_3d = activation.unsqueeze(1)  # bs, 1, co, no, (ho,wo)
            distances = (votes * act_3d).sum(dim=3)  # bs, ci, co, ho, wo
            logits = logits + distances
            if CharacterizationUtils.characterize: 
                self.update_max(logits, f"pre_softmax_{iteration}")
                self.update_sqnr(logits, f"pre_softmax_{iteration}")
                
        return activation


    def update_routing_6D_DeepCaps(self, votes, logits, iterations, bias):  # differs from above
        """ Dynamic routing algorithm used in DeepCaps for the convolutional capsule layers

        Args:
            votes  :  hat{u_{j|i}}  Tensor  [bs, ci, co, no, ho, wo]
            logits  :  b_{ij}        Tensor  [bs, ci, co, ho, wo]
            iterations  :  Number of iterations of the algorithm
            bias  :  Bias term       Tensor  [bs, co, no, 1, 1]

        Returns:
            activation  :  v_j       Tensor  [bs, co, no, ho, wo]

        Meaning of the dimensions:
            bs: batch size
            ci: number of input channels / number of input capsules
            co: number of output channels / number of output capsules
            ni: dimension of input capsules
            no: dimension of output capsules
            ho/wo: height/width of the output feature maps if the capsule layer is convolutional
            """
        # Raise an error if the number of iterations is lower than 1
        if iterations < 1:
            raise ValueError('The number of iterations must be greater or equal than 1')

        bs, ci, co, no, ho, wo = votes.size()

        # Perform different permutations depending on the number of dimensions of the vector (4 or 6)
        for iteration in range(iterations):
            logits_temp = logits.view(bs, ci, -1)  # bs, ci, co*ho*wo
            route_temp = F.softmax(logits_temp, dim=2)  # bs, ci, co*ho*wo
            route = route_temp.view(bs, ci, co, ho, wo)  # bs, ci, co, ho, wo
            if CharacterizationUtils.characterize: 
                self.update_max(route, f"post_softmax_{iteration}")
                self.update_sqnr(route, f"post_softmax_{iteration}")
            preactivate_unrolled = route.unsqueeze(3) * votes  # bs, ci, co, no, ho, wo
            preactivate = preactivate_unrolled.sum(1) + bias  # bs, co, no, ho, wo
            if CharacterizationUtils.characterize: 
                self.update_max(preactivate, f"pre_squash_{iteration}")
                self.update_sqnr(preactivate, f"pre_squash_{iteration}")
            activation = self.squash(preactivate, dim=2)  # bs, co, no, ho, wo
            if CharacterizationUtils.characterize: 
                self.update_max(activation, f"output_{iteration}")
                self.update_sqnr(activation, f"output_{iteration}")
            act_3d = activation.unsqueeze(1)  # bs, 1, co, no, ho, wo
            distances = (act_3d * votes).sum(3)  # bs, ci, co, no, ho, wo --> bs, ci, co, ho, wo
            logits = logits + distances
            if CharacterizationUtils.characterize: 
                self.update_max(logits, f"pre_softmax_{iteration}")
                self.update_sqnr(logits, f"pre_softmax_{iteration}")

        return activation


class ConvPixelToCapsules(nn.Module, DynamicRouting):
    """ Convolutional layer that transforms the traditional feature maps in capsules

    Methods:
        forward:  compute the output of the layer given the input
    """

    def __init__(self, ci, ni, co, no, kernel_size, stride, padding, iterations):
        """ Parameters:
            ci: number of input channels
            ni: dimension of input capsules (1 if the inputs are traditional feature maps)
            co: number of output channels
            no: dimension of output capsules
            kernel_size: dimension of the kernel (square kernel)
            stride : stride parameter (horizontal and vertical strides are equal)
            padding : padding applied to the input
            iterations: number of iterations of the dynamic routing algorithm
        """
        super(ConvPixelToCapsules, self).__init__()
        CharacterizationUtils.__init__(self)

        self.ci = ci
        self.ni = ni
        self.co = co
        self.no = no
        self.iterations = iterations

        self.conv3d = nn.Conv2d(in_channels=ni,
                                out_channels=co * no,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                bias=False)

        self.bias = torch.nn.Parameter(torch.zeros(co, no, 1, 1))
        self.leaf = True

    def forward(self, x):
        """ forward method

        Args:
            x: input Tensor of shape [bs, ci, ni, hi, wi]
        Returns:
            activation: output Tensor of shape [bs, co, no, ho, wo]
        """
        bs, ci, ni, hi, wi = x.size()
        
        if CharacterizationUtils.characterize: 
            self.update_max(x, "input")
            self.update_sqnr(x, "input")
        # Reshape input and perform convolution to compute \hat{u_{j|i}} = votes
        input_reshaped = x.view(bs * ci, ni, hi, wi)
        votes = self.conv3d(input_reshaped)  # bs*ci, co*no, ho, wo
        _, _, ho, wo = votes.size()

        # Reshape votes, initialize logits and perform dynamic routing
        votes_reshaped = votes.view(bs, ci, self.co, self.no, ho, wo).contiguous()
        logits = votes_reshaped.new(bs, ci, self.co, ho, wo).zero_()
        
        if CharacterizationUtils.characterize: 
            self.update_max(votes, "votes")
            self.update_sqnr(votes, "votes")

        activation = self.update_routing(votes_reshaped, logits, self.iterations, self.bias)
        
        return activation


class Capsules(nn.Module, DynamicRouting):
    """ Capsule layer

        Methods:
            forward:  compute the output of the layer given the input
        """

    def __init__(self, ci, ni, co, no, iterations):
        """ Parameters:
            ci: number of input channels
            ni: dimension of input capsules
            co: number of output channels
            no: dimension of output capsules
            iterations: number of iterations of the dynamic routing algorithm
        """
        super(Capsules, self).__init__()
        CharacterizationUtils.__init__(self)

        self.weight = nn.Parameter(torch.randn(ci, ni, co * no))
        self.bias = nn.Parameter(torch.zeros(co, no))
        self.ci = ci
        self.co = co
        self.no = no
        self.ni = ni
        self.iterations = iterations
        self.leaf = True

        init.kaiming_uniform_(self.weight)
        init.constant_(self.bias, 0.1)

    def forward(self, x):
        """ forward method

        Args:
            x: input Tensor of shape [bs, ci, ni]
        Returns:
            activation: output Tensor of shape [bs, co, no]
        """
        bs = x.size(0)
        
        if CharacterizationUtils.characterize: 
            self.update_max(x, "input")
            self.update_sqnr(x, "input")

        # Compute \hat{u_{j|i}} = votes
        votes = (x.unsqueeze(3) * self.weight).sum(dim=2).view(-1, self.ci, self.co, self.no)
        
        if CharacterizationUtils.characterize: 
            self.update_max(votes, "votes")
            self.update_sqnr(votes, "votes")

        # Initialize logits and perform dynamic routing
        logits = votes.new(bs, self.ci, self.co).zero_()
        activation = self.update_routing(votes, logits, self.iterations, self.bias)
            
        return activation


class Conv2DCaps(nn.Module, DynamicRouting): 
    """ 2D Convolutional layer used in DeepCaps (no dynamic routing)

        Methods:
            forward: computes the output given the input
    """
    def __init__(self, ci, ni, co, no, kernel_size, stride, padding):
        """ Parameters:
            ci: number of input channels
            ni: dimension of input capsules
            co: number of output channels
            no: dimension of output capsules
            kernel_size: kernel size
            stride: stride
            padding: padding
        """
        super(Conv2DCaps, self).__init__()
        CharacterizationUtils.__init__(self)

        self.ci = ci  # number of capsules in input layer
        self.ni = ni  # atoms of capsules in input layer
        self.co = co  # number of capsules in output layer
        self.no = no  # atoms of capsules in output layer

        # input shape:   bs, ci, ni, hi, wi

        self.conv = nn.Conv2d(in_channels=ci * ni,
                              out_channels=co * no,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        init.xavier_uniform_(self.conv.weight)
        init.zeros_(self.conv.bias)
        
        self.leaf = True

    def forward(self, x):
        """ forward method

            Args:
                x: input Tensor of shape [batch_size, ci, ni, hi, wi]
            Returns:
                output: Tensor of shape [bathc_size, co, no, ho, wo]
        """
        if CharacterizationUtils.characterize: 
            self.update_max(x, "input")
            self.update_sqnr(x, "input")
        
        bs, ci, ni, hi, wi = x.size()
        input_reshaped = x.view(bs, ci * ni, hi, wi)

        output_reshaped = self.conv(input_reshaped)  # bs, co*no, ho, wo
        _, _, ho, wo = output_reshaped.size()

        output = output_reshaped.view(bs, self.co, self.no, ho, wo)  # bs, co, no, ho, wo
        
        if CharacterizationUtils.characterize: 
            self.update_max(output, "pre_squash")
            self.update_sqnr(output, "pre_squash")

        output = self.squash(output, dim=2)
        
        if CharacterizationUtils.characterize: 
            self.update_max(output, "output")
            self.update_sqnr(output, "output")

        return output


class Conv3DCaps(nn.Module, DynamicRouting):
    """ 3D convolutional capsule layer with dynamic routing

        Methods:
            forward: computes the output given the input
    """
    def __init__(self, ci, ni, co, no, kernel_size, stride, padding, iterations):
        """ Parameters:
            ci: number of input channels
            ni: dimension of input capsules
            co: number of output channels
            no: dimension of output capsules
            kernel_size: kernel size
            stride: stride
            padding: padding
            iterations: number of iterations of the dynamic routing algorithm
        """
        super(Conv3DCaps, self).__init__()
        CharacterizationUtils.__init__(self)
        self.ci = ci
        self.ni = ni
        self.co = co
        self.no = no

        self.conv = nn.Conv3d(in_channels=1,
                              out_channels=co * no,
                              kernel_size=(ni, kernel_size, kernel_size),
                              stride=(ni, stride, stride),
                              padding=(0, padding, padding))

        self.bias = nn.Parameter(torch.zeros(co, no, 1, 1))

        init.kaiming_uniform_(self.conv.weight)
        init.zeros_(self.conv.bias)
        init.constant_(self.bias, 0.1)

        self.iterations = iterations
        self.leaf = True

    def forward(self, x):
        """ forward method

            Args:
                x: Tensor of shape [batch_size, ci, ni, hi, wi]
            Returns:
                activation: Tensor of shape [batch_size, co, no, ho, wo]
        """
        bs, ci, ni, hi, wi = x.size()
        if CharacterizationUtils.characterize: 
            self.update_max(x, "input")
            self.update_sqnr(x, "input")

        input_tensor_reshaped = x.view(bs, 1, ci * ni, hi, wi)

        conv = self.conv(input_tensor_reshaped)  # bs, co*no, ci, ho, wo
        _, _, _, ho, wo = conv.size()

        votes = conv.permute(0, 2, 1, 3, 4).contiguous().view(bs, ci, self.co, self.no, ho, wo)
        if CharacterizationUtils.characterize: 
            self.update_max(votes, "votes")
            self.update_sqnr(votes, "votes")

        logits = votes.new(bs, ci, self.co, ho, wo).zero_()  # bs, ci, co, ho, wo

        activation = self.update_routing_6D_DeepCaps(votes, logits, self.iterations, self.bias)

        return activation  # bs, co, no, ho, wo


class DeepCapsBlock(nn.Module):
    """ DeepCaps block used in the DeepCaps architecture

        Consists of three serial layers and one parallel layer
        Methods:
            forward: computes the output given the input
    """
    def __init__(self, ci, ni, co, no, kernel_size, stride, padding, iterations):
        """Parameters:
            ci: number of input channels,
            ni: dimension of input capsules
            co: number of output channels
            no: dimension of output capsules
            kernel_size: kernel size of the convolutions
            stride: stride of the first convolution
            padding: list of four elements, padding factors for the four convolutions
            iterations: number of iterations of the dynamic routing. If 1, no dynamic routing is performed
        """
        super(DeepCapsBlock, self).__init__()
        self.leaf = False

        self.l1 = Conv2DCaps(ci=ci, ni=ni, co=co, no=no, kernel_size=kernel_size, stride=stride, padding=padding[0])
        self.l2 = Conv2DCaps(ci=co, ni=no, co=co, no=no, kernel_size=kernel_size, stride=1, padding=padding[1])
        self.l3 = Conv2DCaps(ci=co, ni=no, co=co, no=no, kernel_size=kernel_size, stride=1, padding=padding[2])
        if iterations == 1:
            self.l_skip = Conv2DCaps(ci=co, ni=no, co=co, no=no, kernel_size=kernel_size, stride=1, padding=padding[2])
        else:
            self.l_skip = Conv3DCaps(ci=co, ni=no, co=co, no=no, kernel_size=kernel_size, stride=1, padding=padding[3],
                                     iterations=iterations)

    def forward(self, x):
        """ forward method
            Args:
                x: input Tensor of size [batch_size, ci, ni, hi, wi]
            Returns:
                x: output Tensor of size [batch_size, co, no, ho, wo]
        """
        x = self.l1(x)
        x_skip = self.l_skip(x)
        x = self.l2(x)
        x = self.l3(x)
        x = x + x_skip

        return x
