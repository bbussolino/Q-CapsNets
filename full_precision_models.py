from full_precision_layers import *


class ShallowCapsNet(nn.Module):
    """ CapsNet model as in Sabour et al. 2017

    Methods:
        forward : compute the output of the layer given the input
    """

    def __init__(self, input_wh, input_ch, num_classes, dim_output_capsules):
        """ Parameters:
            input_wh: width/height of the input image in pixels
            input_ch: number of channels of the input images
            num_classes: number of classes of the dataset
            dim_output_capsules: dimension of the last layer capsules
        """
        super(ShallowCapsNet, self).__init__()
        self.conv = Conv2d_ReLU(in_channels=input_ch,
                                  out_channels=256,
                                  kernel_size=9,
                                  stride=1)
        self.primary = ConvPixelToCapsules(ci=1, ni=256, co=32, no=8,
                                             kernel_size=9, stride=2, padding=0, iterations=1)
        self.digit = Capsules(ci=1152, ni=8, co=num_classes, no=dim_output_capsules, iterations=3)

    def forward(self, x):
        """ forward method
            Args:
                x: input Tensor of size [batch_size, ci, hi, wi]
            Returns:
                out_digit: output Tensor of size [batch_size, co, no]"""
        out_conv = (self.conv(x)).unsqueeze(1)

        out_primary = self.primary(out_conv)
        bs, c, n, h, w = out_primary.size()
        out_primary = out_primary.permute(0, 1, 3, 4, 2).contiguous().view(bs, -1, n)

        out_digit = self.digit(out_primary)

        return out_digit


class DeepCaps(nn.Module):
    """ DeepCaps architecture

        Methods:
            forward: compute the output given the input """
    def __init__(self, input_wh, input_ch, num_classes, dim_output_capsules):
        """ Parameters:
                input_wh: width/height of the input image in pixels
                input_ch: number of channels of the input images
                num_classes: number of classes of the dataset
                dim_output_capsules: dimension of the last layer capsules
        """
        super(DeepCaps, self).__init__()  # 1 x 28 x 28   /  3 x 64 x 64

        # First convolution and conversion to capsules
        self.conv1 = Conv2d_BN_ReLU(in_channels=input_ch, out_channels=128, kernel_size=3, stride=1, padding=1,
                                    momentum=0.99, eps=0.001)
        # 128 x 1 x 28 x 28  /  128 x 1 x 64 x 64

        # Block One
        self.block1 = DeepCapsBlock(ci=1, ni=128, co=32, no=4, kernel_size=3,
                                    stride=2, padding=(1, 1, 1, 1),
                                    iterations=1)  # 32 x 4 x 14 x 14   /  32 x 4 x 32 x 32

        # Block Two
        self.block2 = DeepCapsBlock(ci=32, ni=4, co=32, no=8, kernel_size=3,
                                    stride=2, padding=(1, 1, 1, 1),
                                    iterations=1)  # 32 x 8 x 7 x 7    /  32 x 8 x 16 x 16

        # Block Three
        self.block3 = DeepCapsBlock(ci=32, ni=8, co=32, no=8, kernel_size=3,
                                    stride=2, padding=(1, 1, 1, 1), iterations=1)  # 32 x 8 x 4 x 4    /  32 x 8 x 8 x 8

        # Block Four
        self.block4 = DeepCapsBlock(ci=32, ni=8, co=32, no=8, kernel_size=3,
                                    stride=2, padding=(1, 1, 1, 1), iterations=3)  # 32 x 8 x 2 x 2    /  32 x 8 x 4 x 4

        # FC capsule layer
        if input_wh == 28:
            t = 640
        elif input_wh == 64:
            t = 2560
        else:
            raise ValueError('The parameter input_wh must be 28 or 64')

        self.capsLayer = Capsules(ci=t, ni=8, co=num_classes, no=dim_output_capsules, iterations=3)

    def forward(self, x):
        """ forward method

            Args:
                x: input Tensor of size [batch_size, ci, hi, wi]
            Returns:
                out_caps: output Tensor of size [batch_size, co, no] """
        # First convolution and conversion to capsules
        l = self.conv1(x)

        l = l.unsqueeze(1)

        # Block One
        l = self.block1(l)

        # Block Two
        l = self.block2(l)

        # Block Three
        l = self.block3(l)
        l1 = l

        # Block Four
        l = self.block4(l)
        l2 = l

        # Capsule Flattening and collection
        bs, ci, ni, hi, wi = l1.size()
        l1 = l1.permute(0, 1, 3, 4, 2).contiguous().view(bs, -1, ni)
        bs, ci, ni, hi, wi = l2.size()
        l2 = l2.permute(0, 1, 3, 4, 2).contiguous().view(bs, -1, ni)
        l = torch.cat((l1, l2), dim=1)

        # Linear capsule layer
        out_caps = self.capsLayer(l)

        return out_caps
