import torch
import torch.nn as nn


def mask(out_caps, target):
    """ Function that masks the output of the last capsule layer before feeding it to the decoder.

    The function returns only the capsule corresponding to the correct label and discards the others.

    Args:
        out_caps: output of the last capsule layer    Tensor [batch_size, num_classes, dim_capsules]
        target: labels of the dataset                 Tensor [batch_size]
    Returns:
        masked_out: masked output of the last capsule layer   Tensor [batch_size, dim_capsules]
    """

    batch_index = torch.arange(out_caps.size(0))  # indices along batch dimension
    masked_out = out_caps[batch_index, target, :]  # mask with the target

    return masked_out


class FCDecoder(nn.Module):
    """ Decoder with 3 fully-connected layers as in Sabour et al. 2017

        Methods:
            forward : compute the output of the module given the input
    """

    def __init__(self, in_dim, out_dim):
        """ Parameters:
            in_dim : number of input neurons to the first layer
                     (capsule dimension of the last capsule layer)
            out_dim : number of output neurons of the last layer
                      (height*width of the input images)
        """
        super(FCDecoder, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fc1 = nn.Linear(in_features=in_dim, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=out_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, target):
        """ forward method

            Args:
                x: input Tensor of shape [batch_size, ci, ni]
                target: Tensor necessary to mask x, of shape [batch_size]
            Returns:
                fc3_out: output Tensor of shape [batch_size, out_dim]
        """

        # masking function
        vj = mask(x, target)  # [batch_size, ci,  ni] --> [batch_size, ni]

        fc1_out = self.relu(self.fc1(vj))
        fc2_out = self.relu(self.fc2(fc1_out))
        fc3_out = self.sigmoid(self.fc3(fc2_out))

        return fc3_out


class ConvDecoder28(nn.Module):
    """ Decoder with transposed convolution for 28x28 pixels images

        Methods:
            forward : compute the output of the module given the input
    """
    def __init__(self, input_size, out_channels):
        """ Parameters:
            input_size: number of input neurons to the first layer
                     (capsule dimension of the last capsule layer)
            out_channels: number of channels of the reconstructed image
        """
        super(ConvDecoder28, self).__init__()

        self.lin = nn.Linear(in_features=input_size, out_features=16 * 7 * 7)
        self.reluin = nn.ReLU()
        self.bn = nn.BatchNorm2d(num_features=16, momentum=0.8)
        self.dc1 = nn.ConvTranspose2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dc2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1,
                                      output_padding=1)
        self.dc3 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1,
                                      output_padding=1)
        self.dc4 = nn.ConvTranspose2d(in_channels=16, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.reluout = nn.ReLU()

    def forward(self, x, target):
        """ forward method

            Args:
                x: input Tensor of shape [batch_size, ci, ni]
                target: Tensor necessary to mask x, of shape [batch_size]
            Returns:
                vj: output Tensor of shape [batch_size, 28, 28]
        """
        # masking function
        vj = mask(x, target)  # [batch_size, ci,  ni] --> [batch_size, ni]

        vj = self.lin(vj)
        vj = self.reluin(vj)
        vj = vj.view(vj.size(0), 16, 7, 7)
        vj = self.bn(vj)
        vj = self.dc1(vj)
        vj = self.dc2(vj)
        vj = self.dc3(vj)
        vj = self.dc4(vj)
        vj = self.reluout(vj)

        return vj


class ConvDecoder64(nn.Module):
    """ Decoder with transposed convolution for 28x28 pixels images

        Methods:
            forward : compute the output of the module given the input
    """
    def __init__(self, input_size, out_channels):
        """ Parameters:
            input_size: number of input neurons to the first layer
                     (capsule dimension of the last capsule layer)
            out_channels: number of channels of the reconstructed image
        """
        super(ConvDecoder64, self).__init__()

        self.lin = nn.Linear(in_features=input_size, out_features=16 * 8 * 8)
        self.reluin = nn.ReLU()
        self.bn = nn.BatchNorm2d(num_features=16, momentum=0.8)
        self.dc1 = nn.ConvTranspose2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dc2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.dc3 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.dc4 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1)
        self.dc5 = nn.ConvTranspose2d(in_channels=8, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.reluout = nn.ReLU()

    def forward(self, x, target):
        """ forward method

            Args:
                x: input Tensor of shape [batch_size, ci, ni]
                target: Tensor necessary to mask x, of shape [batch_size]
            Returns:
                vj: output Tensor of shape [batch_size, 64, 64]
        """
        # masking function
        vj = mask(x, target)  # [batch_size, ci,  ni] --> [batch_size, ni]

        vj = self.lin(vj)
        vj = self.reluin(vj)
        vj = vj.view(vj.size(0), 16, 8, 8)
        vj = self.bn(vj)
        vj = self.dc1(vj)
        vj = self.dc2(vj)
        vj = self.dc3(vj)
        vj = self.dc4(vj)
        vj = self.dc5(vj)
        vj = self.reluout(vj)

        return vj
