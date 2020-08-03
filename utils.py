import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms, datasets
import torchvision.utils as vutils
import argparse
import math

from data_loaders import *


def one_hot_encode(target, length):
    """Converts a batch of class indices to a batch of one-hot vectors.

    Args:
        target  :  Tensor [batch size]    Labels for dataset.
        length  :  int                    Number of classes of the dataset

    Returns:
        one_hot vec  :   Tensor [batch size, length]    One-Hot representation of the target """

    batch_size = target.size(0)
    one_hot_vec = torch.zeros(batch_size, length)

    for i in range(batch_size):
        one_hot_vec[i, target[i]] = 1.0

    return one_hot_vec


def accuracy(output, target):
    """
    Compute accuracy comparing the output of the model and the target.

    Args:
        output  : [batch_size, num_classes, caps_dim] The output from the last caps layer.
        target  : [batch_size] Labels for dataset.

    Returns:
        accuracy (float): The accuracy for a batch.
    """
    batch_size = target.size(0)

    # Compute the norm of the vector capsules
    v_length = torch.sqrt((output ** 2).sum(dim=2))

    # Find the index of the longest vector
    _, max_index = v_length.max(dim=1)

    # vector with 1 where the model makes a correct prediction, 0 where false
    correct_pred = torch.eq(target.cpu(), max_index.data.cpu())

    acc = correct_pred.float().mean()  # mean accuracy of a batch

    return acc


def capsnet_training_loss(out_digit, target_one_hot, scale, reconstruction, image, hard=False):
    """ Function that computes Margin loss, Reconstruction loss and Total loss for a CapsNet

    Args:
        out_digit  :  [batch_size, num_classes, caps_dim]   output of the last capsule layer
        target_one_hot  :  [batch_size, num_classes]    Labels of the dataset in one-hot encoding
        scale  :  float     scale factor for the Reconstruction loss
        reconstruction  :  [batch_size, width*height]    Images reconstructed by the decoder
        image  :  [batch_size, width*height]     Input images
        hard: bool that toggles hard training (default = False)

    Returns:
        tloss: Total loss   (float)
        mloss: Margin loss  (float)
        rloss: Reconstruction loss (float)
        """
    pred = torch.norm(out_digit, dim=2)

    if hard:
        m1 = (F.relu(0.95 - pred)) ** 2
        m0 = (F.relu(pred - 0.05)) ** 2
        coeff = 0.8
    else:
        m1 = (F.relu(0.9 - pred)) ** 2
        m0 = (F.relu(pred - 0.1)) ** 2
        coeff = 0.5

    Lk = target_one_hot * m1 + coeff * (1. - target_one_hot) * m0

    mloss = Lk.sum(dim=1)

    reconstruction = reconstruction.view(reconstruction.size(0), -1)
    mse = (image - reconstruction) ** 2
    rloss = mse.sum(dim=1)

    tloss = (mloss + scale * rloss).mean()
    mloss = mloss.mean()
    rloss = rloss.mean()

    return tloss, mloss, rloss


def capsnet_testing_loss(out_digit, target_one_hot):
    """ Function that computes Margin loss, Reconstruction loss and Total loss for a CapsNet

    Args:
        out_digit  :  [batch_size, num_classes, caps_dim]   output of the last capsule layer
        target_one_hot  :  [batch_size, num_classes]    Labels of the dataset in one-hot encoding

    Returns:
        mloss: Margin loss  (float)
        """

    pred = torch.norm(out_digit, dim=2)

    m1 = (F.relu(0.9 - pred)) ** 2
    m0 = (F.relu(pred - 0.1)) ** 2
    coeff = 0.5

    Lk = target_one_hot * m1 + coeff * (1. - target_one_hot) * m0

    mloss = Lk.sum(dim=1)

    mloss = mloss.mean()

    return mloss


def load_data(args):
    """
    Load dataset

    Args:
        args: arguments set by the user
    """
    dst = args.dataset

    if dst == 'mnist':
        return load_mnist(args)
    if dst == 'fashion-mnist':
        return load_fmnist(args)
    if dst == 'cifar10':
        return load_cifar10(args)
    if dst == 'svhn':
        return load_svhn(args)
    else:
        raise Exception('Invalid dataset, please check the name of dataset:', dst)
