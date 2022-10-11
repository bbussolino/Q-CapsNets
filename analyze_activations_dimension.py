import torch
import os
import sys
import argparse

from full_precision_models import *
from utils import load_data

global args

# Setting the hyper parameters
parser = argparse.ArgumentParser(
    description='Q-CapsNets framework', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Model parameters
parser.add_argument('--model', type=str, default="ShallowCapsNet",
                    help="Name of the model to be used")
parser.add_argument('--model-args', nargs="+", default=[28, 1, 10, 16], type=int,
                    help="arguments for the model instantiation")
parser.add_argument('--decoder', type=str, default="FCDecoder")
parser.add_argument('--decoder-args', nargs="+", default=[16, 28 * 28], type=int,
                    help="arguments for the model instantiation")
parser.add_argument('--dataset', type=str, default='mnist',
                    help='the name of dataset (mnist, cifar10)')

# Parameters for training
parser.add_argument('--no-training', action='store_true', default=False,
                    help='Set no-training for using pre-trained weights')
parser.add_argument('--full-precision-filename', type=str, default="./model.pt",
                    help="name for the full-precision model")
parser.add_argument('--trained-weights-path', type=str, default='./',
                    help='path of the pre-trained weights')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of training epochs. default=10')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate. default=0.001')
parser.add_argument('--batch-size', type=int, default=100,
                    help='training batch size. default=100')
parser.add_argument('--log-interval', type=int, default=10,
                    help='how many batches to wait before logging training status. default=10')
parser.add_argument('--regularization-scale', type=float, default=0.0005,
                    help='regularization coefficient for reconstruction loss. default=0.0005')
parser.add_argument('--decay-steps', type=int, default=2000,
                    help='decay steps for exponential learning rate adjustment.  default = 2000')
parser.add_argument('--decay-rate', type=float, default=0.96,
                    help='decay rate for exponential learning rate adjustment.  default=0.96 (no adjustment)')
parser.add_argument('--hard-training', action='store_true', default=False,
                    help="swith to hard training at the middle of the training phase")

# Parameters for testing
parser.add_argument('--test-batch-size', type=int,
                    default=100, help='testing batch size. default=100')

# GPU parameters
parser.add_argument('--visible-gpus', type=str, default="0",
                    help='set the ids of visible gpus, e.g. \'0\'. Default 0 ')
parser.add_argument('--threads', type=int, default=4,
                    help='number of threads for data loader to use. default=4')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed for training. default=42')


def main(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    args.model = 'ShallowCapsNet'
    args.model_args = [28, 1, 10, 16]
    args.test_batch_size = 1

    datasets = ["mnist", "fashion-mnist"]
    filenames = ["./trained_models/ShallowCapsNet_mnist_top.pt",
                 "./trained_models/ShallowCapsNet_fashionmnist_top.pt"]

    for dataset, filename in zip(datasets, filenames):

        args.dataset = dataset
        args.trained_weights_path = filename

        # Load data
        _, test_loader, num_channels, in_wh, num_classes = load_data(args)

        # Build Capsule Network
        model_class = getattr(sys.modules[__name__], args.model)
        model = model_class(*args.model_args)

        # Load pre-trained weights
        # load pre-trained weights
        model.load_state_dict(torch.load(args.trained_weights_path))

        # Move model to GPU if possible
        if torch.cuda.device_count() > 0:
            print('Use GPUs for computation')
            print('Number of GPUs available:', torch.cuda.device_count())
            device = torch.device("cuda:0")
            model.to(device)

        activation_sizes = {}

        def get_activation_sizes(name):
            # the hook signature
            def hook(model, input, output):
                activation_sizes[name+'_in'] = input[0].numel()
                activation_sizes[name+'_out'] = output.numel()
            return hook

        model.conv.register_forward_hook(get_activation_sizes('conv'))
        model.primary.register_forward_hook(get_activation_sizes('primary'))
        model.digit.register_forward_hook(get_activation_sizes('digit'))

        model.eval()
        for data, _ in test_loader:
            with torch.no_grad():
                if torch.cuda.device_count() > 0:  # if there are available GPUs, move data to the first visible
                    device = torch.device("cuda:0")
                    data = data.to(device)

                # Output predictions
                output = model(data)  # output from DigitCaps (out_digit_caps)
            break

        # print(activation_sizes)
        act_sizes_list = []
        for i, key in enumerate(activation_sizes.keys()):
            if i == 0 or '_out' in key:
                act_sizes_list.append(activation_sizes[key])
        print(act_sizes_list)

    args.model = 'DeepCaps'
    args.test_batch_size = 1

    datasets = ["mnist", "fashion-mnist", "cifar10"]
    filenames = ["./trained_models/DeepCaps_mnist_top.pt",
                 "./trained_models/DeepCaps_fashionmnist_top.pt",
                 "./trained_models/DeepCaps_cifar10_top.pt"]
    model_args = [[28, 1, 10, 32], [28, 1, 10, 32], [64, 3, 10, 32]]

    for dataset, filename, model_arg in zip(datasets, filenames, model_args):

        args.dataset = dataset
        args.trained_weights_path = filename
        args.model_args = model_arg

        # Load data
        _, test_loader, num_channels, in_wh, num_classes = load_data(args)

        # Build Capsule Network
        model_class = getattr(sys.modules[__name__], args.model)
        model = model_class(*args.model_args)

        # Load pre-trained weights
        # load pre-trained weights
        model.load_state_dict(torch.load(args.trained_weights_path))

        # Move model to GPU if possible
        if torch.cuda.device_count() > 0:
            print('Use GPUs for computation')
            print('Number of GPUs available:', torch.cuda.device_count())
            device = torch.device("cuda:0")
            model.to(device)

        activation_sizes = {}

        def get_activation_sizes(name):
            # the hook signature
            def hook(model, input, output):
                activation_sizes[name+'_in'] = input[0].numel()
                activation_sizes[name+'_out'] = output.numel()
            return hook

        model.conv1.register_forward_hook(get_activation_sizes('conv1'))
        model.block1.l1.register_forward_hook(
            get_activation_sizes('block1.l1'))
        model.block1.l2.register_forward_hook(
            get_activation_sizes('block1.l2'))
        model.block1.l3.register_forward_hook(
            get_activation_sizes('block1.l3'))
        model.block1.l_skip.register_forward_hook(
            get_activation_sizes('block1.l_skip'))
        model.block2.l1.register_forward_hook(
            get_activation_sizes('block2.l1'))
        model.block2.l2.register_forward_hook(
            get_activation_sizes('block2.l2'))
        model.block2.l3.register_forward_hook(
            get_activation_sizes('block2.l3'))
        model.block2.l_skip.register_forward_hook(
            get_activation_sizes('block2.l_skip'))
        model.block3.l1.register_forward_hook(
            get_activation_sizes('block3.l1'))
        model.block3.l2.register_forward_hook(
            get_activation_sizes('block3.l2'))
        model.block3.l3.register_forward_hook(
            get_activation_sizes('block3.l3'))
        model.block3.l_skip.register_forward_hook(
            get_activation_sizes('block3.l_skip'))
        model.block4.l1.register_forward_hook(
            get_activation_sizes('block4.l1'))
        model.block4.l2.register_forward_hook(
            get_activation_sizes('block4.l2'))
        model.block4.l3.register_forward_hook(
            get_activation_sizes('block4.l3'))
        model.block4.l_skip.register_forward_hook(
            get_activation_sizes('block4.l_skip'))
        model.capsLayer.register_forward_hook(
            get_activation_sizes('capsLayer'))

        model.eval()
        for data, _ in test_loader:
            with torch.no_grad():
                if torch.cuda.device_count() > 0:  # if there are available GPUs, move data to the first visible
                    device = torch.device("cuda:0")
                    data = data.to(device)

                # Output predictions
                output = model(data)  # output from DigitCaps (out_digit_caps)
            break

        # print(activation_sizes)
        act_sizes_list = []
        for i, key in enumerate(activation_sizes.keys()):
            if i == 0 or '_out' in key:
                act_sizes_list.append(activation_sizes[key])
        print(act_sizes_list)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
