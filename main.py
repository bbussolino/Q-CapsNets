import argparse
from timeit import default_timer as timer
import time
import math

import os
import torch
import torch.cuda
import torch.optim as optim
from torch.backends import cudnn
from torch.autograd import Variable
from tqdm import tqdm
import sys
from itertools import chain

from test_train_functions import *
from full_precision_models import *
from full_precision_decoders import *
from utils import load_data
from q_capsnets import qcapsnets


def main():
    global args

    # Setting the hyper parameters
    parser = argparse.ArgumentParser(description='Q-CapsNets framework')

    # Model parameters
    parser.add_argument('--model', type=str, default="ShallowCapsNet",
                        help="Name of the model to be used")
    parser.add_argument('--model-args', nargs="+", default=[1, 16], type=int,
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
    parser.add_argument('--visible-gpus', type=str, default="-1",
                        help='set the ids of visible gpus, e.g. \'0\'. Default -1 (no visible gpu)')
    parser.add_argument('--threads', type=int, default=4,
                        help='number of threads for data loader to use. default=4')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for training. default=42')

    # Q-CapsNets parameters
    parser.add_argument('--accuracy-tolerance', type=float, default=2,
                        help="accuracy tolerance expressed in percentage (e.g., 20 for 20% tolerance)")
    parser.add_argument('--quantization_method', type=str, default="stochastic_rounding",
                        help="String with the name of the quantization method to use")
    parser.add_argument('--memory-budget', type=float, default=200,
                        help="Memory budget expressed in MB")

    args = parser.parse_args()

    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    # Load data
    train_loader, test_loader, num_channels, in_wh, num_classes = load_data(args)

    # Build Capsule Network
    model_class = getattr(sys.modules[__name__], args.model)
    model = model_class(*args.model_args)
    model_filename = args.full_precision_filename

    if args.no_training:
        # Load pre-trained weights
        model.load_state_dict(torch.load(args.trained_weights_path))  # load pre-trained weights
    else:
        # Build decoder
        decoder_class = getattr(sys.modules[__name__], args.decoder)
        decoder = decoder_class(*args.decoder_args)  # build decoder for training

    # Move model to GPU if possible
    if torch.cuda.device_count() > 0:
        print('Use GPUs for computation')
        print('Number of GPUs available:', torch.cuda.device_count())
        device = torch.device("cuda:0")
        model.to(device)
        if not args.no_training:
            decoder.to(device)
        cudnn.benchmark = True

    # Print the model architecture and parameters
    print('Model architecture:\n{}\n'.format(model))
    if not args.no_training:
        print('Decoder architecture:\n{}\n'.format(decoder))

    if not args.no_training:
        # TRAINING
        # Optimizer
        optimizer = optim.Adam(chain(model.parameters(), decoder.parameters()), lr=args.lr)
        # Learning rate scheduler
        lambda_func = lambda step: args.decay_rate ** (step / args.decay_steps)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda_func)

        if args.hard_training:
            hard_list = [False] * math.ceil(args.epochs / 2) + [True] * math.floor(args.epochs / 2)
        else:
            hard_list = [False] * args.epochs

        best_accuracy = 0
        for epoch in range(1, args.epochs + 1):
            full_precision_training(model, decoder, num_classes, train_loader, optimizer, scheduler, epoch,
                                    hard_list[epoch - 1], args)
            best_accuracy = full_precision_test(model, num_classes, test_loader, model_filename, best_accuracy, True)

        print('\n \n Best Full-Precision Accuracy: ' + str(best_accuracy) + '%')

    else:
        # PRE-TRAINED WEIGHTS EVALUATION
        best_accuracy = 0
        best_accuracy = full_precision_test(model, num_classes, test_loader, model_filename, best_accuracy, False)
        print('\n \n Full-Precision Accuracy: ' + str(best_accuracy) + '%')

    if args.no_training:
        full_precision_filename = args.trained_weights_path
    else:
        full_precision_filename = model_filename

    # Q-CAPSNETS FRAMEWORK
    qcapsnets(args.model, args.model_args, full_precision_filename, num_classes, test_loader, best_accuracy,
              args.accuracy_tolerance, args.memory_budget, args.quantization_method)


if __name__ == "__main__":
    main()
