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
from q_capsnets_v2 import qcapsnets

POSSIBLE_STD_MULTIPLIERS = {}
POSSIBLE_STD_MULTIPLIERS["ShallowCapsNet"] = {
    "mnist": [6, 7], "fashion-mnist": [6, 7, 8]}
POSSIBLE_STD_MULTIPLIERS["DeepCaps"] = {
    "mnist": [6, 7], "fashion-mnist": [6, 7, 8, 1000], "cifar10": [6, 7, 8]}
1


def main():
    global args

    # Setting the hyper parameters
    parser = argparse.ArgumentParser(
        description='Q-CapsNets framework', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model parameters
    # parser.add_argument('--model', type=str, default="ShallowCapsNet",
    #                    help="Name of the model to be used")
    # parser.add_argument('--model-args', nargs="+", default=[28, 1, 10, 16], type=int,
    #                    help="arguments for the model instantiation")
    #parser.add_argument('--decoder', type=str, default="FCDecoder")
    # parser.add_argument('--decoder-args', nargs="+", default=[16, 28 * 28], type=int,
    #                    help="arguments for the model instantiation")
    # parser.add_argument('--dataset', type=str, default='mnist',
    #                    help='the name of dataset (mnist, cifar10)')

    # Parameters for training
    # parser.add_argument('--no-training', action='store_true', default=False,
    #                    help='Set no-training for using pre-trained weights')
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

    # Q-CapsNets parameters
    # parser.add_argument('--accuracy-tolerance', type=float, default=2,
    #                    help="accuracy tolerance expressed in percentage (e.g., 20 for 20%% tolerance)")
    # parser.add_argument('--quantization-method', type=str, default="stochastic_rounding",
    #                    help="String with the name of the quantization method to use")
    # parser.add_argument('--memory-budget', type=float, default=200,
    #                    help="Memory budget expressed in MB")
    # parser.add_argument('--std-multiplier', type=float, default=100,
    #                    help="Set to clamp scaling factor to [-std,std]*std_multiplier. default=100 (no clamping)")

    args = parser.parse_args()

    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    # for network - dataset combination   [5 combinations]
    # for rounding method [round_to_nearest, stochastic_rounding, truncation]
    # for scaling-factor    [8, 7, 6, 5, 4]  [5 combinations]
    # for accuracy-tolerance [0.2, 0.5, 1]  [3 combinations]
    # for memory budget     red [2, 3, 4, 5, 6, 7, 8, 10]  [8 combinations]
    model_dataset_combinations = []
    # model_dataset_combinations.append(["DeepCaps", [28, 1, 10, 32], "ConvDecoder28", [
    #                                  32, 1], "mnist", "./trained_models/DeepCaps_mnist_top.pt", 32.2])
    # model_dataset_combinations.append(["DeepCaps", [28, 1, 10, 32], "ConvDecoder28", [
    #                                  32, 1], "fashion-mnist", "./trained_models/DeepCaps_fashionmnist_top.pt", 32.2])
    # model_dataset_combinations.append(["DeepCaps", [64, 3, 10, 32], "ConvDecoder64", [
    #                                  32, 3], "cifar10", "./trained_models/DeepCaps_cifar10_top.pt", 50.98])
    model_dataset_combinations.append(["ShallowCapsNet", [20, 1, 10, 16], "FCDecoder", [
                                      16, 784], "mnist", "./trained_models/ShallowCapsNet_mnist_top.pt", 25.96])
    model_dataset_combinations.append(["ShallowCapsNet", [20, 1, 10, 16], "FCDecoder", [
                                      16, 784], "fashion-mnist", "./trained_models/ShallowCapsNet_fashionmnist_top.pt", 25.96])

    possible_std_multipliers = [10000]  # , 8, 7, 6, 5]
    possible_acc_tolerance = [0.2, 0.5, 1]
    possible_mem_budget_reduction = [2, 3, 4, 5, 6, 7, 8, 10]
    # , "stochastic_rounding", "truncation"]
    possible_roundings = ["round_to_nearest"]

    f = open("multiple_tests_new_grouped.txt", "a+")

    for model_dataset_info in model_dataset_combinations:
        for rounding in possible_roundings:
            f.write(
                f"{model_dataset_info[0]}\t{model_dataset_info[4]}\t{rounding}\n")
            # for std_multiplier in POSSIBLE_STD_MULTIPLIERS[model_dataset_info[0]][model_dataset_info[4]]:
            for std_multiplier in possible_std_multipliers:
                for acc_tolerance in possible_acc_tolerance:
                    for mem_budget_reduction in possible_mem_budget_reduction:

                        args.model, args.model_args, args.decoder, args.decoder_args, args.dataset, args.trained_weights_path, baseline_mem = model_dataset_info
                        args.std_multiplier = std_multiplier
                        args.accuracy_tolerance = acc_tolerance
                        args.memory_budget = baseline_mem / mem_budget_reduction
                        args.quantization_method = rounding

                        ################### REPEAT THIS #########################
                        # Load data
                        _, test_loader, num_channels, in_wh, num_classes = load_data(
                            args)

                        # Build Capsule Network
                        model_class = getattr(
                            sys.modules[__name__], args.model)
                        model = model_class(*args.model_args)
                        model_filename = args.full_precision_filename

                        # Load pre-trained weights
                        # load pre-trained weights
                        model.load_state_dict(
                            torch.load(args.trained_weights_path))

                        # Move model to GPU if possible
                        if torch.cuda.device_count() > 0:
                            print('Use GPUs for computation')
                            print('Number of GPUs available:',
                                  torch.cuda.device_count())
                            device = torch.device("cuda:0")
                            model.to(device)
                            cudnn.benchmark = True

                        # Print the model architecture and parameters
                        print('Model architecture:\n{}\n'.format(model))

                        # PRE-TRAINED WEIGHTS EVALUATION
                        best_accuracy = 0
                        best_accuracy = full_precision_test(
                            model, num_classes, test_loader, model_filename, best_accuracy, False)
                        print('\n \n Full-Precision Accuracy: ' +
                              str(best_accuracy) + '%')

                        full_precision_filename = args.trained_weights_path

                        # Q-CAPSNETS FRAMEWORK
                        result = qcapsnets(args.model, args.model_args, full_precision_filename, num_classes, test_loader, best_accuracy,
                                           args.accuracy_tolerance, args.memory_budget, args.quantization_method, args.std_multiplier)

                        f.write(
                            f"{std_multiplier}\t{acc_tolerance}\t{mem_budget_reduction}\t{result}\n")
                        print(
                            f"{std_multiplier}\t{acc_tolerance}\t{mem_budget_reduction}\t{result}\n")

    f.close()


if __name__ == "__main__":
    main()
