import argparse
from collections import defaultdict, OrderedDict

import os
import torch
import torch.cuda
from torch.backends import cudnn
import sys
import numpy as np
import matplotlib.pyplot as plt

from test_train_functions import *
from full_precision_models import *
from full_precision_decoders import *
from utils import load_data
from characterization_utils import CharacterizationUtils

lims_dict = {}
lims_dict["ShallowCapsNet_mnist"] = {
    'post_softmax': 1., 'pre_squash': 4.05, 'output': 0.65, 'pre_softmax': 10.}
lims_dict["ShallowCapsNet_fashionmnist"] = {
    'post_softmax': 1., 'pre_squash': 4.95, 'output': 0.61, 'pre_softmax': 11.4}


def main():
    global args

    # Setting the hyper parameters
    parser = argparse.ArgumentParser(
        description='Analyze Activations ShallowCaps', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

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
    parser.add_argument('--batch-size', type=int, default=100,
                        help='training batch size. default=100')
    parser.add_argument('--trained-weights-path', type=str, default='./',
                        help='path of the pre-trained weights')

    # Parameters for testing
    parser.add_argument('--test-batch-size', type=int,
                        default=100, help='testing batch size. default=100')

    parser.add_argument('--full-precision-filename', type=str, default="./model.pt",
                        help="name for the full-precision model")

    # GPU parameters
    parser.add_argument('--visible-gpus', type=str, default="0",
                        help='set the ids of visible gpus, e.g. \'0\'. Default 0 (1 visible gpu)')
    parser.add_argument('--threads', type=int, default=4,
                        help='number of threads for data loader to use. default=4')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for training. default=42')

    args = parser.parse_args()

    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    # Load data
    train_loader, test_loader, num_channels, in_wh, num_classes = load_data(
        args)

    CharacterizationUtils.characterize = True

    # Build Capsule Network
    model_class = getattr(sys.modules[__name__], args.model)
    model = model_class(*args.model_args)
    model_filename = args.full_precision_filename

    # load pre-trained weights
    model.load_state_dict(torch.load(args.trained_weights_path))

    # Move model to GPU if possible
    if torch.cuda.device_count() > 0:
        print('Use GPUs for computation')
        print('Number of GPUs available:', torch.cuda.device_count())
        device = torch.device("cuda:0")
        model.to(device)
        cudnn.benchmark = True

    # Print the model architecture and parameters
    print('Model architecture:\n{}\n'.format(model))

    # PRE-TRAINED WEIGHTS EVALUATION
    best_accuracy = 0
    model.digit.limits_dict = lims_dict["ShallowCapsNet_" +
                                        args.dataset.replace('-', '')]
    best_accuracy = full_precision_test(
        model, num_classes, test_loader, model_filename, best_accuracy, False)
    print('\n \n Full-Precision Accuracy: ' + str(best_accuracy) + '%')

    info = OrderedDict()

    if args.model == "ShallowCapsNet":

        plt.rcParams.update({'font.size': 14})

        #primary_presquash = model.primary.hist["pre_squash_0"]
        digits_pre_squash = []
        digits_pre_softmax = []
        for i in range(3):
            digits_pre_squash.append(
                model.digit.hist[f"pre_squash_{i}"].numpy())
            digits_pre_softmax.append(
                model.digit.hist[f"pre_softmax_{i}"].numpy())

        limit_pre_softmax = lims_dict["ShallowCapsNet_" +
                                      args.dataset.replace('-', '')]['pre_softmax']
        limit_pre_squash = lims_dict["ShallowCapsNet_" +
                                     args.dataset.replace('-', '')]['pre_squash']
        centroids_pre_softmax = np.linspace(-limit_pre_softmax,
                                            limit_pre_softmax, 1000)
        centroids_pre_squash = np.linspace(-limit_pre_squash,
                                           limit_pre_squash, 1000)

        fig, axs = plt.subplots(3, 1, sharex=True, tight_layout=True)

        print("pre_softmax v1")
        for i in range(3):
            axs[i].hist(centroids_pre_softmax, bins=len(centroids_pre_softmax),
                        weights=digits_pre_softmax[i]/(10000*1152*10))
            axs[i].set_yscale('log')
            # print(model.digit.max_values_dict[f"pre_softmax_{i}"])
            # for c, w in zip(centroids_pre_softmax, digits_pre_softmax[i]):
            #    print(c, w)

        fig.savefig(
            f'figures/activations/ShallowCapsNet_{args.dataset.replace("-", "")}_presoftmax.png')
        fig.savefig(
            f'figures/activations/ShallowCapsNet_{args.dataset.replace("-", "")}_presoftmax.svg')

        plt.close()

        fig, ax = plt.subplots(
            1, 1, tight_layout=True, figsize=(4, 4))

        print("pre_softmax v2")
        for i in range(0, 3):
            ax.hist(centroids_pre_softmax, bins=len(centroids_pre_softmax),
                    weights=digits_pre_softmax[i]/(10000*1152*10), alpha=0.7, label=f'iter{i+1}', zorder=-(i-2)*5)

        ax.set_yscale('log')
        ax.grid(True)
        ax.set_xlabel("Numerical range")
        ax.set_ylabel("Normalized values distribution")
        ax.legend()
        if args.dataset == 'mnist':
            dataset_title = 'MNIST'
        else:
            dataset_title = 'FMNIST'
        ax.set_title(dataset_title+" - pre-softmax FM")

        fig.savefig(
            f'figures/activations/ShallowCapsNet_{args.dataset.replace("-", "")}_presoftmax_v2.png')
        fig.savefig(
            f'figures/activations/ShallowCapsNet_{args.dataset.replace("-", "")}_presoftmax_v2.svg')
        fig.savefig(
            f'figures/activations/ShallowCapsNet_{args.dataset.replace("-", "")}_presoftmax_v2.pdf')

        plt.close()

        fig, axs = plt.subplots(3, 1, sharex=True, tight_layout=True)

        print("pre_squash")
        for i in range(3):
            axs[i].hist(centroids_pre_squash, bins=len(centroids_pre_squash),
                        weights=digits_pre_squash[i]/(10000*10*16))
            axs[i].set_yscale('log')
            # print(model.digit.max_values_dict[f"pre_squash_{i}"])

        fig.savefig(
            f'figures/activations/ShallowCapsNet_{args.dataset.replace("-", "")}_presquash.png')
        fig.savefig(
            f'figures/activations/ShallowCapsNet_{args.dataset.replace("-", "")}_presquash.svg')

        plt.close()

        fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(4, 4))

        print("pre_squash")
        for i in range(0, 3):
            ax.hist(centroids_pre_squash, bins=len(centroids_pre_squash),
                    weights=digits_pre_squash[i]/(10000*10*16), alpha=0.7, label=f'iter{i}', zorder=-(i-2)*5)

            # print(model.digit.max_values_dict[f"pre_squash_{i}"])
        ax.set_yscale('log')
        ax.grid(True)
        ax.set_xlabel("Numerical range")
        ax.set_ylabel("Normalized values distribution")
        ax.legend()
        if args.dataset == 'mnist':
            dataset_title = 'MNIST'
        else:
            dataset_title = 'FMNIST'
        ax.set_title(dataset_title+" - pre-squash FM")

        fig.savefig(
            f'figures/activations/ShallowCapsNet_{args.dataset.replace("-", "")}_presquash_v2.png')
        fig.savefig(
            f'figures/activations/ShallowCapsNet_{args.dataset.replace("-", "")}_presquash_v2.svg')
        fig.savefig(
            f'figures/activations/ShallowCapsNet_{args.dataset.replace("-", "")}_presquash_v2.pdf')

        plt.close()

    elif args.model == "DeepCaps":
        max_values = OrderedDict()
        max_values["conv1"] = model.conv1.max_values_dict
        max_values["block1_l1"] = model.block1.l1.max_values_dict
        max_values["block1_l2"] = model.block1.l2.max_values_dict
        max_values["block1_l3"] = model.block1.l3.max_values_dict
        max_values["block1_lskip"] = model.block1.l_skip.max_values_dict
        max_values["block2_l1"] = model.block2.l1.max_values_dict
        max_values["block2_l2"] = model.block2.l2.max_values_dict
        max_values["block2_l3"] = model.block2.l3.max_values_dict
        max_values["block2_lskip"] = model.block2.l_skip.max_values_dict
        max_values["block3_l1"] = model.block3.l1.max_values_dict
        max_values["block3_l2"] = model.block3.l2.max_values_dict
        max_values["block3_l3"] = model.block3.l3.max_values_dict
        max_values["block3_lskip"] = model.block3.l_skip.max_values_dict
        max_values["block4_l1"] = model.block4.l1.max_values_dict
        max_values["block4_l2"] = model.block4.l2.max_values_dict
        max_values["block4_l3"] = model.block4.l3.max_values_dict
        max_values["block4_lskip"] = model.block4.l_skip.max_values_dict
        max_values["capsLayer"] = model.capsLayer.max_values_dict

        #torch.save(max_values, os.path.join("trained_models", "DeepCaps_"+args.dataset+"_top_actsf.pt"))

        scaling_factors = []
        scaling_factors.append(max_values["conv1"]["input"].item())
        scaling_factors.append(max_values["conv1"]["output"].item())
        for i in range(1, 5):  # 1,2,3,4
            for j in ["l1", "l2", "l3", "lskip"]:
                if i == 4 and j == "lskip":
                    continue
                scaling_factors.append(
                    max_values[f"block{i}_{j}"]["pre_squash"].item())
                scaling_factors.append(
                    max_values[f"block{i}_{j}"]["output"].item())

        scaling_factors.append(max_values["block4_lskip"]["votes"].item())
        for i in range(0, 3):
            scaling_factors.append(
                max_values["block4_lskip"][f"post_softmax_{i}"].item())
            scaling_factors.append(
                max_values["block4_lskip"][f"pre_squash_{i}"].item())
            scaling_factors.append(
                max_values["block4_lskip"][f"output_{i}"].item())
            scaling_factors.append(
                max_values["block4_lskip"][f"pre_softmax_{i}"].item())

        scaling_factors.append(max_values["capsLayer"]["votes"].item())
        for i in range(0, 3):
            scaling_factors.append(
                max_values["capsLayer"][f"post_softmax_{i}"].item())
            scaling_factors.append(
                max_values["capsLayer"][f"pre_squash_{i}"].item())
            scaling_factors.append(
                max_values["capsLayer"][f"output_{i}"].item())
            scaling_factors.append(
                max_values["capsLayer"][f"pre_softmax_{i}"].item())

        scaling_factors = torch.Tensor(scaling_factors)

        info["scaling_factors"] = scaling_factors

        layers_list = [model.conv1,
                       model.block1.l1, model.block1.l2, model.block1.l3, model.block1.l_skip,
                       model.block2.l1, model.block2.l2, model.block2.l3, model.block2.l_skip,
                       model.block3.l1, model.block3.l2, model.block3.l3, model.block3.l_skip,
                       model.block4.l1, model.block4.l2, model.block4.l3, model.block4.l_skip,
                       model.capsLayer]
        layers_list_name = ["conv1",
                            "block1.l1", "block1.l2", "block1.l3", "block1.l_skip",
                            "block2.l1", "block2.l2", "block2.l3", "block2.l_skip",
                            "block3.l1", "block3.l2", "block3.l3", "block3.l_skip",
                            "block4.l1", "block4.l2", "block4.l3", "block4.l_skip",
                            "capsLayer"]

        info["sqnr"] = OrderedDict()
        for layer, layer_name in zip(layers_list, layers_list_name):
            sum = 0
            n = 0
            for key, value in layer.sqnr_dict.items():
                sum += value
                n += 1
            info["sqnr"][layer_name] = sum/n

        # torch.save(info, os.path.join("trained_models", "DeepCaps_" +
        #           args.dataset.replace('-', '')+"_top_a_info.pt"))

    else:
        raise ValueError("Not supported network")


if __name__ == "__main__":
    main()
