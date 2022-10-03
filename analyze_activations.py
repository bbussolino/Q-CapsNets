import argparse
from collections import defaultdict, OrderedDict
from symbol import pass_stmt
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
from characterization_utils import CharacterizationUtils


def main():
    global args

    # Setting the hyper parameters
    parser = argparse.ArgumentParser(description='Analyze Activations ShallowCaps', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

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
    train_loader, test_loader, num_channels, in_wh, num_classes = load_data(args)
    
    CharacterizationUtils.characterize = True 

    # Build Capsule Network
    model_class = getattr(sys.modules[__name__], args.model)
    model = model_class(*args.model_args)
    model_filename = args.full_precision_filename

    model.load_state_dict(torch.load(args.trained_weights_path))  # load pre-trained weights

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
    best_accuracy = full_precision_test(model, num_classes, test_loader, model_filename, best_accuracy, False)
    print('\n \n Full-Precision Accuracy: ' + str(best_accuracy) + '%')
    
    info = OrderedDict()
    
    if args.model == "ShallowCapsNet":
        max_values = OrderedDict()
        max_values["conv"] = model.conv.max_values_dict
        max_values["primary"] = model.primary.max_values_dict 
        max_values["digit"] = model.digit.max_values_dict 
        
        # torch.save(max_values, os.path.join("trained_models", "ShallowCapsNet_"+args.dataset+"_top_actsf.pt"))
        scaling_factors = [] 
        scaling_factors.append(max_values["conv"]["input"].item())
        scaling_factors.append(max_values["conv"]["output"].item())
        for l in ["primary", "digit"]: 
            for key, value in max_values[l].items(): 
                if key == "input": 
                    continue
                scaling_factors.append(value.item())
            
                
        scaling_factors = torch.Tensor(scaling_factors)
        info["scaling_factors"] = scaling_factors
        
        info["sqnr"] = OrderedDict()
        
        sum = 0 
        num_el = 0
        for key, value in model.conv.sqnr_dict.items(): 
            sum += value 
            num_el += 1 
        info["sqnr"]["conv"] = sum/num_el
        sum = 0 
        num_el = 0
        for key, value in model.primary.sqnr_dict.items(): 
            sum += value 
            num_el += 1 
        info["sqnr"]["primary"] = sum/num_el
        sum = 0 
        num_el = 0
        for key, value in model.digit.sqnr_dict.items(): 
            sum += value 
            num_el += 1
        info["sqnr"]["digit"] = sum/num_el
        
        torch.save(info, os.path.join("trained_models", "ShallowCapsNet_"+args.dataset.replace('-', '') +"_top_a_info.pt"))
        
    elif args.model == "DeepCaps":
        max_values = OrderedDict()
        max_values["conv1"] = model.conv1.max_values_dict 
        max_values["block1_l1"] = model.block1.l1.max_values_dict 
        max_values["block1_l2"] = model.block1.l2.max_values_dict 
        max_values["block1_l3"] = model.block1.l2.max_values_dict 
        max_values["block1_lskip"] = model.block1.l_skip.max_values_dict 
        max_values["block2_l1"] = model.block2.l1.max_values_dict 
        max_values["block2_l2"] = model.block2.l2.max_values_dict 
        max_values["block2_l3"] = model.block2.l2.max_values_dict 
        max_values["block2_lskip"] = model.block2.l_skip.max_values_dict 
        max_values["block3_l1"] = model.block3.l1.max_values_dict 
        max_values["block3_l2"] = model.block3.l2.max_values_dict 
        max_values["block3_l3"] = model.block3.l2.max_values_dict 
        max_values["block3_lskip"] = model.block3.l_skip.max_values_dict 
        max_values["block4_l1"] = model.block4.l1.max_values_dict 
        max_values["block4_l2"] = model.block4.l2.max_values_dict 
        max_values["block4_l3"] = model.block4.l2.max_values_dict 
        max_values["block4_lskip"] = model.block4.l_skip.max_values_dict 
        max_values["capsLayer"] = model.capsLayer.max_values_dict 
        
        
        #torch.save(max_values, os.path.join("trained_models", "DeepCaps_"+args.dataset+"_top_actsf.pt"))
        
        scaling_factors = []
        scaling_factors.append(max_values["conv1"]["input"].item())
        scaling_factors.append(max_values["conv1"]["output"].item())
        for i in range(1, 5): #1,2,3,4
            for j in ["l1", "l2", "l3", "lskip"]: 
                if i == 4 and j == "lskip": 
                    continue
                scaling_factors.append(max_values[f"block{i}_{j}"]["pre_squash"].item())
                scaling_factors.append(max_values[f"block{i}_{j}"]["output"].item())
                
        scaling_factors.append(max_values["block4_lskip"]["votes"].item())
        for i in range(0, 3): 
            scaling_factors.append(max_values["block4_lskip"][f"post_softmax_{i}"].item())
            scaling_factors.append(max_values["block4_lskip"][f"pre_squash_{i}"].item())
            scaling_factors.append(max_values["block4_lskip"][f"output_{i}"].item())
            scaling_factors.append(max_values["block4_lskip"][f"pre_softmax_{i}"].item())
            
            
        scaling_factors.append(max_values["capsLayer"]["votes"].item())
        for i in range(0, 3): 
            scaling_factors.append(max_values["capsLayer"][f"post_softmax_{i}"].item())
            scaling_factors.append(max_values["capsLayer"][f"pre_squash_{i}"].item())
            scaling_factors.append(max_values["capsLayer"][f"output_{i}"].item())
            scaling_factors.append(max_values["capsLayer"][f"pre_softmax_{i}"].item())
            
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
            sum  = 0 
            n = 0 
            for key, value in layer.sqnr_dict.items(): 
                sum += value 
                n += 1 
            info["sqnr"][layer_name] = sum/n
            
        torch.save(info, os.path.join("trained_models", "DeepCaps_"+args.dataset.replace('-','')+"_top_a_info.pt"))

    
    else: 
        raise ValueError("Not supported network")
        
        


if __name__ == "__main__":
    main()
