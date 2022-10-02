from multiprocessing.sharedctypes import Value
import torch 
import os 
import sys 
from collections import OrderedDict
from quantized_models import *

from quantization_methods import *
import math 

from q_capsnets import quantized_test
from utils import load_data
import argparse


def main():
    global args

    # Setting the hyper parameters
    parser = argparse.ArgumentParser(description='Q-CapsNets framework', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

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

    

    args = parser.parse_args()

    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
    model_folder = "trained_models"
    files = ["ShallowCapsNet_mnist_top.pt", "ShallowCapsNet_fashionmnist_top.pt", 
            "DeepCaps_mnist_top.pt", "DeepCaps_fashionmnist_top.pt", "DeepCaps_cifar10_top.pt"]


    dict_sf = OrderedDict() 
    
    #f = open("weights.csv", "w+")   

    #for pt_file in files: 
    #    dict = torch.load(os.path.join(model_folder, pt_file), map_location=torch.device('cpu'))
    #    f.write(pt_file[:-3]+"\n")
    #    for key, value in dict.items(): 
    #        if "weight" in key: 
    #            print(key)
    #            maxv = torch.max(torch.abs(value))
    #            mu = torch.mean(value)
    #            chi = torch.std(value)
    #            exp = torch.mean(value*value)
    #            
    #            sqnr = [] 
    #            for b in range(1, 17): 
    #                value_q = round_to_nearest(value, maxv, b)
    #                exp_q = torch.mean((value-value_q)**2)
    #                sqnr.append(10 * math.log10((exp/exp_q).item()))
                    
                
    #            print(f"Max: {maxv}")
    #            print(f"Mean: {mu}")
    #            print(f"Std: {chi}")
    #            print(f"SQNR {sqnr}")
    #            
    #            str_out = key+","
    #            for o in sqnr: 
    #                str_out = str_out + str(o) + ","
    #            str_out = str_out + "\n"
    #            
    #            f.write(str_out)
    #f.close()
            
            
    model_parameters = args.model_args
    full_precision_filename = args.trained_weights_path
    quantization_scheme = "round_to_nearest"
    _, data_loader, num_channels, in_wh, num_classes = load_data(args)

    model_quant_class = getattr(sys.modules[__name__], args.model)
    model_quant_original = model_quant_class(*model_parameters)
    model_quant_original.load_state_dict(torch.load(full_precision_filename))

    weights_scale_factors = torch.load(full_precision_filename[:-3]+'_sf.pt', map_location=torch.device('cpu'))

    # Load scaling factors 
    act_scale_factors = torch.load(full_precision_filename[:-3]+'_actsf.pt', map_location=torch.device('cpu'))
    act_scale_factors = act_scale_factors.tolist()

    # Move the model to GPU if available
    if torch.cuda.device_count() > 0:
        device = torch.device("cuda:0")
        model_quant_original.to(device)

    # create the quantization functions
    possible_functions = globals().copy()
    possible_functions.update(locals())
    quantization_function_activations = possible_functions.get(quantization_scheme)
    if not quantization_function_activations:
        raise NotImplementedError("Quantization function %s not implemented" % quantization_scheme)
    quantization_function_weights = possible_functions.get(quantization_scheme + "_inplace")
    if not quantization_function_weights:
        raise NotImplementedError("Quantization function %s not implemented (inplace version)" % quantization_scheme)



    if type(model_quant_original) == ShallowCapsNet: 
        act_bits = [32, 32, 32]
        dr_bits = [32, 32]
        weight_bits = [32, 32, 32] 
        len_weights = len(weight_bits)
    elif type(model_quant_original) == DeepCaps:
        act_bits = [32 for _ in range(18)]
        dr_bits = [32, 32]
        weight_bits = [32 for _ in range(18)]
        len_weights = len(weight_bits)
    else: 
        raise ValueError("Wrong Model")

    f = open("accuracies.csv", "w+")
    f.write("ShallowCapsNet_fashionmnist_top\n")
    
    leaf_children = []
    leaf_children_names = []
    def get_leaf_modules(top_name, m): 
        # m: generator 
        for key, value in m: 
            if sum(1 for _ in value.children()) == 0: 
                leaf_children.append(value)
                leaf_children_names.append(top_name + '.' + key)
                return 
            else: 
                get_leaf_modules(top_name + '.' + key, value.named_children())


    for l in range(len(weight_bits)): 
        weight_bits = [32 for _ in range(len_weights)] 
        str_out = "\t"
        for b in range(1, 17): 
            weight_bits[l] = b 
            
            quantized_model_temp = copy.deepcopy(model_quant_original)
            
            leaf_children = []
            leaf_children_names = []
            get_leaf_modules("", quantized_model_temp.named_children())
            
            for i, (name, children) in enumerate(zip(leaf_children_names, leaf_children)): 
                for p in children.named_parameters(): 
                    with torch.no_grad(): 
                        #print(f"quantize {name[1:]}.{p[0]} with {weight_bits[i]}")
                        quantization_function_weights(p[1], weights_scale_factors['.'.join([name[1:],p[0]])].item(), weight_bits[i])
                        
            acc = quantized_test(quantized_model_temp, num_classes, data_loader,
                                quantization_function_activations, act_scale_factors, act_bits, dr_bits)
            str_out = str_out + str(acc) + "\t"
            print(weight_bits, acc)
        str_out = str_out+"\n"
        f.write(str_out)
    f.close()
        
        
if __name__ == '__main__': 
    main()
    
    
def get_leaf_modules(top_name, m): 
    # m: generator 
    for key, value in m: 
        if hasattr(value, "leaf") and value.leaf == True: 
            leaf_children.append(value)
            leaf_children_names.append(top_name + '.' + key)
        else: 
            get_leaf_modules(top_name + '.' + key, value.named_children())
    return 