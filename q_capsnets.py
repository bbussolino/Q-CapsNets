from collections import OrderedDict
import torch
import copy
import sys
import numpy as np
from utils import one_hot_encode, capsnet_testing_loss
from torch.autograd import Variable
from torch.backends import cudnn
from quantization_methods import *
from quantized_models import *


def quantized_test(model, num_classes, data_loader, quantization_function, scaling_factors, quantization_bits,
                   quantization_bits_routing):
    """ Function to test the accuracy of the quantized models

        Args:
            model: pytorch model
            num_classes: number ot classes of the dataset
            data_loader: data loader of the test dataset
            quantization_function: quantization function of the quantization method to use
            quantization_bits: list, quantization bits for the activations
            quantization_bits_routing: list, quantization bits for the dynamic routing
        Returns:
            accuracy_percentage: accuracy of the quantized model expressed in percentage """
    # Switch to evaluate mode
    model.eval()

    loss = 0
    correct = 0

    num_batches = len(data_loader)

    for data, target in data_loader:
        batch_size = data.size(0)
        target_one_hot = one_hot_encode(target, length=num_classes)

        if torch.cuda.device_count() > 0:  # if there are available GPUs, move data to the first visible
            device = torch.device("cuda:0")
            data = data.to(device)
            target = target.to(device)
            target_one_hot = target_one_hot.to(device)

        # input quantization 
        data = quantization_function(data, scaling_factors[0], quantization_bits[0])

        # Output predictions
        output = model(data, quantization_function, scaling_factors[1:], quantization_bits, quantization_bits_routing)

        # Sum up batch loss
        m_loss = \
            capsnet_testing_loss(output, target_one_hot)
        loss += m_loss.data

        # Count number of correct predictions
        # Compute the norm of the vector capsules
        v_length = torch.sqrt((output ** 2).sum(dim=2))
        assert v_length.size() == torch.Size([batch_size, num_classes])

        # Find the index of the longest vector
        _, max_index = v_length.max(dim=1)
        assert max_index.size() == torch.Size([batch_size])

        # vector with 1 where the model makes a correct prediction, 0 where false
        correct_pred = torch.eq(target.cpu(), max_index.data.cpu())
        correct += correct_pred.sum()

    # Log test accuracies
    num_test_data = len(data_loader.dataset)
    accuracy_percentage = float(correct) * 100.0 / float(num_test_data)

    return accuracy_percentage


def qcapsnets(model, model_parameters, full_precision_filename, num_classes, data_loader, top_accuracy,
              accuracy_tolerance, memory_budget, quantization_scheme, std_multiplier=100):
    """ Q-CapsNets framework - Quantization

        Args:
            model: string, name of the model
            model_parameters: list, parameters to use for the instantiation of the model class
            full_precision_filename: string, directory of the full-precision weights
            num_classes: number of classes of the dataset
            data_loader: data loader of the testing dataset
            top_accuracy : maximum accuracy reached by the full_precision trained model (percentage)
            accuracy_tolerance: tolerance of the quantized model accuracy with respect to the full precision accuracy.
                                Provided in percentage
            memory_budget: memory budget for the weights of the model. Provided in MB (MegaBytes)
            quantization_scheme: quantization scheme to be used by the framework (string, e.g., "truncation)"
        Returns:
            void
    """
    print("==> Q-CapsNets Framework")
    # instantiate the quantized model with the full-precision weights
    model_quant_class = getattr(sys.modules[__name__], model)
    model_quant_original = model_quant_class(*model_parameters)
    model_quant_original.load_state_dict(torch.load(full_precision_filename))
    
    weights_scale_factors_tmp = torch.load(full_precision_filename[:-3]+'_sf.pt', map_location=torch.device('cpu'))
    weights_scale_factors = OrderedDict()
    for key, value in weights_scale_factors_tmp.items(): 
        weights_scale_factors[key] = torch.min(value[0], value[1]*std_multiplier)
    
    
    # Load scaling factors 
    act_scale_factors = torch.load(full_precision_filename[:-3]+'_actsf.pt', map_location=torch.device('cpu'))
    act_scale_factors = act_scale_factors.tolist()

    # Move the model to GPU if available
    if torch.cuda.device_count() > 0:
        device = torch.device("cuda:0")
        model_quant_original.to(device)
        cudnn.benchmark = True

    # create the quantization functions
    possible_functions = globals().copy()
    possible_functions.update(locals())
    quantization_function_activations = possible_functions.get(quantization_scheme)
    if not quantization_function_activations:
        raise NotImplementedError("Quantization function %s not implemented" % quantization_scheme)
    quantization_function_weights = possible_functions.get(quantization_scheme + "_inplace")
    if not quantization_function_weights:
        raise NotImplementedError("Quantization function %s not implemented (inplace version)" % quantization_scheme)

    # compute the accuracy reduction available for each step
    minimum_accuracy = top_accuracy - accuracy_tolerance / 100 * top_accuracy
    acc_reduction = top_accuracy - minimum_accuracy
    step1_reduction = 5 / 100 * acc_reduction
    step1_min_acc = top_accuracy - step1_reduction

    print(f"Full-precision accuracy: {top_accuracy:.2f} %")
    print(f"Minimum quantized accuracy: {minimum_accuracy:.2f} %")
    print(f"Memory budget: {memory_budget:.2f} MB")
    print(f"Quantization method: {quantization_scheme}")
    
    
    tot_numer_of_weights = 0 
    for i, c in enumerate(model_quant_original.named_children()):
        for p in c[1].named_parameters():
            tot_numer_of_weights += p[1].numel()
            
    tot_memory_b = tot_numer_of_weights * 32 
    tot_memory_B = tot_memory_b // 8 
    tot_memory_MB = tot_memory_B / 2**20
    
    print(f"Baseline memory footprint (MB): {tot_memory_MB:.2f} MB")

    # STEP 1: Layer-Uniform quantization of weights and activations
    print("STEP 1")

    def step1_quantization_test(quantization_bits):
        """ Function to test the model at STEP 1 of the algorithm

            The function receives a single "quantization_bits" value N, and creates two lists [N, N, ..., N] and
            [N, N, ..., N] for the activations and the dynamic routing, since at STEP 1 all the layers are quantized
            uniformly. The weights of each layer are quantized with N bits too and then the accuracy of the model
            is computed.

            Args:
                quantization_bits: single value used for quantizing all the weights and activations
            Returns:
                acc_temp: accuracy of the model quantized uniformly with quantization_bits bits
        """
        quantized_model_temp = copy.deepcopy(model_quant_original)

        step1_act_bits_f = []     # list with the quantization bits for the activations
        step1_dr_bits_f = []      # list with the quantization bits for the dynamic routing
        for c in quantized_model_temp.named_children():
            step1_act_bits_f.append(quantization_bits)
            if c[1].capsule_layer:
                if c[1].dynamic_routing:
                    step1_dr_bits_f.append(quantization_bits)
            for p in c[1].named_parameters():
                if not "batchnorm" in p[0]:
                    with torch.no_grad():
                        quantization_function_weights(p[1], weights_scale_factors['.'.join([c[0],p[0]])].item(), quantization_bits)      # Quantize the weights
        # test with quantized weights and activations
        acc_temp = quantized_test(quantized_model_temp, num_classes, data_loader,
                                  quantization_function_activations, act_scale_factors, step1_act_bits_f, step1_dr_bits_f)
        print(quantization_bits, step1_act_bits_f, step1_dr_bits_f, acc_temp)
        del quantized_model_temp
        return acc_temp

    # BINARY SEARCH of the bitwidth for step 1, starting from 32 bits
    step1_bit_search = [32]
    step1_acc_list = []      # list of accuracy at each step of the search algorithm
    step1_acc = step1_quantization_test(32)
    step1_acc_list.append(step1_acc)
    if step1_acc > step1_min_acc:
        step1_bit_search_sat = [True]    # True is the accuracy is higher than the minimum required
        step1_bit_search.append(16)
        while True:
            step1_acc = step1_quantization_test(step1_bit_search[-1])
            step1_acc_list.append(step1_acc)
            if step1_acc > step1_min_acc:
                step1_bit_search_sat.append(True)
            else:
                step1_bit_search_sat.append(False)
            if (abs(step1_bit_search[-1] - step1_bit_search[-2])) == 1:
                step1_bit_search_sat.reverse()
                step1_bits = step1_bit_search[
                    len(step1_bit_search_sat) - 1 - next(k for k, val in enumerate(step1_bit_search_sat) if val)]
                step1_bit_search_sat.reverse()
                step1_acc = step1_acc_list[
                    len(step1_bit_search_sat) - 1 - next(k for k, val in enumerate(step1_bit_search_sat) if val)]
                break
            else:
                if step1_acc > step1_min_acc:
                    step1_bit_search.append(
                        int(step1_bit_search[-1] - abs(step1_bit_search[-1] - step1_bit_search[-2]) / 2))
                else:
                    step1_bit_search.append(
                        int(step1_bit_search[-1] + abs(step1_bit_search[-1] - step1_bit_search[-2]) / 2))
    else:
        step1_bits = 32
        step1_acc = step1_acc_list[0]

    # Create the lists of bits ofSTEP 1
    step1_act_bits = []
    step1_dr_bits = []
    step1_weight_bits = []
    for c in model_quant_original.children():
        step1_act_bits.append(step1_bits)
        step1_weight_bits.append(step1_bits)
        if c.capsule_layer:
            if c.dynamic_routing:
                step1_dr_bits.append(step1_bits)

    print("STEP 1 output: ")
    print("\t Weight bits: \t\t", step1_weight_bits)
    print("\t Activation bits: \t\t", step1_act_bits)
    print("\t Dynamic Routing bits: \t\t", step1_dr_bits)
    print("STEP 1 accuracy: ", step1_acc)
    print("\n")

    # STEP2 - satisfy memory requirement
    # compute the number of weights and biases of each layer/block
    print("STEP 2")
    number_of_weights_inlayers = []
    for c in model_quant_original.children():
        param_intra_layer = 0
        for p in c.parameters():
            param_intra_layer = param_intra_layer + p.numel()
        number_of_weights_inlayers.append(param_intra_layer)
    number_of_blocks = len(number_of_weights_inlayers)

    memory_budget_bits = memory_budget * 8 * 2**20      # From MB to bits
    minimum_mem_required = np.sum(number_of_weights_inlayers)

    if memory_budget_bits < minimum_mem_required:
        #raise ValueError("The memory budget can not be satisfied, increase it to",
        #                 minimum_mem_required / 8 / 2**20, " MB at least")
        return f"ERROR The memory budget can not be satisfied, increase it to {minimum_mem_required / 8 / 2**20} MB at least"

    # Compute the number of bits that satisfy the memory budget.
    # First try with [N, N-1, N-2, N-3, N-4, N-4, ...].
    # If it is not possible, try with [N, N-1, N-2, N-3, N-3, ...]
    # and so on until [N, N, N, N, ...] (number of bits uniform across the layers)
    decrease_amount = 5
    while decrease_amount >= 0:
        bit_decrease = []
        if number_of_blocks <= decrease_amount:
            i = 0
            for r in range(0, number_of_blocks):
                bit_decrease.append(i)
                i = i - 1
        else:
            i = 0
            for r in range(0, decrease_amount):
                bit_decrease.append(i)
                i = i - 1
            for r in range(decrease_amount, number_of_blocks):
                bit_decrease.append(i + 1)

        bits_memory_sat = 33
        while True:
            # decrease N (bits_memory_sat) until the memory budget is satisfied.
            bits_memory_sat = bits_memory_sat - 1
            memory_occupied = np.sum(np.multiply(number_of_weights_inlayers, np.add(bits_memory_sat + 1, bit_decrease)))
            # +1 because bits_memory_sat are the fractional part bits, but we need one for the integer part
            if memory_occupied <= memory_budget_bits:
                break

        step2_weight_bits = list(np.add(bits_memory_sat, bit_decrease))
        if step2_weight_bits[-1] >= 0:
            break
        else:
            decrease_amount = decrease_amount - 1
            
    if any([w<=0 for w in step2_weight_bits]): 
        return f"ERROR Zero bits in weight bits" 

    # lists of bitwidths for activations and dynamic routing at STEP 1
    step2_act_bits = copy.deepcopy(step1_act_bits)
    step2_dr_bits = copy.deepcopy(step1_dr_bits)

    # Quantizeed the weights
    model_memory = copy.deepcopy(model_quant_original)
    for i, c in enumerate(model_memory.named_children()):
        for p in c[1].named_parameters():
            if not "batchnorm" in p[0]:
                with torch.no_grad():
                    quantization_function_weights(p[1], weights_scale_factors['.'.join([c[0],p[0]])].item(), step2_weight_bits[i])

    step2_acc = quantized_test(model_memory, num_classes, data_loader,
                               quantization_function_activations, act_scale_factors, step2_act_bits, step2_dr_bits)
    print(step2_weight_bits, step2_act_bits, step2_dr_bits, step2_acc)

    print("STEP 2 output: ")
    print("\t Weight bits: \t\t", step2_weight_bits)
    print("\t Activation bits: \t\t", step2_act_bits)
    print("\t Dynamic Routing bits: \t\t", step2_dr_bits)
    print("STEP 2 accuracy: ", step2_acc)
    print("\n")

    # IF the step 2 accuracy is higher that the minimum required accuracy --> BRANCH A
    if step2_acc > minimum_accuracy:
        # What is the accuracy that can still be consumed?
        branchA_accuracy_budget = step2_acc - minimum_accuracy
        step3A_min_acc = step2_acc - branchA_accuracy_budget * 55 / 100

        # STEP 3A  - layer-wise quantization of activations
        print("STEP 3A")
        # get the position of the layers that use dynamic routing bits
        dynamic_routing_bits_bool = []
        for c in model_memory.children():
            if c.capsule_layer:
                if c.dynamic_routing:
                    dynamic_routing_bits_bool.append(True)
            else:
                dynamic_routing_bits_bool.append(False)
        layers_dr_position = [pos for pos, val in enumerate(dynamic_routing_bits_bool) if val]

        step3a_weight_bits = copy.deepcopy(step2_weight_bits)
        step3a_act_bits = copy.deepcopy(step2_act_bits)
        step3a_dr_bits = copy.deepcopy(step2_dr_bits)
        for l in range(0, len(step3a_act_bits)):
            while True:
                step3a_acc = quantized_test(model_memory, num_classes, data_loader,
                                            quantization_function_activations, act_scale_factors, step3a_act_bits, step3a_dr_bits)
                print(step3a_act_bits, step3a_dr_bits, step3a_acc)
                if step3a_acc >= step3A_min_acc:
                    step3a_act_bits[l:] = list(np.add(step3a_act_bits[l:], -1))
                    for x in range(len(layers_dr_position)):
                        step3a_dr_bits[x] = step3a_act_bits[layers_dr_position[x]]
                else:
                    step3a_act_bits[l:] = list(np.add(step3a_act_bits[l:], +1))
                    for x in range(len(layers_dr_position)):
                        step3a_dr_bits[x] = step3a_act_bits[layers_dr_position[x]]
                    break

        step3a_acc = quantized_test(model_memory, num_classes, data_loader,
                                    quantization_function_activations, act_scale_factors, step3a_act_bits, step3a_dr_bits)
        print(step3a_act_bits, step3a_dr_bits, step3a_acc)

        print("STEP 3A output: ")
        print("\t Weight bits: \t\t", step3a_weight_bits)
        print("\t Activation bits: \t\t", step3a_act_bits)
        print("\t Dynamic Routing bits: \t\t", step3a_dr_bits)
        print("STEP 3A accuracy: ", step3a_acc)
        print("\n")

        # STEP 4A  -  layer-wise quantization of dynamic routing
        print("STEP 4A")
        step4a_weight_bits = copy.deepcopy(step2_weight_bits)
        step4a_act_bits = copy.deepcopy(step3a_act_bits)
        step4a_dr_bits = copy.deepcopy(step3a_dr_bits)

        # need to variate only the bits of the layers in which the dynamic routing is actually performed
        # (iterations > 1)
        dynamic_routing_quantization = []
        for c in model_memory.children():
            if c.capsule_layer:
                if c.dynamic_routing:
                    if c.dynamic_routing_quantization:
                        dynamic_routing_quantization.append(True)
                    else:
                        dynamic_routing_quantization.append(True)
        dr_quantization_pos = [pos for pos, val in enumerate(dynamic_routing_quantization) if val]

        # new set of bits only if dynamic routing is performed
        dr_quantization_bits = [step4a_dr_bits[x] for x in dr_quantization_pos]
        for l in range(0, len(dr_quantization_bits)):
            print(f"l {l}")
            while True:
                step4a_acc = quantized_test(model_memory, num_classes, data_loader,
                                            quantization_function_activations, act_scale_factors, step4a_act_bits, step4a_dr_bits)
                print(step4a_act_bits, step4a_dr_bits, step4a_acc)
                if step4a_acc >= minimum_accuracy:
                    dr_quantization_bits[l:] = list(np.add(dr_quantization_bits[l:], -1))
                    # update the whole vector step4a_dr_bits
                    for x in range(0, len(dr_quantization_bits)):
                        step4a_dr_bits[dr_quantization_pos[x]] = dr_quantization_bits[x]
                else:
                    dr_quantization_bits[l:] = list(np.add(dr_quantization_bits[l:], +1))
                    # update the whole vector step4a_dr_bits
                    for x in range(0, len(dr_quantization_bits)):
                        step4a_dr_bits[dr_quantization_pos[x]] = dr_quantization_bits[x]
                    break

        step4a_acc = quantized_test(model_memory, num_classes, data_loader,
                                    quantization_function_activations, act_scale_factors, step4a_act_bits, step4a_dr_bits)
        print(step4a_act_bits, step4a_dr_bits, step4a_acc)

        print("STEP 4A output: ")
        print("\t Weight bits: \t\t", step4a_weight_bits)
        print("\t Activation bits: \t\t", step4a_act_bits)
        print("\t Dynamic Routing bits: \t\t", step4a_dr_bits)
        print("STEP 4A accuracy: ", step4a_acc)
        print("\n")

        print("\n")
        quantized_filename = full_precision_filename[:-3] + '_quantized_satisfied.pt'
        torch.save(model_memory.state_dict(), quantized_filename)
        print("Model-satisfied stored in ", quantized_filename)
        print("\t Weight bits: \t\t", step4a_weight_bits)
        print("\t Activation bits: \t\t", step4a_act_bits)
        print("\t Dynamic Routing bits: \t\t", step4a_dr_bits)
        print("Model-satisfied accuracy: ", step4a_acc)
        
        assert len(number_of_weights_inlayers) == len(step4a_weight_bits)
        final_weight_memory_b = sum([number_of_weights_inlayers[i]*step4a_weight_bits[i] for i in range(len(number_of_weights_inlayers))])
        final_weight_memory_B = final_weight_memory_b / 8 
        wmem_reduction = tot_memory_b / final_weight_memory_b
        print(f"Weight memory reduction: {wmem_reduction:.2f}x")
        
        return step4a_weight_bits, step4a_act_bits, step4a_dr_bits, step4a_acc, wmem_reduction

    else:
        # BRANCH B - STEP 3B  - layer-wise quantization of the weights
        print("STEP 3B")
        step3b_weight_bits = copy.deepcopy(step1_weight_bits)
        step3b_act_bits = copy.deepcopy(step1_act_bits)
        step3b_dr_bits = copy.deepcopy(step1_dr_bits)

        model_accuracy = copy.deepcopy(model_quant_original)
        for i, c in enumerate(model_accuracy.named_children()):
            for p in c[1].named_parameters():
                if not "batchnorm" in p[0]:
                    with torch.no_grad():
                        quantization_function_weights(p[1], weights_scale_factors['.'.join([c[0],p[0]])].item(), step3b_weight_bits[i])

        for l in range(0, len(step3b_weight_bits)):
            while True:
                step3b_acc = quantized_test(model_accuracy, num_classes, data_loader,
                                            quantization_function_activations, act_scale_factors, step3b_act_bits, step3b_dr_bits)
                print(step3b_weight_bits, step3b_act_bits, step3b_dr_bits, step3b_acc)
                if step3b_acc >= minimum_accuracy:
                    step3b_weight_bits[l:] = list(np.add(step3b_weight_bits[l:], -1))
                    model_accuracy = copy.deepcopy(model_quant_original)
                    for i, c in enumerate(model_accuracy.named_children()):
                        for p in c[1].named_parameters():
                            if not "batchnorm" in p[0]:
                                with torch.no_grad():
                                    quantization_function_weights(p[1], weights_scale_factors['.'.join([c[0],p[0]])].item(), step3b_weight_bits[i])
                else:
                    step3b_weight_bits[l:] = list(np.add(step3b_weight_bits[l:], +1))
                    model_accuracy = copy.deepcopy(model_quant_original)
                    for i, c in enumerate(model_accuracy.named_children()):
                        for p in c[1].named_parameters():
                            if not "batchnorm" in p[0]:
                                with torch.no_grad():
                                    quantization_function_weights(p[1], weights_scale_factors['.'.join([c[0],p[0]])].item(), step3b_weight_bits[i])
                    break

        step3b_acc = quantized_test(model_accuracy, num_classes, data_loader,
                                    quantization_function_activations, act_scale_factors, step3b_act_bits, step3b_dr_bits)
        print(step3b_weight_bits, step3b_act_bits, step3b_dr_bits, step3b_acc)

        print("STEP 3B output: ")
        print("\t Weight bits: \t\t", step3b_weight_bits)
        print("\t Activation bits: \t\t", step3b_act_bits)
        print("\t Dynamic Routing bits: \t\t", step3b_dr_bits)
        print("STEP 3B accuracy: ", step3b_acc)
        print("\n")

        print("\n")
        quantized_filename = full_precision_filename[:-3] + '_quantized_memory.pt'
        torch.save(model_memory.state_dict(), quantized_filename)
        print("Model-memory stored in ", quantized_filename)
        print("\t Weight bits: \t\t", step2_weight_bits)
        print("\t Activation bits: \t\t", step2_act_bits)
        print("\t Dynamic Routing bits: \t\t", step2_dr_bits)
        print("Model_memory accuracy: ", step2_acc)
        print("\n")
        quantized_filename = full_precision_filename[:-3] + '_quantized_accuracy.pt'
        torch.save(model_accuracy.state_dict(), quantized_filename)
        print("Model-accuracy stored in ", quantized_filename)
        print("\t Weight bits: \t\t", step3b_weight_bits)
        print("\t Activation bits: \t\t", step3b_act_bits)
        print("\t Dynamic Routing bits: \t\t", step3b_dr_bits)
        print("Model_accuracy accuracy: ", step3b_acc)
        
        assert len(number_of_weights_inlayers) == len(step2_weight_bits)
        final_weight_memory_b = sum([number_of_weights_inlayers[i]*step2_weight_bits[i] for i in range(len(number_of_weights_inlayers))])
        wmem_reduction_mem = tot_memory_b / final_weight_memory_b
        print(f"Weight memory reduction - mem: {wmem_reduction_mem:.2f}x")
        
        assert len(number_of_weights_inlayers) == len(step3b_weight_bits)
        final_weight_memory_b = sum([number_of_weights_inlayers[i]*step3b_weight_bits[i] for i in range(len(number_of_weights_inlayers))])
        wmem_reduction_acc = tot_memory_b / final_weight_memory_b
        print(f"Weight memory reduction - acc: {wmem_reduction_acc:.2f}x")
        
        return step2_weight_bits, step2_act_bits, step2_dr_bits, step2_acc, wmem_reduction_mem, step3b_weight_bits, step3b_act_bits, step3b_dr_bits, step3b_acc, wmem_reduction_acc
