import torch 
import os 
import math 
from collections import OrderedDict

from quantization_methods import round_to_nearest

model_folder = "trained_models"
files = ["ShallowCapsNet_mnist_top.pt", "ShallowCapsNet_fashionmnist_top.pt", 
        "DeepCaps_mnist_top.pt", "DeepCaps_fashionmnist_top.pt", "DeepCaps_cifar10_top.pt"]

for pt_file in files: 
    print(pt_file)

    dict = torch.load(os.path.join(model_folder, pt_file), map_location=torch.device('cpu'))
    
    dict_sf = OrderedDict() 
    dict_std = OrderedDict()
        
    for key, value in dict.items(): 
        #if "weight" in key: 
        if "batchnorm" not in key:
            # print(value)
            maxv = torch.max(torch.abs(value))
            mu = torch.mean(value)
            chi = torch.std(value)
            exp = torch.mean(value*value)
            
            sqnr = [] 
            for b in range(1, 17): 
                value_q = round_to_nearest(value, maxv, b)
                exp_q = torch.mean((value-value_q)**2)
                sqnr.append(10 * math.log10((exp/exp_q).item()))
                
            sqnr = torch.Tensor(sqnr)
            sqnr = torch.mean(sqnr)
            
            dict_sf[key] = [maxv, chi, sqnr]            
            
    torch.save(dict_sf, os.path.join(model_folder, pt_file[:-3]+"_sf.pt"))
            