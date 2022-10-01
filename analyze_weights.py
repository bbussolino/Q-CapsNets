import torch 
import os 
from collections import OrderedDict

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
            dict_sf[key] = [torch.max(torch.abs(value)), torch.std(value)]            
            
    torch.save(dict_sf, os.path.join(model_folder, pt_file[:-3]+"_sf.pt"))
            