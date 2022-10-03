import torch 
import math 
import numpy as np
from collections import defaultdict
from quantization_methods import round_to_nearest

def default_zero(): 
    tmp = torch.Tensor([0.])
    if torch.cuda.device_count() > 0:
        
        tmp = tmp.to(torch.device("cuda:0"))
    return tmp

def default_one(): 
    tmp = torch.Tensor([1.])
    if torch.cuda.device_count() > 0:
        
        tmp = tmp.to(torch.device("cuda:0"))
    return tmp

class CharacterizationUtils: 
    characterize = True 
    n = 16
    def __init__(self): 
        self.max_values_dict = defaultdict(default_zero)
        self.sqnr_dict = {}
        self.signal_power = {}
        self.noise_power = {}
    
    def update_max(self, tensor, name): 
        self.max_values_dict[name] = torch.max(torch.max(torch.abs(tensor.detach())), self.max_values_dict[name])
        
    def update_sqnr(self, tensor, name): 
        curr_signal_power = torch.mean(tensor*tensor)
        quant_signal = round_to_nearest(tensor, self.max_values_dict[name], self.n)
        curr_noise_power = torch.mean((tensor-quant_signal)**2)
        if name not in self.signal_power.keys(): 
            self.signal_power[name] = curr_signal_power.item()
            self.noise_power[name] = curr_noise_power.item()
            self.sqnr_dict[name] = 10 * math.log10((curr_signal_power / curr_noise_power).item())
        else: 
            self.signal_power[name] = np.mean([self.signal_power[name], curr_signal_power.item()])
            self.noise_power[name] = np.mean([self.noise_power[name], curr_noise_power.item()])
            self.sqnr_dict[name] = 10 * math.log10(self.signal_power[name] / self.noise_power[name])
        
        