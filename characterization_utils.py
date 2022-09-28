import torch 
from collections import defaultdict

def default(): 
    tmp = torch.Tensor([0.])
    if torch.cuda.device_count() > 0:
        
        tmp = tmp.to(torch.device("cuda:0"))
    return tmp

class CharacterizationUtils: 
    characterize = True 
    def __init__(self): 
        self.max_values_dict = defaultdict(default)
        
    def update_max(self, tensor, name): 
        self.max_values_dict[name] = torch.max(torch.max(torch.abs(tensor.detach())), self.max_values_dict[name])