import torch
import torch.nn as nn
import math
from typing import List
import numpy as np


class Solution:
    
    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:

        torch.manual_seed(0)

        var_xavier = 2 / (fan_out + fan_in)
        std_xavier = math.sqrt(var_xavier)

        weights = std_xavier * torch.randn(fan_out, fan_in)
        return torch.round(weights, decimals = 4).tolist()
        pass

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)

        var_kaiming = 2 / fan_in
        std_kaiming = math.sqrt(var_kaiming)

        weights = std_kaiming * torch.randn(fan_out, fan_in)
        return torch.round(weights , decimals = 4).tolist()

        pass

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Check if information sigmal survives as it travels through different layers

        torch.manual_seed(0)
        dims = [input_dim] + [hidden_dim]*num_layers

        weights = []
        if init_type.lower() == 'xavier':
            for i in range(num_layers):
                std_init = math.sqrt(2 / (dims[i] + dims[i+1]))
                w = std_init * torch.randn(dims[i + 1], dims[i])
                weights.append(w)
        elif init_type.lower() == 'kaiming':
            for i in range(num_layers):
                std_init = math.sqrt(2 / dims[i])
                w = std_init * torch.randn(dims[i + 1], dims[i])
                weights.append(w)  
        else:
            for i in range(num_layers):
                weights.append(torch.randn(dims[i + 1], dims[i]))
        
        x = torch.randn(1, input_dim)
        std_weights = []

        for w in weights:
            x = x @ w.T
            x = torch.relu(x)

            std_weights.append(torch.std(x).item())
        
        return np.round(std_weights, 2)

        pass
