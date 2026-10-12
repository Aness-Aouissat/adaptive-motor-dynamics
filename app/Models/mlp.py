import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def from_np_to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.tensor(array, dtype = torch.float32)

def from_tensor_to_np(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()
    
class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size: int, hidden_layers: list[int], output_size: int):
        super().__init__() 

        self.layers = nn.ModuleList()

        input = input_size
        for layer in hidden_layers:
            self.layers.append(nn.Linear(input, layer, dtype = torch.float32))
            self.layers.append(nn.ReLU())
            input = layer

        self.layers.append(nn.Linear(input, output_size, dtype = torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for l in self.layers:
            out = l(x)
            x = out
        return out