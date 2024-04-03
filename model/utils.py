import torch
import torch.nn as nn
from model import ProgramSynthesizer

from typing import Union, List

def model_summary(model):
    print("model_summary")
    print()
    print("Layer_name"+"\t"*7+"Number of Parameters")
    print("="*100)
    model_parameters = [layer for layer in model.parameters() if layer.requires_grad]
    print(f"model parameter: {model_parameters}")
    layer_name = [child for child in model.children()]
    print(f"layer name: {layer_name}")
    j = 0
    total_params = 0
    print("\t"*10)
    for i in layer_name:
        print()
        param = 0
        try:
            bias = (i.bias is not None)
        except:
            bias = False  
        if not bias:
            param =model_parameters[j].numel()+model_parameters[j+1].numel()
            j = j+2
        else:
            param =model_parameters[j].numel()
            j = j+1
        print(str(i)+"\t"*3+str(param))
        total_params+=param
    print("="*100)
    print(f"Total Params:{total_params}")   


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: Union[int, List[int], torch.Size], *,
                eps: float = 1e-5,
                elementwise_affine: bool = True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = torch.Size([normalized_shape])
        elif isinstance(normalized_shape, list):
            normalized_shape = torch.Size(normalized_shape)
        assert isinstance(normalized_shape, torch.Size)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.gain = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor):
        assert self.normalized_shape == x.shape[-len(self.normalized_shape):]
        dims = [-(i + 1) for i in range(len(self.normalized_shape))]
        mean = x.mean(dim=dims, keepdim=True)
        print(f"input: {x}")
        print(f"mean: {mean}")
        mean_x2 = (x ** 2).mean(dim=dims, keepdim=True)
        var = mean_x2 - mean ** 2
        print(f"var: {var}")
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            x_norm = self.gain * x_norm + self.bias
        return x_norm

def _test():
    x = torch.tensor([[[76, 2, 43], [79, 50, 29], [59, 78, 73], [95, 94, 76], [9, 74, 64]]])
    print(x.shape)
    ln = LayerNorm(x.shape[2:])
    x = ln(x.float())
    print(x)
    print(x.shape)


if __name__ == '__main__':
    _test()
    model = ProgramSynthesizer("test", n_layers=7, max_sequence_len=7, d_model=6, n_head=1)
    model_summary(model)
