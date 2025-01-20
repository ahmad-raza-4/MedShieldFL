import torch
from torch import nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torch

class ViT(nn.Module):
    def __init__(self) -> None:
        super(ViT, self).__init__()
        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        in_features = self.model.heads[-1].in_features
        self.model.heads[-1] = nn.Linear(in_features, 8) 
        self.model.heads[-1].requires_grad = True  
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class ViT_GPU(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super(ViT_GPU, self).__init__()  
        self.device = device
        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).to(device)
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        in_features = self.model.heads[-1].in_features
        self.model.heads[-1] = nn.Linear(in_features, 8).to(device)
        self.model.heads[-1].requires_grad = True
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.to(self.device))