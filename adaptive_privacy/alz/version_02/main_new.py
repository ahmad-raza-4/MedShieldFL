import torch
from torch import nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.transforms import Resize

class ViT(nn.Module):
    def __init__(self) -> None:
        super(ViT, self).__init__()
        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        
        # Adjust for 32x32 input images by updating the patch embedding layer
        self.model.conv_proj = nn.Conv2d(
            in_channels=3, out_channels=768, kernel_size=(16, 16), stride=(16, 16)
        )
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        in_features = self.model.heads[-1].in_features
        self.model.heads[-1] = nn.Linear(in_features, 4)  # Change to 4 classes
        self.model.heads[-1].requires_grad = True  
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class ViT_GPU(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super(ViT_GPU, self).__init__()  
        self.device = device
        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).to(device)

        # Adjust for 32x32 input images by updating the patch embedding layer
        self.model.conv_proj = nn.Conv2d(
            in_channels=3, out_channels=768, kernel_size=(16, 16), stride=(16, 16)
        ).to(device)
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        in_features = self.model.heads[-1].in_features
        self.model.heads[-1] = nn.Linear(in_features, 4).to(device)  # Change to 4 classes
        self.model.heads[-1].requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.to(self.device))
