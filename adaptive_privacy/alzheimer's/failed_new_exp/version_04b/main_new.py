import torch
from torch import nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from opacus.layers import DPMultiheadAttention

class ViT(nn.Module):
    def __init__(self) -> None:
        super(ViT, self).__init__()
        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        self.model.image_size = 32  # Add this line
        
        # Resize positional embeddings for 32x32 input
        num_patches = (32 // self.model.patch_size) ** 2
        self.model.encoder.pos_embedding = nn.Parameter(
            self.model.encoder.pos_embedding[:, :num_patches + 1, :]
        )

        # Replace attention layers
        for layer in self.model.encoder.layers:
            original_attn = layer.self_attention  # Corrected attribute name
            dp_attn = DPMultiheadAttention(
                embed_dim=original_attn.embed_dim,
                num_heads=original_attn.num_heads,
                dropout=original_attn.dropout,
                bias=True,
                batch_first=True
            )
            dp_attn.load_state_dict(original_attn.state_dict())
            layer.self_attention = dp_attn.to("cpu")  # Corrected attribute name

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.encoder.layers[-2:].parameters():  # Unfreeze last 2 layers
            param.requires_grad = True
        
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
        

        # Update image_size to 32
        self.model.image_size = 32  # Add this line
        
        # Resize positional embeddings for 32x32 input
        num_patches = (32 // self.model.patch_size) ** 2
        self.model.encoder.pos_embedding = nn.Parameter(
            self.model.encoder.pos_embedding[:, :num_patches + 1, :]
        )

        # Replace standard attention with DP-compatible attention
        for layer in self.model.encoder.layers:
            original_attn = layer.self_attention  # Corrected attribute name
            dp_attn = DPMultiheadAttention(
                embed_dim=original_attn.embed_dim,
                num_heads=original_attn.num_heads,
                dropout=original_attn.dropout,
                bias=True,
                batch_first=True
            )
            dp_attn.load_state_dict(original_attn.state_dict())
            layer.self_attention = dp_attn.to(device)  # Corrected attribute name

        # Freeze logic (now safe for DP)
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.encoder.layers[-2:].parameters():
            param.requires_grad = True

        in_features = self.model.heads[-1].in_features
        self.model.heads[-1] = nn.Linear(in_features, 4).to(device)
        self.model.heads[-1].requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.to(self.device))
