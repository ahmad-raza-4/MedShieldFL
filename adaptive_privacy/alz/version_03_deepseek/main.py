import torch
from torch import nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.nn import functional as F

class ViT(nn.Module):
    def __init__(self) -> None:
        super(ViT, self).__init__()
        # Load pre-trained model with validation checks
        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        
        # Adjust for 32x32 input
        self._adapt_positional_embeddings()
        
        # Parameter freezing strategy
        self._configure_parameter_groups()
        
        # Enhanced classification head
        self._build_custom_head()
        
        # Regularization
        self.dropout = nn.Dropout(0.2)
        
        # Initialize head properly
        self._initialize_weights()

    def _adapt_positional_embeddings(self):
        """Adjust positional embeddings for smaller input size"""
        patch_size = self.model.patch_size
        num_patches = (32 // patch_size) ** 2
        orig_embedding = self.model.encoder.pos_embedding
        
        # Linear interpolation for positional embeddings
        new_embedding = F.interpolate(
            orig_embedding.permute(0, 2, 1),
            size=num_patches + 1,
            mode='linear'
        ).permute(0, 2, 1)
        
        self.model.encoder.pos_embedding = nn.Parameter(new_embedding)
        self.model.image_size = 32  # Explicitly set expected input size

    def _configure_parameter_groups(self):
        """Configure trainable parameter groups with gradual unfreezing"""
        # Freeze all parameters initially
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Unfreeze last 4 transformer blocks + positional embeddings
        for block in self.model.encoder.layers[-4:]:
            for param in block.parameters():
                param.requires_grad = True
                
        # Unfreeze positional embeddings
        self.model.encoder.pos_embedding.requires_grad = True

    def _build_custom_head(self):
        """Build enhanced classification head"""
        in_features = self.model.heads[-1].in_features
        
        # Multi-layer head with residual connection
        self.model.heads = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 4)
        )

    def _initialize_weights(self):
        """Proper weight initialization for head layers"""
        for layer in self.model.heads:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return self.dropout(x)


class ViT_GPU(ViT):
    """GPU-optimized version with mixed precision support"""
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.model = self.model.to(device)
        
        # Enable automatic mixed precision
        self.amp_enabled = True
        
        # Gradient scaling for mixed precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=self.amp_enabled):
            return super().forward(x.to(self.device))