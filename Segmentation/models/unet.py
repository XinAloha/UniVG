"""
U-Net: Convolutional Networks for Biomedical Image Segmentation

Reference:
    Ronneberger et al., 2015
    https://arxiv.org/abs/1505.04597
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetConvBlock(nn.Module):
    """Double convolution block for U-Net."""
    
    def __init__(self, in_channels, out_channels, padding=True, batch_norm=False):
        super().__init__()
        
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=int(padding)))
        layers.append(nn.ReLU(inplace=True))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=int(padding)))
        layers.append(nn.ReLU(inplace=True))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class UNetUpBlock(nn.Module):
    """Upsampling block for U-Net decoder."""
    
    def __init__(self, in_channels, out_channels, up_mode='upconv', padding=True, batch_norm=False):
        super().__init__()
        
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        else:  # upsample
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2, align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
            )
        
        self.conv_block = UNetConvBlock(in_channels, out_channels, padding, batch_norm)
    
    def center_crop(self, layer, target_size):
        """Center crop layer to match target size."""
        _, _, h, w = layer.size()
        diff_y = (h - target_size[0]) // 2
        diff_x = (w - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]
    
    def forward(self, x, skip):
        up = self.up(x)
        crop = self.center_crop(skip, up.shape[2:])
        out = torch.cat([up, crop], dim=1)
        return self.conv_block(out)


class UNet(nn.Module):
    """
    U-Net architecture for image segmentation.
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        n_classes: Number of output classes (default: 2 for binary segmentation)
        depth: Depth of the network (default: 5)
        wf: Number of filters in first layer is 2**wf (default: 6 -> 64 filters)
        padding: If True, apply padding to maintain input/output size
        batch_norm: If True, use batch normalization
        up_mode: 'upconv' for transposed conv, 'upsample' for bilinear upsampling
    """
    
    def __init__(
        self,
        in_channels=3,
        n_classes=2,
        depth=5,
        wf=6,
        padding=True,
        batch_norm=False,
        up_mode='upconv',
    ):
        super().__init__()
        
        assert up_mode in ('upconv', 'upsample')
        
        self.padding = padding
        self.depth = depth
        
        # Encoder (downsampling path)
        self.down_path = nn.ModuleList()
        prev_channels = in_channels
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)
        
        # Decoder (upsampling path)
        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)
        
        # Final 1x1 convolution
        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)
    
    def forward(self, x):
        blocks = []
        
        # Encoder
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)
        
        # Decoder
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])
        
        return self.last(x)


if __name__ == "__main__":
    # Test the model
    model = UNet(in_channels=3, n_classes=2, padding=True, batch_norm=False)
    x = torch.randn(1, 3, 512, 512)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params / 1e6:.2f}M")
