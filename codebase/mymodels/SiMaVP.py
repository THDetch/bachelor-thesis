"""
SiMaVP: A High-Performance U-Net with a Mamba Bottleneck

Architecture Overview:
This model is a novel, state-of-the-art architecture for image-to-image
translation, designed to balance high capacity with computational efficiency.
It fuses two powerful concepts: the classic, parallelizable U-Net architecture
and the recent, sequential Mamba state space model.

Key Design Principles:
1.  **U-Net Backbone:** The model uses a standard, high-performance U-Net with a
    convolutional encoder and decoder. This ensures the bulk of the feature
    extraction and reconstruction is fast and GPU-friendly.
2.  **Mamba as Bottleneck:** The novelty lies in its bottleneck. Instead of a
    standard convolutional block, it uses a BidirectionalMambaBlock. This
    strategically places the powerful sequential processing of Mamba at the
    point where the feature map is smallest, allowing the model to capture
    global context and long-range dependencies with minimal performance impact.
3.  **Hybrid Approach:** Unlike other models that might use Mamba to enhance or
    filter features, this architecture uses Mamba as the fundamental structural
    bridge between the encoder and decoder.

This design results in a model that is both powerful (with a parameter count
around 26M) and extremely fast, making it suitable for demanding image-to-image
tasks.

Suggested Names:
Given the new architecture, more descriptive names could be:
- SiMa-UNet: Retains the original name while clarifying the U-Net structure.
- MambaBridge-UNet: Highlights the unique role of the Mamba bottleneck.
- MambaUNet: A classic and descriptive name for this hybrid architecture.
"""
import torch
import torch.nn as nn
from mamba_ssm import Mamba
import time

class ConvBlock(nn.Module):
    """A standard convolutional block with two convolutions."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.gn1 = nn.GroupNorm(8, out_channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.gn2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.GELU()

    def forward(self, x):
        x = self.act1(self.gn1(self.conv1(x)))
        x = self.act2(self.gn2(self.conv2(x)))
        return x

class BidirectionalMambaBlock(nn.Module):
    """The Mamba block for the bottleneck, now simplified."""
    def __init__(self, d_model, d_state=32, d_conv=4, expand=2):
        super().__init__()
        self.forward_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.backward_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.fusion = nn.Linear(2 * d_model, d_model)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).transpose(1, 2) # (B, H*W, C)
        
        forward_out = self.forward_mamba(x_flat)
        backward_out = self.backward_mamba(torch.flip(x_flat, dims=[1]))
        backward_out = torch.flip(backward_out, dims=[1])
        
        combined = torch.cat([forward_out, backward_out], dim=-1)
        fused = self.fusion(combined)
        
        return fused.transpose(1, 2).view(B, C, H, W)

class SiMaVP(nn.Module):
    """
    A redesigned SiMaVP with a U-Net architecture and a Mamba bottleneck.
    This is designed to be fast, powerful, and efficient.
    """
    def __init__(self, in_channels, out_channels, base_channels=96):
        super().__init__()
        
        # --- Encoder ---
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)
        
        self.downsample = nn.MaxPool2d(2)

        # --- Mamba Bottleneck ---
        self.bottleneck = BidirectionalMambaBlock(d_model=base_channels * 8, d_state=32)

        # --- Decoder ---
        self.up4 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(base_channels * 8, base_channels * 4) # 4 + 4
        
        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_channels * 4, base_channels * 2) # 2 + 2
        
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 2, base_channels) # 1 + 1

        # --- Final Output ---
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.GroupNorm):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # Expected input: (B, H, W, C) -> permute to (B, C, H, W)
        x = x.permute(0, 3, 1, 2)

        # --- Encoder Path ---
        skip1 = self.enc1(x)
        x = self.downsample(skip1)
        
        skip2 = self.enc2(x)
        x = self.downsample(skip2)
        
        skip3 = self.enc3(x)
        x = self.downsample(skip3)
        
        x = self.enc4(x)

        # --- Bottleneck ---
        x = self.bottleneck(x)

        # --- Decoder Path ---
        x = self.up4(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.dec4(x)

        x = self.up3(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.dec2(x)
        
        # --- Final Output ---
        output = self.final_conv(x)
        
        # Permute back to (B, H, W, C_out)
        return output.permute(0, 2, 3, 1)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'CPU'}")
    
    B, H, W = 8, 64, 64
    C_in, C_out = 2, 13
    x = torch.randn(B, H, W, C_in).to(device)
    
    print(f"\n--- Model Test ---")
    print(f"Input shape: {x.shape}")
    
    model = SiMaVP(in_channels=C_in, out_channels=C_out).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params/1e6:.2f}M")

    # --- Profiling ---
    print("\n--- Performance Profiling ---")
    model.eval()
    num_runs = 20
    times = []
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)
            
    # Timed runs
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start_time = time.time()
            _ = model(x)
            torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)
    
    avg_inference_time = sum(times) / len(times)
    print(f"Average forward pass time: {avg_inference_time*1000:.2f} ms")

    output = model(x)
    print(f"\nOutput shape: {output.shape}")
    expected_shape = (B, H, W, C_out)
    print(f"Expected output shape: {expected_shape}")
    assert output.shape == expected_shape, "Output shape mismatch!"
    print("--- Test Successful ---")