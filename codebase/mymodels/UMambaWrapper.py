import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add U-Mamba to sys.path
# This is now handled by the main training script
try:
    from umamba.nnunetv2.nets.UMambaEnc_2d import UMambaEnc
except ImportError as e:
    print("Could not import UMambaEnc. Make sure the U-Mamba repository is in the correct path and readable.")
    # The path is now set in the main script, so we don't try to manipulate it here.
    # We print the current path for debugging purposes.
    print(f"Current sys.path: {sys.path}")
    print(f"Original Error: {e}")
    # Re-raising the error is better than sys.exit() as it gives a full traceback
    raise

from typing import Tuple

class UMambaWrapper(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, patch_size: Tuple[int, int] = (64, 64)):
        super().__init__()
        
        n_stages = 5
        features_per_stage = [32, 64, 128, 256, 512]
        conv_op = nn.Conv2d
        kernel_sizes = [[3, 3]] * n_stages
        # First stage stride is 1, subsequent are 2 for downsampling
        strides = [[1, 1]] + [[2, 2]] * (n_stages - 1)
        n_conv_per_stage = [2] * n_stages
        n_conv_per_stage_decoder = [2] * (n_stages - 1)

        self.model = UMambaEnc(
            input_size=patch_size,
            input_channels=in_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            num_classes=out_channels,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=True,
            norm_op=nn.InstanceNorm2d,
            norm_op_kwargs={'eps': 1e-5, 'affine': True},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True},
            deep_supervision=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the UMambaWrapper.
        Args:
            x: Input tensor of shape (batch, height, width, channels_in)
        Returns:
            Output tensor of shape (batch, height, width, channels_out)
        """
        # Model expects: (batch, channels_in, height, width)
        x = x.permute(0, 3, 1, 2)
        
        logits = self.model(x)
        
        # Output shape from model: (batch, channels_out, height, width)
        # Permute to: (batch, height, width, channels_out)
        out = logits.permute(0, 2, 3, 1)
        
        return out

if __name__ == '__main__':
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'CPU'}")

    B, H, W = 8, 64, 64
    C_in, C_out = 2, 13
    x = torch.randn(B, H, W, C_in).to(device)

    print(f"\n--- Model Test ---")
    print(f"Input shape: {x.shape}")

    model = UMambaWrapper(
        in_channels=C_in,
        out_channels=C_out,
        patch_size=(H, W)
    ).to(device)

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
