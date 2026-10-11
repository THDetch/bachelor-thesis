import torch
import torch.nn as nn
import sys
from pathlib import Path
import argparse

# --- Path setup for Swin-Unet ---
try:
    swin_repo_path = Path(__file__).parent / 'Swin-Unet'
    if swin_repo_path.is_dir():
        sys.path.insert(0, str(swin_repo_path))
    else:
        # Fallback for different execution environments
        sys.path.insert(0, './Swin-Unet')

    from networks.vision_transformer import SwinUnet
    from config import get_config
except ImportError as e:
    print("Could not import from the Swin-Unet model.")
    print("Please ensure the 'Swin-Unet' repository is correctly placed.")
    print(f"Attempted to add to path: {swin_repo_path}")
    print(f"Current sys.path: {sys.path}")
    print(f"Error: {e}")
    raise

class SwinUnetWrapper(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, img_size: int = 64):
        super().__init__()

        # --- Model Configuration ---
        # Create a mock args object to satisfy the config parser's requirements.
        # This is necessary because get_config expects an object with specific attributes
        # that would normally come from command-line arguments.
        mock_args = argparse.Namespace()
        mock_args.cfg = str(swin_repo_path / 'configs' / 'swin_tiny_patch4_window7_224_lite.yaml')
        mock_args.opts = []
        mock_args.batch_size = 0  # We let the config file define the batch size
        mock_args.zip = False
        mock_args.cache_mode = 'part'
        mock_args.resume = ''
        mock_args.accumulation_steps = 0
        mock_args.use_checkpoint = False
        mock_args.amp_opt_level = ''
        mock_args.tag = ''
        mock_args.eval = False
        mock_args.throughput = False
        
        config = get_config(mock_args)

        # Update config with our parameters
        config.defrost()
        # If we have a single channel, we will repeat it to 3 channels to match the model's expectation.
        config.MODEL.SWIN.IN_CHANS = 3 if in_channels == 1 else in_channels
        config.MODEL.NUM_CLASSES = out_channels
        config.DATA.IMG_SIZE = img_size
        # The Swin Transformer partitions the image into non-overlapping windows.
        # The size of the feature map at the bottleneck must be divisible by the window size.
        # With a patch size of 4, a 64x64 image gives a 16x16 feature map, and 128x128 gives 32x32.
        # The default window size of 7 is not compatible. We set it to 4, which works for both.
        config.MODEL.SWIN.WINDOW_SIZE = 4
        config.freeze()

        self.model = SwinUnet(config, img_size=img_size, num_classes=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SwinUnetWrapper.
        Args:
            x: Input tensor of shape (batch, height, width, channels_in)
        Returns:
            torch.Tensor: Output tensor of shape (N, H, W, out_channels)
        """
        # If the input is single-channel, the model might internally
        # expand it to 3 channels. To prevent this, we explicitly
        # repeat the single channel to create a 3-channel input.
        if x.shape[3] == 1:
            x = x.repeat(1, 1, 1, 3)

        # Permute input from (N, H, W, C) to (N, C, H, W)
        x = x.permute(0, 3, 1, 2)

        logits = self.model(x)
        
        # Permute output back to (N, H, W, C)
        logits = logits.permute(0, 2, 3, 1)
        
        return logits

if __name__ == '__main__':
    # Test the forward pass
    batch_size = 2
    height = 64
    width = 64
    channels_in = 2   # Example: SAR data (e.g., VV, VH)
    channels_out = 13 # Example: Multispectral output

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        wrapper = SwinUnetWrapper(
            in_channels=channels_in,
            out_channels=channels_out,
            img_size=height
        ).to(device)
        wrapper.eval() # Set model to evaluation mode for the test

        # Calculate total parameters
        total_params = sum(p.numel() for p in wrapper.parameters())
        print(f"Total model parameters: {total_params:,}")

        # Create a random tensor
        input_tensor = torch.randn(batch_size, height, width, channels_in).to(device)
        print(f"Input tensor shape: {input_tensor.shape}")

        # Forward pass
        with torch.no_grad():
            output_tensor = wrapper(input_tensor)
        
        print(f"Output tensor shape: {output_tensor.shape}")

        # Check if the output shape is as expected
        expected_shape = (batch_size, height, width, channels_out)
        assert output_tensor.shape == expected_shape, \
            f"Shape mismatch! Expected {expected_shape}, but got {output_tensor.shape}"

        print("Forward pass test successful!")

    except NameError:
        print("Could not run test because modules could not be imported from Swin-Unet.")
    except Exception as e:
        print(f"An error occurred during the test run: {e}")
