import torch
import torch.nn as nn
import sys
from pathlib import Path
import math
########################  IMPORTANT ########################
# --- Path setup for pytorch-CycleGAN-and-pix2pix ---
# This wrapper assumes it's located you cloned the original Pix2Pix source code and it is in the same directory as the wrapper and both are saved in a package called mymodels 
# (you can name it whatever you want, but you will then have to change a lot of paths across many scripts. Good luck :) )
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# Or simply unzip the mymodels.zip file 
#
# Note: The original implementation exhibited some GAN-related issues (slow convergence, checkerboard artifacts).
# To overcome these issues, some "minor" modifications were done to the original implementaion
# To apply the same modifications, refer to the thesis document (Sec:  Challenges Due to Inherent Model Characteristics)
try:
    gan_repo_path = Path(__file__).parent / 'pytorch-CycleGAN-and-pix2pix'
    if gan_repo_path.is_dir():
        sys.path.insert(0, str(gan_repo_path))
    else:
        # Fallback for different execution environments
        sys.path.insert(0, './pytorch-CycleGAN-and-pix2pix')

    from models.networks import UnetGenerator, get_norm_layer, init_net
except ImportError as e:
    print("Could not import from the pix2pix model.")
    print("Please ensure the 'pytorch-CycleGAN-and-pix2pix' repository is correctly placed.")
    print(f"Attempted to add to path: {gan_repo_path}")
    print(f"Current sys.path: {sys.path}")
    print(f"Error: {e}")
    raise

class Pix2PixWrapper(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, img_size: int = 256):
        super().__init__()
        
        # --- Generator Configuration ---
        # The 'unet_256' architecture is too deep for 64x64 images, causing an error.
        # We need a U-Net with fewer downsampling layers. For a 64x64 input, 6
        # downsampling layers will result in a 1x1 feature map at the bottleneck.
        num_downs = int(math.log2(img_size))
        print("num_downs: ", num_downs)
        ngf = 64 # the number of filters in the last conv layer
        norm_layer = get_norm_layer(norm_type='batch')
        
        # We instantiate the UnetGenerator directly to control the 'num_downs' parameter.
        # We also call 'init_net' to apply the weight initialization as the original script does.
        generator = UnetGenerator(
            input_nc=in_channels,
            output_nc=out_channels,
            num_downs=num_downs,
            ngf=ngf,
            norm_layer=norm_layer,
            up_mode="resize_conv",
            use_dropout=False
        )

        self.model = init_net(generator, init_type='normal', init_gain=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Pix2PixWrapper.
        Args:
            x: Input tensor of shape (batch, height, width, channels_in)
        Returns:
            Output tensor of shape (batch, height, width, channels_out)
        """
        # Model expects: (batch, channels_in, height, width)
        x = x.permute(0, 3, 1, 2)
        
        # The generator's output is also (batch, channels_out, height, width)
        logits = self.model(x)
        
        # Permute to: (batch, height, width, channels_out)
        out = logits.permute(0, 2, 3, 1)
        
        return out

if __name__ == '__main__':
    # Test the forward pass
    batch_size = 2
    height = 256
    width = 256
    channels_in = 2   # Example: SAR data (e.g., VV, VH)
    channels_out = 13 # Example: Multispectral output

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        wrapper = Pix2PixWrapper(
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
        print("Could not run test because modules could not be imported from pix2pix.")
    except Exception as e:
        print(f"An error occurred during the test run: {e}")
