import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt  # For visualizing input/reconstruction
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
#from pytorch_msssim import ms_ssim
from utils import ms_ssim
import seaborn as sns

class VAE_MP2(nn.Module):
    def __init__(self, img_channels=2, latent_dim=2, img_size=140):
        super(VAE_MP2, self).__init__()

        # **Encoder: Uses MaxPooling for Downsampling**
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=img_channels, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduces spatial size by half (140 → 70)
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 70 → 35
        )

        # Compute the flattened feature size after encoding
        self.flat_dim = 32 * (img_size // 4) * (img_size // 4)  # 64 x 35 x 35
        
        # Latent space representation (fully connected layers)
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)       # Mean (μ)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)   # Log variance (logσ²)

        # Linear layer to project back to feature space
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)

        # **Decoder: Uses ConvTranspose2D to Upsample**
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1), # 35 -> 70
            nn.ReLU(),  # Use Sigmoid if images are normalized to [0,1],
            
            nn.ConvTranspose2d(32, img_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # 70 -> 140
            nn.Sigmoid()  # Sigmoid to normalize reconstructed images to [0, 1] 
        )

    def encode(self, x):
        h = self.encoder(x)             # Shape: (batch, 32, 70, 70)
        h_flat = h.view(-1, self.flat_dim)   # Flatten
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  # Reparameterization trick

    def decode(self, z):
        h_flat = self.fc_decode(z)          # Map back to (batch, 32 * 70 * 70)
        h = h_flat.view(-1, 64, 35, 35)     # Reshape to feature map
        x_recon = self.decoder(h)      # Upsample to (batch, img_channels, 140, 140)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    

class VAE_BN2(nn.Module):
    def __init__(self, img_channels=2, latent_dim=2, img_size=140):
        super(VAE_BN2, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=3, stride=2, padding=1),  # 140 -> 70
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 70 -> 35
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Dynamically compute the flattened feature map size
        dummy_input = torch.zeros(1, img_channels, img_size, img_size)
        with torch.no_grad():
            dummy_output = self.encoder(dummy_input)
        self.flat_dim = dummy_output.view(1, -1).shape[1]
        self.encoder_output_shape = dummy_output.shape[1:]  # e.g. (64, 35, 35)

        # Latent space: fully connected layers
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)

        # Decoder: Two ConvTranspose2d layers to upsample back to original size
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=0),  # 35 -> 70
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1, output_padding=0),  # 70 -> 140
            nn.Sigmoid()  # Ensure output values are between 0 and 1
        )

    def encode(self, x):
        h = self.encoder(x)                          # h: (batch, 64, 35, 35)
        h_flat = h.view(-1, self.flat_dim)            # Flatten to (batch, flat_dim)
        mu = self.fc_mu(h_flat)                       # (batch, latent_dim)
        logvar = self.fc_logvar(h_flat)               # (batch, latent_dim)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h_flat = self.fc_decode(z)                    # (batch, flat_dim)
        # Reshape back to the feature map shape from the encoder
        h = h_flat.view(-1, *self.encoder_output_shape)  # e.g. (batch, 64, 35, 35)
        x_recon = self.decoder(h)                     # Upsample to (batch, img_channels, 140, 140)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    
class VAE_MP3(nn.Module):
    def __init__(self, img_channels=2, latent_dim=2, img_size=140):
        super(VAE_MP3, self).__init__()
        
        # ---------------------
        # Encoder: 3 Blocks
        # ---------------------
        self.encoder = nn.Sequential(
            # Block 1: 140x140 → 70x70
            nn.Conv2d(img_channels, 32, kernel_size=3, padding=1),  # 140x140
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # → 70x70
            
            # Block 2: 70x70 → 35x35
            nn.Conv2d(32, 64, kernel_size=3, padding=1),            # 70x70
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # → 35x35
            
            # Block 3: 35x35 → 17x17 (floor(35/2)=17)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),           # 35x35
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)                   # → 17x17
        )
        
        # Dynamically compute the flattened dimension and output shape from the encoder.
        dummy_input = torch.zeros(1, img_channels, img_size, img_size)
        with torch.no_grad():
            dummy_output = self.encoder(dummy_input)
        self.encoder_output_shape = dummy_output.shape[1:]          # e.g., (128, 17, 17)
        self.flat_dim = dummy_output.view(1, -1).shape[1]           # 128*17*17
        
        # ---------------------
        # Latent Space
        # ---------------------
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)
        
        # ---------------------
        # Decoder: 3 Blocks (using ConvTranspose2d)
        # ---------------------
        self.decoder = nn.Sequential(
            # Block 1: Upsample from 17x17 → 35x35
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            
            # Block 2: Upsample from 35x35 → 70x70
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            
            # Block 3: Upsample from 70x70 → 140x140
            nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.Sigmoid()  # Constrain output to [0, 1]
        )
        
    def encode(self, x):
        h = self.encoder(x)                   # h: (batch, 128, 17, 17)
        h_flat = h.view(-1, self.flat_dim)      # Flatten: (batch, flat_dim)
        mu = self.fc_mu(h_flat)               # (batch, latent_dim)
        logvar = self.fc_logvar(h_flat)       # (batch, latent_dim)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)         # Compute standard deviation
        eps = torch.randn_like(std)           # Sample from N(0,1)
        return mu + eps * std                 # Reparameterization trick
    
    def decode(self, z):
        h_flat = self.fc_decode(z)            # Map latent vector to flattened features
        h = h_flat.view(-1, *self.encoder_output_shape)  # Reshape to (batch, 128, 17, 17)
        x_recon = self.decoder(h)             # Upsample to (batch, img_channels, 140, 140)
        return x_recon
    
    def forward(self, x):
        mu, logvar = self.encode(x)           # Encode input
        z = self.reparameterize(mu, logvar)     # Sample latent vector
        recon_x = self.decode(z)              # Decode latent vector
        return recon_x, mu, logvar
    

class VAE_BN3(nn.Module):
    def __init__(self, img_channels=2, latent_dim=2, img_size=140):
        super(VAE_BN3, self).__init__()
        
        # ---------------------
        # Encoder: 3 Blocks
        # (Conv + BatchNorm + ReLU) with stride=2 for downsampling
        # ---------------------
        self.encoder = nn.Sequential(
            # Block 1: 140x140 → ~70x70
            nn.Conv2d(img_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Block 2: 70x70 → ~35x35
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Block 3: 35x35 → ~18x18
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Dynamically compute flattened dimension
        dummy_input = torch.zeros(1, img_channels, img_size, img_size)
        with torch.no_grad():
            dummy_output = self.encoder(dummy_input)
        self.encoder_output_shape = dummy_output.shape[1:]  # e.g., (128, 18, 18)
        self.flat_dim = dummy_output.view(1, -1).shape[1]   # 128 * 18 * 18 = 41472 (example)
        
        # ---------------------
        # Latent Space
        # ---------------------
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)
        
        # ---------------------
        # Decoder: 3 Blocks
        # (ConvTranspose + BatchNorm + ReLU) with stride=2 for upsampling
        # Because stride=2 doubles the spatial size each time, we might end up ~144x144
        # We'll add a final upsample to exactly 140x140
        # ---------------------
        self.decoder_blocks = nn.Sequential(
            # Block 1: ~18x18 → ~36x36
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Block 2: ~36x36 → ~72x72
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Block 3: ~72x72 → ~144x144
            nn.ConvTranspose2d(32, img_channels, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.Sigmoid()  # final activation to constrain output in [0, 1]
        )

        # Final upsampling layer to enforce exactly 140x140
        self.final_upsample = nn.Upsample(size=(140, 140), mode='bilinear', align_corners=False)
        
    def encode(self, x):
        """Encode the input image into latent parameters (mu, logvar)."""
        h = self.encoder(x)                          # e.g., (batch, 128, 18, 18)
        h_flat = h.view(-1, self.flat_dim)           # Flatten: (batch, flat_dim)
        mu = self.fc_mu(h_flat)                      # (batch, latent_dim)
        logvar = self.fc_logvar(h_flat)              # (batch, latent_dim)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Sample from the latent distribution using the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode the latent vector z back to an image."""
        h_flat = self.fc_decode(z)                   # (batch, flat_dim)
        h = h_flat.view(-1, *self.encoder_output_shape)  # e.g., (batch, 128, 18, 18)
        x_recon = self.decoder_blocks(h)             # e.g., ~144x144
        x_recon = self.final_upsample(x_recon)       # Force exactly 140x140
        return x_recon
    
    def forward(self, x):
        """Forward pass: encode → reparameterize → decode."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    
class VAE_BN4(nn.Module):
    def __init__(self, img_channels=2, latent_dim=2, img_size=140):
        super(VAE_BN4, self).__init__()
        
        # ---------------------
        # Encoder: 4 Blocks (Conv + BatchNorm + ReLU, stride=2)
        # ---------------------
        self.encoder = nn.Sequential(
            # Block 1: 140x140 -> 70x70
            nn.Conv2d(img_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Block 2: 70x70 -> 35x35
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Block 3: 35x35 -> ~18x18
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Block 4: ~18x18 -> ~9x9
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # Dynamically compute the flattened dimension and the encoder output shape.
        dummy_input = torch.zeros(1, img_channels, img_size, img_size)
        with torch.no_grad():
            dummy_output = self.encoder(dummy_input)
        self.encoder_output_shape = dummy_output.shape[1:]  # e.g. (256, 9, 9)
        self.flat_dim = dummy_output.view(1, -1).shape[1]
        
        # ---------------------
        # Latent Space
        # ---------------------
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)
        
        # ---------------------
        # Decoder: 4 Blocks (ConvTranspose + BatchNorm + ReLU)
        # ---------------------
        self.decoder = nn.Sequential(
            # Block 1: Upsample from 9x9 -> ~18x18
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Block 2: Upsample from ~18x18 -> ~35x35
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Block 3: Upsample from ~35x35 -> ~70x70
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Block 4: Upsample from ~70x70 -> ~140x140
            nn.ConvTranspose2d(32, img_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Constrain outputs to [0, 1]
        )
        
        # Optional: Final upsampling to ensure exactly 140x140
        self.final_upsample = nn.Upsample(size=(img_size, img_size), mode='bilinear', align_corners=True)
        
    def encode(self, x):
        h = self.encoder(x)                         # e.g., (batch, 256, 9, 9)
        h_flat = h.view(-1, self.flat_dim)            # Flatten: (batch, flat_dim)
        mu = self.fc_mu(h_flat)                       # (batch, latent_dim)
        logvar = self.fc_logvar(h_flat)               # (batch, latent_dim)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)                 # Standard deviation
        eps = torch.randn_like(std)                   # Sample noise
        return mu + eps * std                         # Reparameterization trick
    
    def decode(self, z):
        h_flat = self.fc_decode(z)                    # (batch, flat_dim)
        h = h_flat.view(-1, *self.encoder_output_shape)  # Reshape to (batch, 256, 9, 9)
        x_recon = self.decoder(h)                     # Upsample to approx. (batch, img_channels, 140, 140)
        x_recon = self.final_upsample(x_recon)        # Ensure exact output size
        return x_recon
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    
class VAE_MP4(nn.Module):
    def __init__(self, img_channels=2, latent_dim=2, img_size=140):
        super(VAE_MP4, self).__init__()
        
        # ---------------------
        # Encoder: 4 Blocks
        # ---------------------
        self.encoder = nn.Sequential(
            # Block 1: 140x140 → 70x70
            nn.Conv2d(img_channels, 32, kernel_size=3, padding=1),  # 140x140
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # → 70x70
            
            # Block 2: 70x70 → 35x35
            nn.Conv2d(32, 64, kernel_size=3, padding=1),            # 70x70
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # → 35x35
            
            # Block 3: 35x35 → 17x17
            nn.Conv2d(64, 128, kernel_size=3, padding=1),           # 35x35
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # → 17x17
            
            # Block 4: 17x17 → 8x8 (floor(17/2)=8)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),          # 17x17
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)                   # → 8x8
        )
        
        # Dynamically compute feature map size from the encoder
        dummy_input = torch.zeros(1, img_channels, img_size, img_size)
        with torch.no_grad():
            dummy_output = self.encoder(dummy_input)
        self.encoder_output_shape = dummy_output.shape[1:]          # (256, 8, 8)
        self.flat_dim = dummy_output.view(1, -1).shape[1]           # 256*8*8
        
        # ---------------------
        # Latent Space
        # ---------------------
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)
        
        # ---------------------
        # Decoder: 4 Blocks (using ConvTranspose2d)
        # ---------------------
        self.decoder = nn.Sequential(
            # Block 1: Upsample from 8x8 → 17x17
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            
            # Block 2: Upsample from 17x17 → 35x35
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            
            # Block 3: Upsample from 35x35 → 70x70
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            
            # Block 4: Upsample from 70x70 → 140x140
            nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.Sigmoid()  # Constrain output to [0, 1]
        )
        
    def encode(self, x):
        h = self.encoder(x)                   # h: (batch, 256, 8, 8)
        h_flat = h.view(-1, self.flat_dim)    # Flatten: (batch, flat_dim)
        mu = self.fc_mu(h_flat)               # (batch, latent_dim)
        logvar = self.fc_logvar(h_flat)       # (batch, latent_dim)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)         # Compute standard deviation
        eps = torch.randn_like(std)           # Sample from N(0,1)
        return mu + eps * std                 # Reparameterization trick
    
    def decode(self, z):
        h_flat = self.fc_decode(z)            # Map latent vector to flattened features
        h = h_flat.view(-1, *self.encoder_output_shape)  # Reshape to (batch, 256, 8, 8)
        x_recon = self.decoder(h)             # Upsample to (batch, img_channels, 140, 140)
        return x_recon
    
    def forward(self, x):
        mu, logvar = self.encode(x)           # Encode input
        z = self.reparameterize(mu, logvar)   # Sample latent vector
        recon_x = self.decode(z)              # Decode latent vector
        return recon_x, mu, logvar


class VAE_StrideConv4(nn.Module):
    def __init__(self, latent_dim=2):
        super(VAE_StrideConv4, self).__init__()

        # Encoder
        self.enc_conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.enc_conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.fc_mu = nn.Linear(128 * 9 * 9, latent_dim)
        self.fc_logvar = nn.Linear(128 * 9 * 9, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, 128 * 9 * 9)
        self.dec_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv4 = nn.ConvTranspose2d(16, 2, kernel_size=3, stride=2, padding=1, output_padding=1)

        # **Ensure Final Output is Exactly 140x140**
        #self.upsample = nn.Upsample(size=(140, 140), mode="bilinear", align_corners=True)  # Bilinear upsampling


    def encode(self, x):
        x = F.leaky_relu(self.enc_conv1(x))
        x = F.leaky_relu(self.enc_conv2(x))
        x = F.leaky_relu(self.enc_conv3(x))
        x = F.leaky_relu(self.enc_conv4(x))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_dec(z).view(-1, 128, 9, 9)
        x = F.leaky_relu(self.dec_conv1(x))
        x = F.leaky_relu(self.dec_conv2(x))
        x = F.leaky_relu(self.dec_conv3(x))
        x = torch.sigmoid(self.dec_conv4(x))
        
        # **Ensure the output size is exactly (140, 140)**
        x = F.interpolate(x, size=(140, 140), mode='bilinear', align_corners=True)
        
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar