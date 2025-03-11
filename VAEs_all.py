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

def compute_final_size(input_size, num_layers):
    size = input_size
    for _ in range(num_layers):
        size = (size - 1) // 2 + 1
    return size

class VAE_MP2(nn.Module):
    def __init__(self, img_channels=2, latent_dim=2, img_size=140):
        super(VAE_MP2, self).__init__()
        # Manually set the final spatial size after 2 pooling layers:
        # 140 → 70 → 35
        self.final_size = img_size // 4  # 140 // 4 = 35
        # Encoder will output 32 channels
        self.flat_dim = 64 * (self.final_size ** 2)  # 32 * 35 * 35 = 39200

        # Encoder: Two blocks using MaxPooling
        self.encoder = nn.Sequential(
            # Block 1: 140x140 → 70x70
            nn.Conv2d(in_channels=img_channels, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 140 → 70

            # Block 2: 70x70 → 35x35
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 70 → 35
        )

        # Latent space fully-connected layers
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)

        # Decoder: Upsample using ConvTranspose2d
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, 
                               kernel_size=3, stride=2, padding=1, output_padding=1),  # 35 → 70
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=img_channels, 
                               kernel_size=3, stride=2, padding=1, output_padding=1),  # 70 → 140
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)  # Expected shape: (batch, 32, 35, 35)
        h_flat = h.view(x.size(0), -1)  # Flatten
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h_flat = self.fc_decode(z)  # (batch, 39200)
        h = h_flat.view(-1, 64, self.final_size, self.final_size)  # Reshape to (batch, 32, 35, 35)
        x_recon = self.decoder(h)  # Upsample to (batch, img_channels, 140, 140)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    
class VAE_BN2(nn.Module):
    def __init__(self, img_channels=2, latent_dim=2, img_size=140):
        super(VAE_BN2, self).__init__()
        # With 2 stride=2 conv layers, final spatial size: 140 → 70 → 35.
        self.final_size = 35
        # Encoder outputs 64 channels.
        self.flat_dim = 64 * (self.final_size ** 2)  # 64 * 35 * 35 = 78400

        # Encoder: Two Blocks with strided convolutions and BatchNorm
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=3, stride=2, padding=1),  # 140 → 70
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 70 → 35
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Latent space fully-connected layers
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)

        # Decoder: Two Blocks using ConvTranspose2d and BatchNorm
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=0),  # 35 → 70
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1, output_padding=0),  # 70 → 140
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)  # Expected shape: (batch, 64, 35, 35)
        h = h.view(x.size(0), -1)  # Flatten: (batch, 78400)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)  # (batch, 78400)
        h = h.view(-1, 64, self.final_size, self.final_size)  # Reshape to (batch, 64, 35, 35)
        x_recon = self.decoder(h)  # Upsample to (batch, img_channels, 140, 140)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    
class VAE_MP3(nn.Module):
    def __init__(self, img_channels=2, latent_dim=2, img_size=140):
        super(VAE_MP3, self).__init__()

        # Manually determine the final spatial size after 3 pooling layers:
        # 140 → 70 → 35 → 17 (since 35 // 2 = 17)
        self.final_size = img_size // 8  # 140 // 8 = 17
        # Encoder will output 128 channels at 17×17
        self.flat_dim = 128 * (self.final_size ** 2)  # 128 * 17 * 17 = 36992

        # ---------------------
        # Encoder: 3 Blocks (Conv + MaxPool)
        # ---------------------
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=3, padding=1),  # 140×140
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # → 70×70
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 70×70
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # → 35×35
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 35×35
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # → 17×17
        )

        # ---------------------
        # Latent Space (FC Layers)
        # ---------------------
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)

        # ---------------------
        # Decoder: 3 Blocks (ConvTranspose2D)
        # ---------------------
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=1),  # 17 → 35
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=0),  # 35 → 70
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1, output_padding=0),  # 70 → 140
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)  # Expected shape: (batch, 128, 17, 17)
        h_flat = h.view(x.size(0), -1)  # Flatten: (batch, 36992)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  # Reparameterization trick

    def decode(self, z):
        h_flat = self.fc_decode(z)  # (batch, 36992)
        h = h_flat.view(-1, 128, self.final_size, self.final_size)  # Reshape to (batch, 128, 17, 17)
        x_recon = self.decoder(h)  # Upsample to (batch, img_channels, 140, 140)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    

class VAE_BN3(nn.Module):
    def __init__(self, img_channels=2, latent_dim=2, img_size=140):
        super(VAE_BN3, self).__init__()

        # **Final output size before flattening**
        self.final_size = 18  # Manually set to match actual downsampling
        self.flat_dim = 128 * (self.final_size ** 2)  # 128 * 18 * 18 = 41472

        # ---------------------
        # **Encoder: 3 Blocks**
        # ---------------------
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=3, stride=2, padding=1),  # 140 → 70
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 70 → 35
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 35 → 18
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # ---------------------
        # **Latent Space (Fully Connected Layers)**
        # ---------------------
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)

        # ---------------------
        # **Decoder: 3 Blocks**
        # ---------------------
        self.decoder_blocks = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 18 → 36
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 36 → 72
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, img_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # 72 → 144
            nn.Sigmoid()  # Final activation to constrain output in [0, 1]
        )

        # Final upsampling layer to **force exactly 140×140**
        self.final_upsample = nn.Upsample(size=(140, 140), mode='bilinear', align_corners=False)

    def encode(self, x):
        """Encode the input image into latent parameters (mu, logvar)."""
        h = self.encoder(x)  # Expected shape: (batch, 128, 18, 18)
        h_flat = h.view(x.size(0), -1)  # Flatten to match `self.flat_dim`

        # Debugging
        print(f"Flattened feature shape: {h_flat.shape}")  # Should be (batch_size, 41472)

        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Sample from the latent distribution using the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  # Reparameterization trick

    def decode(self, z):
        """Decode the latent vector z back to an image."""
        h_flat = self.fc_decode(z)  # (batch, flat_dim)
        h = h_flat.view(-1, 128, self.final_size, self.final_size)  # Reshape to (batch, 128, 18, 18)
        x_recon = self.decoder_blocks(h)  # Upsample to (batch, img_channels, 144, 144)
        x_recon = self.final_upsample(x_recon)  # Force exactly 140×140
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
        
        # Manually calculated final feature map size:
        # For input 140, using 4 layers with kernel=3, stride=2, padding=1:
        # Layer1: 140 → 70, Layer2: 70 → 35, Layer3: 35 → 18, Layer4: 18 → 9.
        self.final_size = 9  
        self.flat_dim = 256 * (self.final_size ** 2)  # 256 * 9 * 9 = 20736

        # ---------------------
        # Encoder: 4 Blocks (Conv + BatchNorm + ReLU, stride=2)
        # ---------------------
        self.encoder = nn.Sequential(
            # Block 1: 140x140 → 70x70
            nn.Conv2d(img_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Block 2: 70x70 → 35x35
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Block 3: 35x35 → 18x18
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Block 4: 18x18 → 9x9
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
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
            # Block 1: Upsample from 9x9 → 18x18
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Block 2: Upsample from 18x18 → 35x35
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Block 3: Upsample from 35x35 → 70x70
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Block 4: Upsample from 70x70 → 140x140
            nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.Sigmoid()  # Constrain outputs to [0, 1]
        )
        
        self.final_upsample = nn.Upsample(size=(img_size, img_size), mode='bilinear', align_corners=True)


    def encode(self, x):
        # Pass input through encoder: expected shape (batch, 256, 9, 9)
        h = self.encoder(x)
        # Flatten with correct batch size:
        h_flat = h.reshape(x.size(0), -1)  # (batch, 20736)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # Map latent vector back to flattened feature map
        h_flat = self.fc_decode(z)  # (batch, 20736)
        # Reshape to (batch, 256, 9, 9)
        h = h_flat.view(-1, 256, self.final_size, self.final_size)
        # Decode (upsample) to (batch, img_channels, 140, 140)
        x_recon = self.decoder(h)
        x_recon = self.final_upsample(x_recon)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    
class VAE_MP4(nn.Module):
    def __init__(self, img_channels=2, latent_dim=2, img_size=140):
        super(VAE_MP4, self).__init__()

        # Calculate final feature map size after downsampling (140 → 8)
        self.final_size = img_size // 16  # 140 → 70 → 35 → 17 → 8
        self.flat_dim = 256 * self.final_size * self.final_size  # 256 * 8 * 8

        # ---------------------
        # Encoder: 4 Blocks with MaxPooling
        # ---------------------
        self.encoder = nn.Sequential(
            # Block 1: 140x140 → 70x70
            nn.Conv2d(img_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  

            # Block 2: 70x70 → 35x35
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  

            # Block 3: 35x35 → 17x17
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  

            # Block 4: 17x17 → 8x8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  
        )

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
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            # Block 4: Upsample from 70x70 → 128x128
            nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )

        # Final Upsample to **force output to 140x140**
        self.final_upsample = nn.Upsample(size=(img_size, img_size), mode='bilinear', align_corners=True)

    def encode(self, x):
        """Encode the input image into latent parameters (mu, logvar)."""
        h = self.encoder(x)  # (batch, 256, 8, 8)
        h_flat = h.view(-1, self.flat_dim)  # Flatten to (batch, 256 * 8 * 8)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Sample from the latent distribution using the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode the latent vector z back to an image."""
        h_flat = self.fc_decode(z)  # (batch, 256 * 8 * 8)
        h = h_flat.view(-1, 256, self.final_size, self.final_size)  # Reshape to (batch, 256, 8, 8)
        x_recon = self.decoder(h)  # Upsample to (~128, ~128)
        x_recon = self.final_upsample(x_recon)  # 🔹 Ensure exactly (batch, 2, 140, 140)
        return x_recon

    def forward(self, x):
        """Forward pass: encode → reparameterize → decode."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
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
    

class VAE_StridedBN4(nn.Module):
    def __init__(self, img_channels=2, latent_dim=20, img_size=140):
        super(VAE_StridedBN4, self).__init__()
        
        # ---------------------
        # Encoder: 4 Blocks using strided convolution instead of maxpooling
        # With kernel_size=3, stride=2, padding=1, each layer roughly halves the spatial dimension.
        # 140 -> 70 -> 35 -> 18 -> 9 (approximately)
        # Final feature map will be (256, 9, 9)
        self.final_size = 9  
        self.flat_dim = 256 * self.final_size * self.final_size  # 256*9*9
        
        self.encoder = nn.Sequential(
            # Block 1: 140x140 -> 70x70
            nn.Conv2d(img_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Block 2: 70x70 -> 35x35
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Block 3: 35x35 -> 18x18
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Block 4: 18x18 -> 9x9
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # ---------------------
        # Latent Space
        # ---------------------
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)
        
        # ---------------------
        # Decoder: Mirror the encoder using ConvTranspose2d layers
        # The decoder will upsample from 9x9 back to ~128x128, then a final upsample to force 140x140
        self.decoder = nn.Sequential(
            # Block 1: 9x9 -> ~18x18
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Block 2: ~18x18 -> ~36x36
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Block 3: ~36x36 -> ~72x72
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Block 4: ~72x72 -> ~144x144
            nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(img_channels),
            nn.Sigmoid()  # Constrain outputs to [0, 1]
        )
        
        # Final upsampling to force exactly 140x140 (if needed)
        self.final_upsample = nn.Upsample(size=(img_size, img_size), mode='bilinear', align_corners=False)
    
    def encode(self, x):
        h = self.encoder(x)  # Expected shape: (batch, 256, 9, 9)
        h_flat = h.view(h.size(0), -1)  # Flatten to (batch, 256*9*9)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h_flat = self.fc_decode(z)  # (batch, 256*9*9)
        h = h_flat.view(-1, 256, self.final_size, self.final_size)  # Reshape to (batch, 256, 9, 9)
        x_recon = self.decoder(h)  # Upsample to approximately (batch, img_channels, 144, 144)
        x_recon = self.final_upsample(x_recon)  # Force to exactly (batch, img_channels, 140, 140)
        return x_recon
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    

class VAE_BN_MP4(nn.Module):
    def __init__(self, img_channels=2, latent_dim=2, img_size=140):
        super(VAE_BN_MP4, self).__init__()

        # ---------------------
        # Calculate final spatial size
        # ---------------------
        # Using 4 blocks of MaxPool2d(kernel_size=2, stride=2):
        # 140 → 70 → 35 → 17 → 8 (floor division in PyTorch).
        self.final_size = img_size // 16  # Should be 8 if 140 // 16 = 8
        self.flat_dim = 256 * self.final_size * self.final_size  # 256 * 8 * 8 = 16384

        # ---------------------
        # Encoder: 4 Blocks (Conv + BN + ReLU + MaxPool)
        # ---------------------
        self.encoder = nn.Sequential(
            # Block 1: 140x140 → 70x70
            nn.Conv2d(img_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 70x70 → 35x35
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 35x35 → ~17x17
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 17x17 → ~8x8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # ---------------------
        # Latent Space
        # ---------------------
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)

        # ---------------------
        # Decoder: 4 Blocks (ConvTranspose + BN + ReLU)
        # ---------------------
        # We'll upsample from 8×8 back to ~128×128, then do a final upsample to 140×140
        self.decoder = nn.Sequential(
            # Block 1: 8x8 → ~16x16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Block 2: ~16x16 → ~32x32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Block 3: ~32x32 → ~64x64
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Block 4: ~64x64 → ~128x128
            nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(img_channels),
            # If you want [0,1] outputs, use Sigmoid:
            nn.Sigmoid()
        )

        # Force final output to exactly 140x140
        self.final_upsample = nn.Upsample(size=(img_size, img_size), mode='bilinear', align_corners=False)

    def encode(self, x):
        """Encode the input image into latent parameters (mu, logvar)."""
        h = self.encoder(x)  # (batch, 256, 8, 8) if dimension math is correct
        h_flat = h.view(h.size(0), -1)  # (batch, 256*8*8)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Sample from the latent distribution using the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode the latent vector z back to an image."""
        h_flat = self.fc_decode(z)  # (batch, 256*8*8)
        h = h_flat.view(-1, 256, self.final_size, self.final_size)  # (batch, 256, 8, 8)
        x_recon = self.decoder(h)   # ~ (batch, 2, 128, 128)
        x_recon = self.final_upsample(x_recon)  # (batch, 2, 140, 140)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    
