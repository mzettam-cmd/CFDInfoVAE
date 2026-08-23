"""
The architecture for InfoVAE 
(adapted from beta-VAE)
"""

import torch 
from torch import nn 


class InfoVAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim
        self.encoder = self.buildEncoder(latent_dim)
        self.decoder = self.buildDecoder(latent_dim)

    def buildEncoder(self, latent_dim):
        encoder = nn.Sequential(

            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.ConstantPad3d((0, 1, 0, 0), 0),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.ConstantPad3d((0, 0, 0, 1), 0),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.ConstantPad3d((0, 1, 0, 0), 0),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.ConstantPad3d((0, 0, 0, 1), 0),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            nn.Flatten(start_dim=1, end_dim=-1),

            nn.Linear(2560, 256),
            nn.ELU(),

            nn.Linear(256, latent_dim * 2),
        )
        return encoder


    def buildDecoder(self, latent_dim):
        decoder = nn.Sequential(

            nn.Linear(latent_dim, 256),
            nn.ELU(),

            nn.Linear(256, 256 * 5 * 2),
            nn.ELU(),

            nn.Unflatten(dim=1, unflattened_size=(256, 2, 5)),

            nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1),
            nn.ConstantPad2d((0, 0, 0, -1), 0),
            nn.ELU(),

            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1),
            nn.ConstantPad2d((0, -1, 0, 0), 0),
            nn.ELU(),

            nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1),
            nn.ConstantPad2d((0, 0, 0, -1), 0),
            nn.ELU(),

            nn.ConvTranspose2d(32, 16, 3, 2, 1, output_padding=1),
            nn.ConstantPad2d((0, -1, 0, 0), 0),
            nn.ELU(),

            nn.ConvTranspose2d(16, 8, 3, 2, 1, output_padding=1),
            nn.ConstantPad2d((0, 0, 0, 0), 0),
            nn.ELU(),

            nn.ConvTranspose2d(8, 2, 3, 2, 1, output_padding=1),
        )
        return decoder


    def sample(self, mean, logvariance):
        """
        Reparameterization trick (correct version)
        """
        std = torch.exp(0.5 * logvariance)

        # 🔥 IMPORTANT: normal distribution (pas uniforme)
        epsilon = torch.randn_like(std)

        return mean + epsilon * std


    def forward(self, data):

        mean_logvariance = self.encoder(data)

        mean, logvariance = torch.chunk(mean_logvariance, 2, dim=1)

        z = self.sample(mean, logvariance)

        reconstruction = self.decoder(z)

        #   DIFFÉRENCE CLÉ InfoVAE
        return reconstruction, mean, logvariance, z