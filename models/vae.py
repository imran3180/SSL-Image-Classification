import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class VAE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d

        self.encoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
	        nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, d * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(d, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.Sigmoid(),
        )

    def reparameterise(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu_logvar = self.encoder(x.view(-1, 1024)).view(-1, 2, self.d)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        z = self.reparameterise(mu, logvar)
        return self.decoder(z), mu, logvar

def vae_net(latent_size):
    model = VAE(latent_size)
    return model