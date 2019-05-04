import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class VAE(nn.Module):
    def __init__(self, d, flatten_img_size):
        super().__init__()
        self.d = d
        self.input_size = flatten_img_size
        self.inter_dim = 2048
        self.encoder = nn.Sequential(
            nn.Linear(flatten_img_size, self.inter_dim),
            nn.BatchNorm1d(self.inter_dim),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
	        nn.Linear(self.inter_dim, self.inter_dim),
            nn.BatchNorm1d(self.inter_dim),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.Linear(self.inter_dim, self.inter_dim),
            nn.BatchNorm1d(self.inter_dim),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.Linear(self.inter_dim, self.inter_dim),
            nn.BatchNorm1d(self.inter_dim),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.Linear(self.inter_dim, d * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(d, self.inter_dim),
            nn.BatchNorm1d(self.inter_dim),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.Linear(self.inter_dim, self.inter_dim),
            nn.BatchNorm1d(self.inter_dim),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.Linear(self.inter_dim, self.inter_dim),
            nn.BatchNorm1d(self.inter_dim),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.Linear(self.inter_dim, self.inter_dim),
            nn.BatchNorm1d(self.inter_dim),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.Linear(self.inter_dim, flatten_img_size),
            nn.Sigmoid()
        )

    def reparameterise(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu_logvar = self.encoder(x.view(-1, self.input_size)).view(-1, 2, self.d)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        z = self.reparameterise(mu, logvar)
        return self.decoder(z), mu, logvar

def vae_net(**kwargs):
    model = VAE(512, kwargs['flatten_img_size'])
    return model