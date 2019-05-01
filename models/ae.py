import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class AE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d

        self.encoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, d)
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
        z = self.encoder(x.view(-1, 1024)).view(-1, self.d)
        return self.decoder(z)

def ae_net(**kwargs):
    model = AE(20)
    return model
