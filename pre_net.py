"""
_________________________________________________________________________________________________
                                                                                                 |
Authors: * Ulrich Prestel    <Ulrich.Prestel@protonmail.com>                                     |
_________________________________________________________________________________________________|
"""

import torch
import torch.nn as nn


class Flatten(nn.Module):
    """
        Simple tensor flattening module
    """

    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    """
        Simple tensor unflattening module
    """

    def forward(self, input, size=64):
        return input.view(input.size(0), size, 1, 1)


class PREEncoder(nn.Module):
    """
        Encoder architecture for the velocities
        We have two channels: one for the x-component, one for the y-component
    """

    def __init__(self, nz=32, ndf=64):

        super(PREEncoder, self).__init__()
        self.nc = 2

        self.main = nn.Sequential(
            
            # input is (nc) x 64 x 64
            nn.Conv2d(self.nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #   (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            #   (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            #   (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            #   (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 64, 4, 1, 0, bias=False),
            Flatten(),
            nn.Linear(64, nz)
        )

    def forward(self, x):
        for layer in self.main:
            #print(x.shape, layer)
            x = layer(x)
        return x


class PREDecoder(nn.Module):
    def __init__(self, nz=32, ngf=64, ndf=64, h_dim=128):
        """
        :param nz: the dimension of the latent space
        :param ngf: the decoder (generator) filter factor
        """
        super(PREDecoder, self).__init__()
        self.nc = 2

        self.main = nn.Sequential(

            nn.Linear(nz, 64),
            UnFlatten(),
            # input is Z, going into a convolution
            nn.ConvTranspose2d(64, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            #   (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            #   (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            #   (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            #   (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, self.nc, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
            #   (nc) x 64 x 64
        )

    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        return x


class PREDiscriminator(nn.Module):
    """
        The discriminator architecture for the velocity autoencoder
    """

    def __init__(self, nc=2, nz=32, ngf=64, ndf=64, h_dim=128):
        super(PREDiscriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(nz, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        for layer in self.main:
            # print("input: ", x.shape, "with ", nz, h_dim)
            x = layer(x)
        return x

if __name__ == "__main__":
    # simple dimension debugging
    Q_pre = PREEncoder()
    P_pre = PREDecoder()

    x = torch.randn(4, 2, 64, 64)
    z = Q_pre(x)
    y = P_pre(z)
    print(x.shape, z.shape, y.shape)
