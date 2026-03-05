import torch.nn as nn
from numpy.array_api import squeeze
from torch.ao.nn.qat import Conv2d


class Discriminator(nn.Module):
    def __init__(self,nfd,nc):
        super(Discriminator, self).__init__()
        self.main=nn.Sequential(
            #input_size:1x28x28
            nn.Conv2d(nc,nfd,4,2,1),
            nn.LeakyReLU(0.2, inplace= True),
            #input_size:32x14x14
            nn.Conv2d(nfd,nfd*2,4,2,1),
            nn.LeakyReLU(0.2, inplace= True),
            #input_size:64x7x7
            nn.Conv2d(nfd*2,1,7,1,0),
            #nn.Sigmoid() #可选
        )
    def forward(self, input):
        x = self.main(input)
        return x.view(x.size(0),-1)

class Generator(nn.Module):
    def __init__(self,in_channel, nfg, nc):
        super(Generator, self).__init__()
        self.main=nn.Sequential(
            #input_size=100x1x1
            nn.ConvTranspose2d(in_channel,nfg*2,7,1,0, bias=False),
            #nn.BatchNorm2d( nfg *2),
            nn.ReLU(),
            #input_size=64x7x7
            nn.ConvTranspose2d(nfg*2, nfg,4,2,1, bias=False),
            #nn.BatchNorm2d(nfg),
            nn.ReLU(),
            #input_size=32x14x14
            nn.ConvTranspose2d(nfg, nc, 4,2,1),
            nn.Tanh()
        )
    def forward(self,input):
        if input.dim() == 2:
            input = input.view(input.size(0), input.size(1), 1, 1)
        return self.main(input)