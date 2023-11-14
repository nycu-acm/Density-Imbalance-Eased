# from model.atlasnet import Atlasnet
# from model.model_blocks import PointNet
import torch.nn as nn
# import model.resnet as resnet
import importlib
from model.dense_encoder import Pct
from model.decoder import Decoder

import torch 


class EncoderDecoder(nn.Module):
    """
    Wrapper for a encoder and a decoder.
    Author : Thibault Groueix 01.11.2019
    """

    def __init__(self, opt):
        super(EncoderDecoder, self).__init__()

        self.encoder = Pct(opt)
  
        self.decoder = Decoder(opt)

        self.to(opt.device)


        # if not opt.SVR:
        #     self.apply(weights_init)  # initialization of the weights
        self.eval()

    def forward(self, points, train=True):
        features = self.encoder(points)
        final = self.decoder(features, points, train=train)
        return final

    def generate_mesh(self, x):
        return self.decoder.generate_mesh(self.encoder(x))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
