import torch.nn as nn
import importlib
from model.PUGAN_encoder import Pct
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

    def forward(self, point_list, train=True):
        if train== True:
            representation_list, Loss_feat_list = self.encoder(point_list, train=train)
            final_list = self.decoder(representation_list, train=train)
            return final_list, Loss_feat_list
        else:
            features = self.encoder(point_list, train=train)
            final = self.decoder(features, train=train)
            return final

    def generate_mesh(self, x):
        return self.decoder.generate_mesh(self.encoder(x))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
