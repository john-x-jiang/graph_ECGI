import torch
import torch.nn as nn
import numpy as np
from model.modules import *
from abc import abstractmethod

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class EuclideanModel(BaseModel):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.LSTM(input_dim, 800)
        self.fc21 = nn.LSTM(800, 50)
        self.fc22 = nn.LSTM(800, 50)
        self.fc3 = nn.LSTM(50, 800)
        self.fc41 = nn.LSTM(800, input_dim)
        self.fc42 = nn.LSTM(800, input_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def setup(self, heart_name, data_path, batch_size, ecgi, graph_method):
        pass
    
    def encode(self, x):
        out, hidden = self.fc1(x)
        h1 = self.relu(out)
        out21,hidden21 = self.fc21(h1)
        out22, hidden22 = self.fc22(h1)
        return out21, out22

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        out3, hidden3 = self.fc3(z)
        h3 = self.relu(out3)
        out1,hidden1 = self.fc41(h3)
        out2, hidden2 = self.fc42(h3)
        return (out1), (out2)

    def forward(self, x, heart_name):
        x = x.permute(2, 0, 1).contiguous()
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        muTheta, logvarTheta = self.decode(z)
        
        muTheta = muTheta.permute(1, 2, 0).contiguous()
        logvarTheta = logvarTheta.permute(1, 2, 0).contiguous()
        mu = mu.permute(1, 2, 0).contiguous()
        logvar = logvar.permute(1, 2, 0).contiguous()
        return (muTheta, logvarTheta), (mu, logvar)


class ST_GCNN(BaseModel):
    def __init__(self,
                 num_filters,
                 len_seq,
                 latent_f_dim,
                 latent_s_dim,):
        super().__init__()
        self.num_filters = num_filters
        self.len_seq = len_seq
        self.latent_f_dim = latent_f_dim
        self.latent_s_dim = latent_s_dim

        self.encoder = Encoder(num_filters, len_seq, latent_f_dim)
        self.decoder = Decoder(num_filters, len_seq, latent_f_dim)

        # TODO: add necessary nn modules for latent modeling below
    
        self.bg = dict()
        self.bg1 = dict()
        self.bg2 = dict()
        self.bg3 = dict()
        self.bg4 = dict()

    def setup(self, heart_name, data_path, batch_size, ecgi, graph_method):
        params = get_params(data_path, heart_name, batch_size, ecgi, graph_method)
        self.bg[heart_name] = params["bg"]
        self.bg1[heart_name] = params["bg1"]
        self.bg2[heart_name] = params["bg2"]
        self.bg3[heart_name] = params["bg3"]
        self.bg4[heart_name] = params["bg4"]
        
        self.encoder.setup(heart_name, params)
        self.decoder.setup(heart_name, params)
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def latent_modeling(self, z, heart_name):
        # TODO: need to design a graph-specific probabilistic model in latent space
        raise NotImplemented
    
    def forward(self, x, heart_name):
        z = self.encoder(x, heart_name)
        mu, logvar = self.latent_modeling(z, heart_name)
        x = self.decoder(z, heart_name)
        return (x, None), (mu, logvar)
