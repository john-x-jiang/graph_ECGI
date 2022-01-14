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


class BayesianFilter(BaseModel):
    def __init__(self,
                 num_channel,
                 latent_dim,
                 obs_dim,
                 ode_func_type,
                 ode_num_layers,
                 ode_method,
                 rnn_type):
        super().__init__()
        self.nf = num_channel
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.ode_func_type = ode_func_type
        self.ode_num_layers = ode_num_layers
        self.ode_method = ode_method
        self.rnn_type = rnn_type

        # encoder
        self.encoder = Encoder(num_channel, latent_dim)

        # Domain model
        self.domain_embedding = RnnEncoder(latent_dim, latent_dim,
                                           dim=3,
                                           kernel_size=3,
                                           norm=False,
                                           n_layer=1,
                                           rnn_type=rnn_type,
                                           bd=False,
                                           reverse_input=False)
        self.domain = DomainEncoder(latent_dim, latent_dim, obs_dim, stochastic=False)
        
        self.propagation = Propagation(latent_dim, fxn_type=ode_func_type, num_layers=ode_num_layers, method=ode_method, rtol=1e-5, atol=1e-7)
        self.correction = Correction(latent_dim, rnn_type=rnn_type, dim=3, kernel_size=3, norm=False)

        # decoder
        self.decoder = Decoder(num_channel, latent_dim)

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
    
    def time_modeling(self, x, heart_name):
        N, V, C, T = x.shape
        edge_index, edge_attr = self.bg4[heart_name].edge_index, self.bg4[heart_name].edge_attr

        x = x.permute(3, 0, 1, 2).contiguous()
        last_h = x[0]

        # Domain
        _x = self.domain_embedding(x, edge_index, edge_attr)
        z_D = self.domain(_x)

        z = []
        z.append(last_h.view(1, N, V, C))

        x = x.view(T, N * V, C)
        for t in range(1, T):
            last_h = last_h.view(N, V, -1)

            # Propagation
            last_h = self.propagation(last_h, z_D, 1, steps=1)
            # Corrrection
            last_h = last_h.view(N * V, -1)
            h = self.correction(x[t], last_h, edge_index, edge_attr)

            last_h = h
            z.append(h.view(1, N, V, C))
        
        z = torch.cat(z, dim=0)
        z = z.permute(1, 2, 3, 0).contiguous()
        return z
    
    def forward(self, x, heart_name):
        embed = self.encoder(x, heart_name)
        z = self.time_modeling(embed, heart_name)
        x = self.decoder(z, heart_name)
        return (x, None), (None, None, None, None)
