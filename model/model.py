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
    
    def setup(self, heart_name, data_path, batch_size, load_torso, graph_method):
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


class ST_GCNN_TorsoHeart(BaseModel):
    def __init__(self, 
                 num_filters,
                 len_seq,
                 latent_f_dim,
                 latent_s_dim):
        super().__init__()
        self.num_filters = num_filters
        self.len_seq = len_seq
        self.latent_f_dim = latent_f_dim
        self.latent_s_dim = latent_s_dim

        self.encoder = EncoderTorso(num_filters, len_seq, latent_f_dim)
        self.decoder = Decoder(num_filters, len_seq, latent_f_dim)

        self.trans = Spline(self.latent_f_dim, self.latent_f_dim, dim=3, kernel_size=3, norm=False, degree=2, root_weight=False, bias=False)

        self.H_inv = dict()
        self.P = dict()
    
    def setup(self, heart_name, data_path, batch_size, ecgi, graph_method):
        params = get_params(data_path, heart_name, batch_size, ecgi, graph_method)
        self.H_inv[heart_name] = params["H_inv"]
        self.P[heart_name] = params["P"]

        self.encoder.setup(heart_name, params)
        self.decoder.setup(heart_name, params)
    
    def inverse(self, z, heart_name):
        batch_size = z.shape[0]
        x = z.view(batch_size, -1, self.latent_f_dim, self.latent_s_dim)

        edge_index, edge_attr = self.H_inv[heart_name].edge_index, self.H_inv[heart_name].edge_attr
        num_heart, num_torso = self.P[heart_name].shape[0], self.P[heart_name].shape[1]
        
        x_bin = torch.zeros(batch_size, num_heart, self.latent_f_dim, self.latent_s_dim).to(device)
        x_bin = torch.cat((x_bin, x), 1)
        
        x_bin = x_bin.permute(3, 0, 1, 2).contiguous()
        x_bin = x_bin.view(-1, self.latent_f_dim)
        edge_index, edge_attr = expand(batch_size, num_heart + num_torso, self.latent_s_dim, edge_index, edge_attr)

        x_bin = self.trans(x_bin, edge_index, edge_attr)
        x_bin = x_bin.view(self.latent_s_dim, batch_size, -1, self.latent_f_dim)
        x_bin = x_bin.permute(1, 2, 3, 0).contiguous()
        
        x_bin = x_bin[:, 0:-num_torso, :, :]
        return x_bin
    
    def forward(self, y, heart_name):
        z = self.encoder(y, heart_name)
        z_h = self.inverse(z, heart_name)
        x = self.decoder(z_h, heart_name)
        return (x, None), None


class ST_GCNN_HeartOnly(BaseModel):
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
    
    def setup(self, heart_name, data_path, batch_size, load_torso, graph_method):
        params = get_params(data_path, heart_name, batch_size, load_torso, graph_method)
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
        # mu, logvar = self.latent_modeling(z, heart_name)
        # z = self.reparameterize(mu, logvar)
        x = self.decoder(z, heart_name)
        # return (x, None), (mu, logvar)
        return (x, None), (None, None)


class MetaDynamics(BaseModel):
    def __init__(self,
                 num_channel,
                 latent_dim,
                 obs_dim,
                 init_dim,
                 rnn_type,
                 trans_model,
                 trans_args):
        super().__init__()
        self.nf = num_channel
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.init_dim = init_dim
        self.rnn_type = rnn_type
        self.trans_model = trans_model
        self.trans_args = trans_args

        # Domain model
        self.domain = DomainEncoder(obs_dim, num_channel, latent_dim)
        self.mu_c = nn.Linear(latent_dim, latent_dim)
        self.var_c = nn.Linear(latent_dim, latent_dim)

        # initialization
        self.initial = InitialEncoder(init_dim, num_channel, latent_dim)
        self.mu_0 = nn.Linear(latent_dim, latent_dim)
        self.var_0 = nn.Linear(latent_dim, latent_dim)

        # time modeling
        if trans_model == 'recurrent':
            self.propagation = Transition_Recurrent(**trans_args)
        elif trans_model == 'ODE':
            self.propagation = Transition_ODE(**trans_args)

        # decoder
        self.decoder = SpatialDecoder(num_channel, latent_dim)

    def setup(self, heart_name, data_path, batch_size, ecgi, graph_method):
        params = get_params(data_path, heart_name, batch_size, ecgi, graph_method)        
        self.domain.setup(heart_name, params)
        self.initial.setup(heart_name, params)
        self.decoder.setup(heart_name, params)
    
    def latent_domain(self, D, K, heart_name):
        N, _, V, T = D.shape
        D_z_c = []
        for i in range(K):
            Di = D[:, i, :, :].view(N, V, T)
            z_c_i = self.domain(Di, heart_name)
            D_z_c.append(z_c_i)

        z_c = sum(D_z_c) / len(D_z_c)
        mu_c = self.mu_c(z_c)
        logvar_c = self.var_c(z_c)
        mu_c = torch.clamp(mu_c, min=-100, max=85)
        logvar_c = torch.clamp(logvar_c, min=-100, max=85)
        z = self.reparameterization(mu_c, logvar_c)

        return z, mu_c, logvar_c
    
    def latent_initial(self, x, heart_name):
        x = x[:, :, :self.init_dim]
        z_0 = self.initial(x, heart_name)
        mu_0 = self.mu_0(z_0)
        logvar_0 = self.var_0(z_0)
        mu_0 = torch.clamp(mu_0, min=-100, max=85)
        logvar_0 = torch.clamp(logvar_0, min=-100, max=85)
        z = self.reparameterization(mu_0, logvar_0)
        
        return z, mu_0, logvar_0
    
    def time_modeling(self, T, z_0, z_c):
        N, V, C = z_0.shape

        if self.trans_model in ['recurrent',]:
            z_prev = z_0
            z = []
            for i in range(1, T):
                z_t = self.propagation(z_prev, z_c)
                z_prev = z_t
                z_t = z_t.view(1, N, V, C)
                z.append(z_t)
            z = torch.cat(z, dim=0)
            z_0 = z_0.view(1, N, V, C)
            z = torch.cat([z_0, z], dim=0)
        elif self.trans_model in ['ODE',]:
            z = self.propagation(T, z_0, z_c)
        z = z.permute(1, 2, 3, 0).contiguous()

        return z
    
    def reparameterization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y, D_x, D_y, heart_name):
        N, V, T = x.shape

        # q(c | D)
        K = D_x.shape[1]
        z_c, mu_c, logvar_c = self.latent_domain(D_x, K, heart_name)

        # q(z)
        z_0, mu_0, logvar_0 = self.latent_initial(x, heart_name)

        # p(x | z, c)
        z = self.time_modeling(T, z_0, z_c)
        x_ = self.decoder(z, heart_name)

        # KL on all data
        x = x.view(N, 1, -1, T)
        D_x_cat = torch.cat([D_x, x], dim=1)
        _, mu_t, logvar_t = self.latent_domain(D_x_cat, K, heart_name)

        return (x_, ), (mu_c, logvar_c, mu_t, logvar_t, mu_0, logvar_0)

    def prediction(self, qry_x, spt_x, D_x, D_y, heart_name):
        N, V, T = qry_x.shape

        # q(c | D)
        K = D_x.shape[1]
        z_c, mu_c, logvar_c = self.latent_domain(D_x, K, heart_name)

        # q(z)
        z_0, mu_0, logvar_0 = self.latent_initial(qry_x, heart_name)

        # p(x | z, c)
        z = self.time_modeling(T, z_0, z_c)
        x_ = self.decoder(z, heart_name)

        return (x_, ), (None, None, None, None, None, None)
