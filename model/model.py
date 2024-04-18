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
    
    def setup(self, heart_name, data_path, batch_size, load_torso, load_physics, graph_method):
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
    
    def setup(self, heart_name, data_path, batch_size, load_torso, load_physics, graph_method):
        params = get_params(data_path, heart_name, batch_size, load_torso, load_physics, graph_method)
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
    
    def setup(self, heart_name, data_path, batch_size, load_torso, load_physics, graph_method):
        params = get_params(data_path, heart_name, batch_size, load_torso, load_physics, graph_method)
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
                 trans_model,
                 trans_args):
        super().__init__()
        self.nf = num_channel
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.init_dim = init_dim
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

    def setup(self, heart_name, data_path, batch_size, load_torso, load_physics, graph_method):
        params = get_params(data_path, heart_name, batch_size, load_torso, load_physics, graph_method)        
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

    def prediction(self, qry_x, qry_y, D_x, D_y, heart_name):
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


class MetaDynamics_MaskIn(BaseModel):
    def __init__(self,
                 num_channel,
                 latent_dim,
                 obs_dim,
                 trans_model,
                 trans_args):
        super().__init__()
        self.nf = num_channel
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.trans_model = trans_model
        self.trans_args = trans_args

        # encoder
        self.signal_encoder = SpatialEncoder(num_channel, latent_dim, cond=True)

        # Domain model
        self.domain_seq = RnnEncoder(latent_dim, latent_dim,
                                     dim=3,
                                     kernel_size=3,
                                     norm=False,
                                     n_layer=1,
                                     bd=False)
        self.domain = Aggregator(latent_dim, latent_dim, obs_dim, stochastic=False)
        self.mu_c = nn.Linear(latent_dim, latent_dim)
        self.var_c = nn.Linear(latent_dim, latent_dim)

        # initialization
        self.condition_encoder = SpatialEncoder(num_channel, latent_dim)
        self.initial = nn.Linear(latent_dim, latent_dim)

        # time modeling
        if trans_model == 'recurrent':
            self.propagation = Transition_Recurrent(**trans_args)
        elif trans_model == 'ODE':
            self.propagation = Transition_ODE(**trans_args)
        
        # decoder
        self.decoder = SpatialDecoder(num_channel, latent_dim)

        self.bg = dict()
        self.bg1 = dict()
        self.bg2 = dict()
        self.bg3 = dict()
        self.bg4 = dict()

    def setup(self, heart_name, data_path, batch_size, load_torso, load_physics, graph_method):
        params = get_params(data_path, heart_name, batch_size, load_torso, load_physics, graph_method)
        self.bg[heart_name] = params["bg"]
        self.bg1[heart_name] = params["bg1"]
        self.bg2[heart_name] = params["bg2"]
        self.bg3[heart_name] = params["bg3"]
        self.bg4[heart_name] = params["bg4"]
        
        self.signal_encoder.setup(heart_name, params)
        self.condition_encoder.setup(heart_name, params)
        self.decoder.setup(heart_name, params)
    
    def latent_domain(self, D_x, D_y, K, heart_name):
        edge_index, edge_attr = self.bg4[heart_name].edge_index, self.bg4[heart_name].edge_attr

        N, _, V, T = D_x.shape
        D_z_c = []
        for i in range(K):
            D_xi = D_x[:, i, :, :].view(N, V, T)
            D_yi = D_y[:, i, :]
            D_yi = one_hot_label(D_yi[:, 2] - 1, N, V, T)
            z_i = self.signal_encoder(D_xi, heart_name, D_yi)
            z_c_i = self.domain_seq(z_i, edge_index, edge_attr)
            D_z_c.append(self.domain(z_c_i))

        z_c = sum(D_z_c) / len(D_z_c)
        mu_c = self.mu_c(z_c)
        logvar_c = self.var_c(z_c)
        mu_c = torch.clamp(mu_c, min=-100, max=85)
        logvar_c = torch.clamp(logvar_c, min=-100, max=85)
        z = self.reparameterization(mu_c, logvar_c)

        return z, mu_c, logvar_c
    
    def latent_initial(self, y, N, V, heart_name):
        y = one_hot_label(y[:, 2] - 1, N, V, 1)
        y = y[:, :].view(N, V, 1)
        z_0 = self.condition_encoder(y, heart_name)
        z_0 = torch.squeeze(z_0)
        z_0 = self.initial(z_0)
        
        return z_0
    
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
        z_c, mu_c, logvar_c = self.latent_domain(D_x, D_y, K, heart_name)

        # q(z)
        z_0 = self.latent_initial(y, N, V, heart_name)

        # p(x | z, c)
        z = self.time_modeling(T, z_0, z_c)
        x_ = self.decoder(z, heart_name)

        # KL on all data
        x = x.view(N, 1, -1, T)
        y = y.view(N, 1, -1)
        D_x_cat = torch.cat([D_x, x], dim=1)
        D_y_cat = torch.cat([D_y, y], dim=1)
        _, mu_t, logvar_t = self.latent_domain(D_x_cat, D_y_cat, K, heart_name)

        return (x_, ), (mu_c, logvar_c, mu_t, logvar_t)

    def prediction(self, x, y, D_x, D_y, heart_name):
        N, V, T = x.shape

        # q(c | D)
        K = D_x.shape[1]
        z_c, mu_c, logvar_c = self.latent_domain(D_x, D_y, K, heart_name)

        # q(z)
        z_0 = self.latent_initial(y, N, V, heart_name)

        # p(x | z, c)
        z = self.time_modeling(T, z_0, z_c)
        x_ = self.decoder(z, heart_name)

        return (x_, ), (None, None, None, None)


class HybridSSM(BaseModel):
    def __init__(self, 
                 num_filters,
                 latent_dim,
                 ode_args=None):
        super().__init__()
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.ode_args = ode_args

        self.encoder = SpatialEncoderTorso(num_filters, latent_dim)
        self.decoder = SpatialDecoder(num_filters, latent_dim)

        self.transition = Propagation(**ode_args)
        self.correction = Correction(latent_dim, dim=3, kernel_size=3, norm=False)

        self.trans = Spline(self.latent_dim, self.latent_dim, dim=3, kernel_size=3, norm=False, degree=2, root_weight=False, bias=False)

        self.H_inv = dict()
        self.P = dict()
        self.H = dict()
        self.L = dict()
    
    def setup(self, heart_name, data_path, batch_size, load_torso, load_physics, graph_method):
        params = get_params(data_path, heart_name, batch_size, load_torso, load_physics, graph_method)
        self.H_inv[heart_name] = params["H_inv"]
        self.P[heart_name] = params["P"]
        self.H[heart_name] = params["H"]
        self.L[heart_name] = params["L"]

        self.encoder.setup(heart_name, params)
        self.decoder.setup(heart_name, params)
    
    def inverse(self, z, heart_name):
        batch_size, seq_len = z.shape[0], z.shape[-1]
        x = z.view(batch_size, -1, self.latent_dim, seq_len)

        edge_index, edge_attr = self.H_inv[heart_name].edge_index, self.H_inv[heart_name].edge_attr
        num_heart, num_torso = self.P[heart_name].shape[0], self.P[heart_name].shape[1]
        
        x_bin = torch.zeros(batch_size, num_heart, self.latent_dim, seq_len).to(device)
        x_bin = torch.cat((x_bin, x), 1)
        
        x_bin = x_bin.permute(3, 0, 1, 2).contiguous()
        x_bin = x_bin.view(-1, self.latent_dim)
        edge_index, edge_attr = expand(batch_size, num_heart + num_torso, seq_len, edge_index, edge_attr)

        x_bin = self.trans(x_bin, edge_index, edge_attr)
        x_bin = x_bin.view(seq_len, batch_size, -1, self.latent_dim)
        x_bin = x_bin.permute(1, 2, 3, 0).contiguous()
        
        x_bin = x_bin[:, 0:-num_torso, :, :]
        return x_bin
    
    def time_modeling(self, x, heart_name):
        N, V, C, T = x.shape
        edge_index, edge_attr = self.decoder.bg4[heart_name].edge_index, self.decoder.bg4[heart_name].edge_attr

        x = x.permute(3, 0, 1, 2).contiguous()
        last_h = x[0]

        outputs = []
        outputs.append(last_h.view(1, N, V, C))

        outputs_ode = []
        outputs_ode.append(last_h.view(1, N, V, C))

        x = x.view(T, N * V, C)
        for t in range(1, T):
            last_h = last_h.view(N, V, -1)

            # Propagation
            # TODO: change dt and steps by temporal resolution
            dt = 1
            steps = 1
            last_h = self.transition(last_h, dt, steps=steps)

            last_h = last_h[-1, :, :, :]
            outputs_ode.append(last_h.view(1, N, V, C))
            # Corrrection
            last_h = last_h.view(N * V, -1)
            h = self.correction(x[t], last_h, edge_index, edge_attr)

            last_h = h
            outputs.append(h.view(1, N, V, C))
        
        outputs = torch.cat(outputs, dim=0)
        outputs = outputs.permute(1, 2, 3, 0).contiguous()
        return outputs

    def regularization(self, x_, heart_name):
        LX = torch.matmul(self.L[heart_name], x_)
        y_ = torch.matmul(self.H[heart_name], x_)
        return LX, y_

    def forward(self, y, heart_name):
        z_torso = self.encoder(y, heart_name)
        z_heart = self.inverse(z_torso, heart_name)
        z = self.time_modeling(z_heart, heart_name)
        x = self.decoder(z, heart_name)
        LX, y = self.regularization(x, heart_name)
        return (x, LX, y), None
