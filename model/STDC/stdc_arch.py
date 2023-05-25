import torch
import torch.nn as nn
import torch.nn.functional as F

class STD(nn.Module):

    def __init__(self,num_nodes,input_len,emb_dim) -> None:
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, emb_dim)
        self.LFC = nn.Conv2d(in_channels=input_len, out_channels=emb_dim, kernel_size=(1, 1))
        self._init_weight_()

    def _init_weight_(self):
        nn.init.xavier_uniform_(self.node_emb.weight)

    def forward(self,x):
        batch_size = x.shape[0]
        SE = self.node_emb.weight.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1)
        L = self.LFC(x)
        return SE,L


class STDELayer(nn.Module):
    def __init__(self,num_nodes,emb_dim,input_len,steps_per_day) -> None:
        super().__init__()
        self.STD = STD(num_nodes,input_len,emb_dim)
        self.TFC = nn.Conv2d(in_channels=steps_per_day + 7, out_channels=emb_dim, kernel_size=(1, 1))
        self.steps_per_day = steps_per_day

    def forward(self, input):

        x = input[..., 0].unsqueeze(-1)
        SE,L = self.STD(x)
        x_tod = input[..., 1]
        t_tod = F.one_hot((x_tod[:, -1, :] * 288).type(torch.LongTensor),num_classes = self.steps_per_day).transpose(1,2).unsqueeze(-1)
        x_dow = input[..., 2]
        t_dow = F.one_hot((x_dow[:, -1, :] ).type(torch.LongTensor),num_classes = 7).transpose(1,2).unsqueeze(-1)
        TE = self.TFC(torch.concat([t_tod,t_dow], dim=1).type(torch.FloatTensor).cuda())
        H = torch.concat([SE,TE,L],dim=1)
        return H


class SNorm(nn.Module):
    def __init__(self,  channels):
        super(SNorm, self).__init__()
        self.beta = nn.Parameter(torch.zeros(channels))
        self.gamma = nn.Parameter(torch.ones(channels))

    def forward(self, x):
        x_norm = (x - x.mean(2, keepdims=True))  / (x.var(2, keepdims=True, unbiased=True) + 0.00001) ** 0.5
        out = x_norm * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)
        return out

class TNorm(nn.Module):
    def __init__(self, num_nodes, channels, track_running_stats=True, momentum=0.1):
        super(TNorm, self).__init__()
        self.track_running_stats = track_running_stats
        self.beta = nn.Parameter(torch.zeros(1, channels, num_nodes,1))
        self.gamma = nn.Parameter(torch.ones(1, channels, num_nodes,1))
        self.register_buffer('running_mean', torch.zeros(1, channels, num_nodes,1))
        self.register_buffer('running_var', torch.ones(1, channels, num_nodes,1))
        self.momentum = momentum

    def forward(self, x):
        if self.track_running_stats:
            mean = x.mean((0, 3), keepdims=True)
            var = x.var((0, 3), keepdims=True, unbiased=False)
            if self.training:
                n = x.shape[3] * x.shape[0]
                with torch.no_grad():
                    self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
                    self.running_var = self.momentum * var * n / (n - 1) + (1 - self.momentum) * self.running_var
            else:
                mean = self.running_mean
                var = self.running_var
        else:
            mean = x.mean((3), keepdims=True)
            var = x.var((3), keepdims=True, unbiased=True)
        x_norm = (x - mean) / (var + 0.00001) ** 0.5
        out = x_norm * self.gamma + self.beta
        return out


class STCBlcok(nn.Module):

    def __init__(self,hidden_units,num_nodes,dropout) -> None:
        super().__init__()
        self.spatial_norm = SNorm(channels=hidden_units)
        self.temporal_norm = TNorm(num_nodes=num_nodes, channels=hidden_units)
        self.FC1 = nn.Conv2d(in_channels=hidden_units * 3, out_channels=hidden_units, kernel_size=(1, 1))
        self.FC2 = nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=(1, 1))
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, h):
        hs = self.spatial_norm(h)
        ht = self.temporal_norm(h)
        z = self.drop(torch.concat([ht,hs,h],dim=1))
        out = self.FC2(self.drop(self.relu(self.FC1(z))))
        return out + h

class STDC(nn.Module):
    def __init__(self, num_nodes, emb_dim, input_dim, input_len, num_layer, output_len,droupt, steps_per_day,**kwargs) -> None:
        super().__init__()
        stacking_dim = emb_dim * input_dim
        self.STDELayer = STDELayer(num_nodes=num_nodes, emb_dim=emb_dim,input_len=input_len,steps_per_day=steps_per_day)
        self.STCLayer = nn.Sequential(
            *[STCBlcok(hidden_units=stacking_dim, num_nodes=num_nodes, dropout=droupt)
            for _ in range(num_layer)])
        self.outputLayer = nn.Conv2d(in_channels=stacking_dim, out_channels=output_len, kernel_size=(1, 1))

    def forward(self, history_data, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs):
        input = history_data
        h = self.STDELayer(input)
        h = self.STCLayer(h)
        out = self.outputLayer(h)
        return out


