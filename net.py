import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, SAGPooling
from torch_geometric.data import Batch

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.gp_gnn = ConvGNN()
        self.sp_gnn = ConvGNN()
        self.fc = FC()
    
    def forward(self, input):
        # 分别为全局plan，当前plan，query
        gp, sp= input
        if gp is None:
            out1 = torch.zeros(self.gp_gnn.out_channels)
        else :
            out1 = self.gp_gnn(gp)
        out2 = self.sp_gnn(sp)
        concat = torch.concatenate((out1, out2.view(-1)))
        out = self.fc(concat)
        return out
    
def FC(d_in = 128 * 2, d_out = 256, fc_nlayers = 4, drop = 0.5):
    dims = torch.linspace(d_in, d_out, fc_nlayers+1, dtype=torch.long)
    layers = []
    for i in range(fc_nlayers-1):
        layers.extend([nn.Linear(int(dims[i]), int(dims[i+1])),
                       nn.Dropout(drop), nn.LayerNorm([int(dims[i+1])]), nn.ReLU()])
    layers.append(nn.Linear(int(dims[-2]), d_out))
    return nn.Sequential(*layers)
    
class ConvGNN(nn.Module):
    def __init__(self, in_channels = 45, hidden_channels = 128, out_channels = 128, num_layers=4):
        super(ConvGNN, self).__init__()
        self.in_chaanels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        # 创建交替的GCN卷积层和池化层
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            out_dim = hidden_channels
            self.convs.append(GCNConv(in_dim, out_dim))
            self.pools.append(SAGPooling(out_dim, ratio=0.5))
        
        # 最终全连接层
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, input):
        x = input.x
        edge_index = input.edge_index
        for i, (conv, pool) in enumerate(zip(self.convs, self.pools)):
            x = conv(x, edge_index)
            x = F.relu(x)
            x, edge_index ,_,_,_,_= pool(x, edge_index)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
