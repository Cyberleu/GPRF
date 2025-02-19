import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, TopKPooling
from torch_geometric.data import Batch

class Net():
    def __init__(self, config):
        self.gp_gnn = GCNConv()
        self.sp_gnn = GCNConv()
    
    def forward(self, input):
        # 分别为全局plan，当前plan，query
        gp, sp= input
        out1 = self.gp_gnn(gp)
        out2 = self.sp_gnn(sp)
        concat = torch.concatenate((out1, out2))
        out = FC(concat)
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
        
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        # 创建交替的GCN卷积层和池化层
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            out_dim = hidden_channels
            
            # 添加图卷积层
            self.convs.append(GCNConv(in_dim, out_dim))
            
            # 每隔一层添加池化层
            if i % 2 == 1:
                self.pools.append(TopKPooling(out_dim, ratio=0.5))
        
        # 最终全连接层
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, input, batch = torch.cat([torch.zeros(100)]).long()):
        # 保存中间结果用于跳跃连接
        x = input.x
        edge_index = input.edge_index
        x_all = []
        edge_index_all = []
        batch_all = []
        
        for i, (conv, pool) in enumerate(zip(self.convs, self.pools)):
            # 图卷积操作
            x = conv(x, edge_index)
            x = F.relu(x)
            
            # 每隔一层进行池化
            if i % 2 == 1:
                x, edge_index, _, batch, _, _ = pool(x, edge_index, None, batch)
                
                # 保存中间结果
                x_all.append(x)
                edge_index_all.append(edge_index)
                batch_all.append(batch)
        
        # 使用所有中间层的特征进行平均
        x = torch.cat([global_mean_pool(x, batch) for x, batch in zip(x_all, batch_all)], dim=1)
        
        # 全连接层
        x = self.lin(x)
        return F.log_softmax(x, dim=1)

# 使用示例
model = ConvGNN(in_channels=64,    # 输入特征维度
               hidden_channels=128, # 隐藏层维度
               out_channels=10,     # 输出类别数
               num_layers=4)        # 总层数

# 假设输入数据
x = torch.randn(100, 64)          # 100个节点，每个节点64维特征
edge_index = torch.randint(0, 100, (2, 200))  # 随机边
batch = torch.cat([torch.zeros(100)]).long()  # 两个图的batch

output = model(x, edge_index, batch)
print(output)  # torch.Size([2, 10])

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
