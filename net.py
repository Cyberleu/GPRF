import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, SAGPooling
from torch_geometric.data import Batch

class Net(nn.Module):
    def __init__(self, d_out):
        super().__init__()
        self.gp_gnn = ConvGNN()
        self.sp_gnn = ConvGNN()
        self.lstm = LSTMNetwork(input_size = 21)
        self.fc = FC(d_out = d_out)
    
    def forward(self, input):
        # 分别为全局plan，当前plan，batch中所有查询的table序列
        gp, sp, bt= input
        if gp is None:
            out1 = torch.zeros(self.gp_gnn.output_dim)
        else :
            out1 = self.gp_gnn(gp)
        out2 = self.sp_gnn(sp)
        out3 = self.lstm(bt)
        concat = torch.concatenate((out1.view(-1), out2.view(-1),out3.view(-1)))
        out = self.fc(concat)
        return out
    
def FC(d_in = 128 * 3, d_out = 256, fc_nlayers = 4, drop = 0.5):
    dims = torch.linspace(d_in, d_out, fc_nlayers+1, dtype=torch.long)
    layers = []
    for i in range(fc_nlayers-1):
        layers.extend([nn.Linear(int(dims[i]), int(dims[i+1])),
                       nn.Dropout(drop), nn.LayerNorm([int(dims[i+1])]), nn.ReLU()])
    layers.append(nn.Linear(int(dims[-2]), d_out))
    return nn.Sequential(*layers)
    
# class ConvGNN(nn.Module):
#     def __init__(self, in_channels = 45, hidden_channels = 128, out_channels = 128, num_layers=4):
#         super(ConvGNN, self).__init__()
#         self.in_chaanels = in_channels
#         self.hidden_channels = hidden_channels
#         self.out_channels = out_channels
#         self.num_layers = num_layers
#         self.convs = nn.ModuleList()
#         self.pools = nn.ModuleList()
        
#         # 创建交替的GCN卷积层和池化层
#         for i in range(num_layers):
#             in_dim = in_channels if i == 0 else hidden_channels
#             out_dim = hidden_channels
#             self.convs.append(GCNConv(in_dim, out_dim))
#             self.pools.append(SAGPooling(out_dim, ratio=0.5))
        
#         # 最终全连接层
#         self.lin = nn.Linear(hidden_channels, out_channels)

#     def forward(self, input):
#         x = input.x
#         edge_index = input.edge_index
#         for i, (conv, pool) in enumerate(zip(self.convs, self.pools)):
#             x = conv(x, edge_index)
#             x = F.relu(x)
#             x, edge_index ,_,_,_,_= pool(x, edge_index)
#         x = self.lin(x)
#         return F.log_softmax(x, dim=1)

import torch
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, SAGPooling, global_mean_pool
from torch_geometric.utils import add_self_loops

class ConvGNN(nn.Module):
    def __init__(self, input_dim = 45, hidden_dim = 128, output_dim = 128, num_layers=4, init_ratio=0.8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        # 动态调整池化比例：每层减少 20% 的节点
        self.ratios = [init_ratio * (0.8 ** i) for i in range(num_layers)]
        
        # 初始化交替的卷积和池化层
        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden_dim
            self.convs.append(GCNConv(in_channels, hidden_dim))
            self.pools.append(SAGPooling(hidden_dim, ratio=self.ratios[i]))
        
        # 全局池化和全连接层
        self.global_pool = global_mean_pool
        self.fc = FC(d_in = hidden_dim, d_out = output_dim)
    
    def forward(self, data_list):
        data = Batch.from_data_list(data_list)
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for i in range(self.num_layers):
            # 图卷积层
            x = self.convs[i](x, edge_index).relu()
            
            # 动态池化层（更新节点和边索引）
            x, edge_index, _, batch, _, _ = self.pools[i](x, edge_index, batch = batch)
        
        # 全局平均池化得到图级向量
        x = self.global_pool(x, batch)
        return self.fc(x)
    
import torch
import torch.nn as nn

class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMNetwork, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x 形状: (batch_size, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 取最后一个时间步的隐藏状态 [更常用]
        # h_n 形状: (num_layers, batch_size, hidden_size)
        last_hidden = h_n[-1]  # 取最后一层的隐藏状态
        
        # 或者也可以用最后一个时间步的输出：
        # last_output = lstm_out[:, -1, :]
        
        output = self.fc(last_hidden)
        return output

# 示例使用
if __name__ == "__main__":
    # 超参数
    batch_size = 4
    seq_len = 10    # 节点序列长度（定长）
    input_size = 32  # 每个节点的编码维度
    hidden_size = 64
    output_size = 128
    
    # 初始化模型
    model = LSTMNetwork(input_size, hidden_size, output_size)
    
    # 生成随机输入数据
    x = torch.randn(seq_len, input_size)
    
    # 前向传播
    out = model(x)
    
    print("输入形状:", x.shape)        # 输出: torch.Size([4, 10, 32])
    print("输出形状:", out.shape)      # 输出: torch.Size([4, 128])
    
