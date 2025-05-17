import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, SAGPooling
from torch_geometric.data import Batch
import config

class Net(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.gp_gnn = ConvGNN()
        self.sp_gnn = ConvGNN()
        self.lstm = LSTMNetwork(input_size = self.env.N_rels)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc = FC(d_out = self.env.out, device = self.device)
        self.mlp1 = FC(d_in = self.env.N_rels * self.env.N_rels * 2 * config.d['sys_args']['sql_batch_size'], d_out = self.env.N_rels * self.env.N_rels * 2)
        self.mlp2 = FC(d_in = self.env.N_rels * self.env.N_rels * 4, d_out = self.env.out)
        self.mlp3 = FC(d_in = self.env.N_rels * self.env.N_rels * 2, d_out = self.env.out, fc_nlayers = 1)
    def forward(self, input):
        # 节点Meta信息编码
        gp = [row[0] for row in input]
        sp = [row[1] for row in input]
        out1 = self.gp_gnn(gp)
        out2 = self.sp_gnn(sp)
        concat = torch.concat((out1, out2), dim = 1)
        meta_out = self.fc(concat)
        # 节点位置信息编码
        gb_vecs = torch.zeros((0, input[0][2].shape[0])).to(self.device)
        sp_vecs = torch.zeros((0, input[0][3].shape[0])).to(self.device)
        for row in input:
            gb_vecs = torch.vstack((gb_vecs, row[2]))
            sp_vecs = torch.vstack((sp_vecs, row[3]))
        out1 = self.mlp1(gb_vecs)
        concat = torch.cat((out1, sp_vecs), dim = 1)
        loc_out = self.mlp2(concat)
        out = self.mlp3(torch.cat((meta_out, loc_out), dim = 1))
        return out

    # def forward(self, input):
    #     gb_vecs = torch.zeros((0, input[0][0].shape[0])).to(self.device)
    #     sp_vecs = torch.zeros((0, input[0][1].shape[0])).to(self.device)
    #     for row in input:
    #         gb_vecs = torch.vstack((gb_vecs, row[0]))
    #         sp_vecs = torch.vstack((sp_vecs, row[1]))
    #     out1 = self.mlp1(gb_vecs)
    #     concat = torch.cat((out1, sp_vecs), dim = 1)
    #     out = self.mlp2(concat)
    #     return out
    
def FC(d_in = 128 * 2, d_out = 256, fc_nlayers = 4, drop = 0.2, device = 'cuda'):
    dims = torch.linspace(d_in, d_out, fc_nlayers+1, dtype=torch.long).to(device)
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
    def __init__(self, input_dim = config.d['net_args']['node_shape'], hidden_dim = 128, output_dim = 128, num_layers=4, init_ratio=0.8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self.fc = FC(d_in = hidden_dim, d_out = output_dim,device = self.device)
    
    # batch中可能有空图（如刚开始的GlobalPlan），需要单独处理。
    def forward(self, data_list):
        def add_dummy_node(data_list, num_features=config.d['net_args']['node_shape']):
            dummy_feature = torch.zeros(1, num_features).to(self.device)  # 虚拟节点特征
            for data in data_list:
                if data.num_nodes == 0:
                    data.x = dummy_feature  # 填充为 [1, num_features]
                    data.edge_index = torch.empty(2, 0, dtype=torch.long).to(self.device)  # 保持边为空
            return data_list
        data_list = add_dummy_node(data_list)
        n = len(data_list)
        index = [False if item is None else True for item in data_list]
        

        batch = Batch.from_data_list(data_list)

        
        x, edge_index , batch = batch.x, batch.edge_index, batch.batch
        
        
        for i in range(self.num_layers):
            # 图卷积层
            x = self.convs[i](x, edge_index).relu()
            
            # 动态池化层（更新节点和边索引）
            x, edge_index, _, batch, _, _ = self.pools[i](x, edge_index, batch = batch)
        
        # 全局平均池化得到图级向量
        x = self.global_pool(x, batch)
        out = torch.zeros(n, x.size(1), device=x.device)
        out[index] = x
    
        return self.fc(out)
    
import torch
import torch.nn as nn

class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size = 64, output_size = 128):
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
    
# Copyright 2018-2021 Tsinghua DBGroup
#
# Licensed under the Apache License, Version 2.0 (the "License"): you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
import torch
from torch.nn import init
import torch.nn as nn


class TreeLSTM(nn.Module):
    def __init__(self, num_units):
        super(TreeLSTM, self).__init__()
        self.num_units = num_units
        self.FC1 = nn.Linear(num_units, 5 * num_units)
        self.FC2 = nn.Linear(num_units, 5 * num_units)
        self.FC0 = nn.Linear(num_units, 5 * num_units)
        self.LNh = nn.LayerNorm(num_units,elementwise_affine = False)
        self.LNc = nn.LayerNorm(num_units,elementwise_affine = False)
    def forward(self, left_in, right_in,inputX):
        lstm_in = self.FC1(left_in[0])
        lstm_in += self.FC2(right_in[0])
        lstm_in += self.FC0(inputX)
        a, i, f1, f2, o = lstm_in.chunk(5, 1)
        c = (a.tanh() * i.sigmoid() + f1.sigmoid() * left_in[1] +
             f2.sigmoid() * right_in[1])
        h = o.sigmoid() * c.tanh()
        h = self.LNh(h)
        return h,c
class TreeRoot(nn.Module):
    def __init__(self,num_units):
        super(TreeRoot, self).__init__()
        self.num_units = num_units
        self.FC = nn.Linear(num_units, num_units)
        self.rootPool = 'meanPool'
        if self.rootPool == 'meanPool':
            self.sum_pooling = nn.AdaptiveAvgPool2d((1,num_units))
        else:
            self.sum_pooling = nn.AdaptiveMaxPool2d((1,num_units))
        
        # self.sum_pooling = nn.AdaptiveMaxPool2d((1,num_units))
        # self.max_pooling = nn.AdaptiveAvgPool2d((1,num_units))
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, tree_list):

        return self.relu(self.FC(self.sum_pooling(tree_list)).view(-1,self.num_units))

class SPINN(nn.Module):

    def __init__(self, n_classes, size, n_words, mask_size,device,max_column_in_table = 15):
        super(SPINN, self).__init__()
        self.size = size
        self.tree_lstm = TreeLSTM(size)
        self.tree_root = TreeRoot(size)
        self.FC = nn.Linear(size*2, size)
        self.table_embeddings = nn.Embedding(n_words, size)#2 * max_column_in_table * size)
        self.column_embeddings = nn.Embedding(n_words, (1+2 * max_column_in_table) * size)
        self.out = nn.Linear(size*2, size)
        self.out2 = nn.Linear(size, n_classes)
        self.outFc = nn.Linear(mask_size, size)
        self.rootPool = 'meanPool'
        if self.rootPool == 'meanPool':
            self.max_pooling = nn.AdaptiveAvgPool2d((1,size))
        else:
            self.max_pooling = nn.AdaptiveMaxPool2d((1,size))
        self.max_pooling = nn.AdaptiveMaxPool2d((1,size))
        self.relu = nn.ReLU()
        self.sigmoid = nn.ReLU()
        self.leafFC = nn.Linear(size, size)
        self.sigmoid = nn.Sigmoid()
        self.LN1 = nn.LayerNorm(size,)
        self.LN2 = nn.LayerNorm(size,)
        self.max_column_in_table = max_column_in_table
        self.leafLn = nn.LayerNorm(size,elementwise_affine = False)
        self.device = device
        self.sigmoid = nn.Sigmoid()

    def leaf(self, word_id, table_fea=None):
        # print('tlstm_wi',word_id)
        all_columns = table_fea.view(-1,self.max_column_in_table*2+1,1)*self.column_embeddings(word_id).reshape(-1,2 * self.max_column_in_table+1,self.size)
        all_columns = self.relu(self.leafFC(all_columns))
        table_emb = self.max_pooling(all_columns.view(-1,self.max_column_in_table*2+1,self.size)).view(-1,self.size)
        return self.leafLn(table_emb), torch.zeros(word_id.size()[0], self.size,device = self.device,dtype = torch.float32)
    def inputX(self,left_emb,right_emb):
        cat_emb = torch.cat([left_emb,right_emb],dim = 1)
        return self.relu(self.FC(cat_emb))
    def childrenNode(self, left_h, left_c, right_h, right_c,inputX):
        return self.tree_lstm((left_h, left_c), (right_h, right_c),inputX)
    def root(self,tree_list):
        return self.tree_root(tree_list).view(-1,self.size)
    def logits(self, encoding,join_matrix,prt=False):
        encoding = self.root(encoding.view(1,-1,self.size))
        # if prt:
        #     print(encoding)
        matrix = self.relu(self.outFc(join_matrix))
        # outencoding = torch.cat([encoding,encoding],dim = 1)
        outencoding = torch.cat([encoding,matrix],dim = 1)
        return self.out2(self.relu(self.out(outencoding)))

# 示例使用
if __name__ == "__main__":
    # 超参数
    # batch_size = 4
    # seq_len = 10    # 节点序列长度（定长）
    # input_size = 32  # 每个节点的编码维度
    # hidden_size = 64
    # output_size = 128
    
    # # 初始化模型
    # model = LSTMNetwork(input_size, hidden_size, output_size)
    
    # # 生成随机输入数据
    # x = torch.randn(seq_len, input_size)
    
    # # 前向传播
    # out = model(x)
    
    # print("输入形状:", x.shape)        # 输出: torch.Size([4, 10, 32])
    # print("输出形状:", out.shape)      # 输出: torch.Size([4, 128])
    # fc = FC(d_out = 20)
    x = torch.randn((5,3*128))
    fc = FC(d_out = 20)
    y = fc(x)
    print(y.shape)
    
