from plan import *
from db_utils import *
from tree import *
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import time
from torch.utils.tensorboard import SummaryWriter
import re
from env import Env
import config
import random
from net import Net

import logging
LOG = logging.getLogger(__name__)

INF = 1e9
TARGET_REPLACE_ITER = 5                       # 目标网络更新频率(固定不懂的Q网络)
MEMORY_CAPACITY = 500                          # 记忆库容量
BATCH_SIZE = 60   
GET_SAMPLE = 0  # 0为随机获取sample， 1为获取最新n个sample
REWARD_UPDATE_METHOD = 0 # 0为更新成相同的reward 1为按比例减小reward
REWARD_UPDATE_COEF = 0.9
GAMMA = 0.9
LR = 0.01

class Agent(nn.Module):
    def __init__(self, eval_net, target_net, eps=0.5, device='cpu'):
        super().__init__()
        self.eval_net = eval_net
        self.target_net = target_net
        self.learn_step_counter = 0
        self.device = device
        self.eps = eps
        self.loss_func = nn.MSELoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(),lr=LR)
        self.ts = TrajectoryStorage()
        # if self.net.pretrained_path:
        #     self.net.load_state_dict(torch.load(self.net.pretrained_path))
        #     if len(self.net.fit_pretrained_layers) > 0:
        #         self.net.requires_grad_(False)
        #         unfreezing_p = []
        #         for n, m in self.net.named_parameters():
        #             for l in self.net.fit_pretrained_layers:
        #                 pattern = re.compile(f"{l}\.|{l}$")
        #                 if re.match(pattern, n):
        #                     unfreezing_p.append(n)
        #                     m.requires_grad_(True)
        #         LOG.debug(f"Training parameters: {unfreezing_p}")

    def predict(self, inputs, mask):
        mask_dim = mask.shape
        logit = self.eval_net([inputs])
        logit = logit.view(-1)
        dims = logit.shape
        masked_logit = torch.where(mask.view(dims).to(logit.device),
                                   logit, torch.tensor(float("-inf")).to(logit.device))
        probs = masked_logit.softmax(-1).detach().numpy()
        if np.random.uniform() < self.eps:
            action_idx = np.random.choice(range(0,len(probs)), 1, p=probs)[0]
        else :
            action_idx = np.argmax(probs)
        action = np.unravel_index(action_idx, mask_dim)
        assert mask[action[0]][action[1]] == True
        return action

    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())   # 将评价网络的权重参数赋给目标网络
        self.learn_step_counter +=1                 # 目标函数的学习次数+1
        
        # 抽buffer中的数据学习
        if GET_SAMPLE == 0:
            data = self.ts.get_dataset_random(BATCH_SIZE)
        elif GET_SAMPLE == 1:
            data = self.ts.get_dataset_lastn(BATCH_SIZE)
        length = len(data)
        mask_length = data[0][5].shape[0]
        b_s = [row[0] for row in data]
        b_a = [[row[1][0] *  mask_length + row[1][1]] for row in data]
        b_r  = [row[2] for row in data]
        b_r = torch.tensor(b_r, dtype=torch.float32)
        b_ns = [row[3] for row in data]
        b_d = [row[4] for row in data]
        b_nm = torch.empty((0,mask_length, mask_length), dtype=bool)
        for row in data:
            b_nm = torch.concat((b_nm, row[5].unsqueeze(0)))
        
        
        q_eval = self.eval_net(b_s).gather(1, torch.tensor(b_a)).view(-1)
        
        logit = self.target_net(b_ns)
        q_next = torch.where(b_nm.view(length,-1).to(logit.device),
                                   logit, torch.tensor(float("-inf")).to(logit.device)).detach()
        
        q_target = b_r + GAMMA * q_next.max(1)[0]
        # q_next.max(1)[0]表示只返回每一行的最大值，不返回索引(长度为32的一维张量)；.view()表示把前面所得到的一维张量变成(BATCH_SIZE, 1)的形状；最终通过公式得到目标值
        loss = self.loss_func(q_eval, q_target)
        # 输入32个评估值和32个目标值，使用均方损失函数
        self.optimizer.zero_grad()                                      # 清空上一步的残余更新参数值
        loss.backward()                                                 # 误差反向传播, 计算参数更新值
        self.optimizer.step()                                           # 更新评估网络的所有参数


    def train_net(self, train_data, epochs, criterion, batch_size, lr, scheduler, gamma, value_loss_coef, entropy_loss_coef, weight_decay, clip_grad_norm, betas, val_data=None, val_steps=100, min_iters=1000):
        LOG.info(f"Start training: {time.ctime()}")
        opt = torch.optim.Adam(self.net.parameters(),
                               lr=lr, betas=betas, weight_decay=weight_decay)

        def lambda_lr(epoch): return scheduler ** np.sqrt(epoch)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda_lr)
        train_dl = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, drop_last=False, shuffle=True, collate_fn=self.collate_fn, num_workers=0)
        iters = max(min_iters, int(epochs*len(train_dl)))
        di = iter(train_dl)
        for i in range(iters):
            try:
                x_batch, y_batch = next(di)
            except:
                di = iter(train_dl)
                x_batch, y_batch = next(di)
            x_batch, y_batch = to_device(
                x_batch, self.device), to_device(y_batch, self.device)
            pred = self.net(x_batch)
            pg_loss, value_loss, entropy_loss = criterion(
                pred, *y_batch, gamma=gamma)
            (pg_loss + value_loss_coef*value_loss
             + entropy_loss_coef*entropy_loss).backward()
            torch.nn.utils.clip_grad_norm_(
                self.net.parameters(), clip_grad_norm, norm_type=2.0)
            opt.step()
            opt.zero_grad()
            sched.step()

        LOG.info(
            f"""End training: {time.ctime()},
                Policy loss: {pg_loss.item():.2f},
                Value loss: {value_loss.item():.2f},
                Entropy loss {entropy_loss.item():.2f},
                {iters} iterations.""")
        return pg_loss.item(), value_loss.item(), entropy_loss.item()

class TrajectoryStorage():
    def __init__(self):
        self.memory = []
        self.MAX_CAPACITY = 500
        self.memory_idx = -1

    def set_env(self, env):
        self.env = env

    def store_transition(self,state, action, reward, next_state, is_done, next_mask):
        if(len(self.memory) < MEMORY_CAPACITY ):
            self.memory.append([state, action, reward, next_state, is_done, next_mask])
            self.memory_idx += 1
        else:
            self.memory_idx = (self.memory_idx+1) % MEMORY_CAPACITY
            self.memory[self.memory_idx] = [state, action, reward, next_state, is_done, next_mask]

    # 包括当前idx，往前update_count个更新成新的reward
    def update_reward(self, reward, update_count):
        idx = self.memory_idx
        while(update_count):
            if REWARD_UPDATE_METHOD == 0:
                self.memory[idx][2] = reward
            elif REWARD_UPDATE_METHOD == 1:
                self.memory[idx][2] = reward
                reward = reward * REWARD_UPDATE_COEF
            update_count -= 1
            idx -= 1

    def get_dataset_lastn(self, n=BATCH_SIZE):
        """Get last n trajectories"""
        n = max(len(self.memory), n)
        start = self.memory_idx-n
        dataset = []
        if start < 0:
            dataset.extend(self.memory[start:])
            dataset.extend(self.memory[:self.memory_idx+1])
        else:
            dataset.append(self.memory[start:self.memory_idx+1])
        return dataset
    def get_dataset_random(self, n = BATCH_SIZE):
        n = max(len(self.memory), n)
        indexes = np.random.choice(len(self.memory),n)
        dataset = []
        for index in indexes:
            dataset.append(self.memory[index])
        return dataset
        

def random_batch_splitter(sql_list, batch_size):
    """
    将给定的SQL语句列表按照指定的batch_size随机分割成多个子列表。

    Args:
        sql_list (list): 包含SQL语句的列表。
        batch_size (int): 每一批的大小。

    Returns:
        list: 包含随机排序并按batch_size分组后的SQL语句的列表。
    """
    if not sql_list or batch_size <= 0:
        return []

    # 创建一个副本以避免修改原列表
    shuffled_sql = sql_list.copy()
    random.shuffle(shuffled_sql)

    batches = []
    for i in range(0, len(shuffled_sql), batch_size):
        batches.append(shuffled_sql[i:i+batch_size])

    return batches

class GPRF():
    def __init__(self, baseline):
        self.baseline = baseline
        self.d = config.d
        self.env_config = config.env_config
        self.sql_names = list(self.env_config['db_data'].keys())
        self.env = Env()
        self.agent = Agent(self.env.eval_net,self.env.target_net)
        self.count = 0
    def run(self):
        for epoch in range(self.d['train_args']['epochs']):
        # 模拟实际情况，sql按批到来
            batches = random_batch_splitter(self.sql_names, config.d['sys_args']['sql_batch_size'])
            for batch_idx in range(len(batches)):
                batch = batches[batch_idx]
                for ep in range(self.d['train_args']['episodes']):
                    total_reward = 0
                    self.env.reset(batch)
                    state = self.env.get_state()
                    while not self.env.is_complete():
                        
                        mask = self.env.get_mask()
                        # state中包含三个部分，分别是全局plan状态（以树形表示），当前plan的编码（以树形表示），当前batch的编码（以向量表示）
                        action = self.agent.predict(state, mask)   
                        next_state ,reward, is_done, is_complete, next_mask= self.env.step(action)
                        self.agent.ts.store_transition(state, action, reward, next_state, is_done, next_mask) 
                        if is_done:
                            # 在plan的最终结果出来后更新前面的所有reward,update_count表示前面涉及到了几个join，总join数应是表数-1，出去最后一次，所以要-2
                            update_count = len(self.env.global_plan.singlePlans[-1].alias_to_table)-1
                            self.agent.ts.update_reward(reward, update_count)
                            self.agent.learn()
                            print(f'当前训练进度 epoch:{epoch} batch_idx:{batch_idx} episode:{ep} sql_name:{self.env.sql_names[self.env.plan_idx-1]}')
                        
                        state = next_state
                        # if is_done:
        

            
def to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device=device)
    if isinstance(obj, tuple) or isinstance(obj, list):
        if isinstance(obj[0], float) or isinstance(obj[0], int):
            return obj
        return [to_device(o, device=device) for o in obj]

def find_inner_join_actions(p):
    actions = []
    roots = p.get_roots()
    for i, n1 in enumerate(roots):
        for j, n2 in enumerate(roots):
            if i != j and p.is_inner_join(n1, n2):
                actions.append((i, j))
    return actions

def get_mask(p):
    size = len(p.get_roots())
    m = torch.zeros((size, size), dtype=bool)
    m[list(zip(*find_inner_join_actions(p)))] = 1
    return m


def evaluation_mode(model):
    '''Temporarily switch to evaluation mode. Keeps original training state of every submodule'''
    with torch.no_grad():
        train_state = dict((m, m.training) for m in model.modules())
        try:
            model.eval()
            yield model
        finally:
            # restore initial training state
            for k, v in train_state.items():
                k.training = v
