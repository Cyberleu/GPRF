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
import matplotlib.pyplot as plt
import numpy as np
from replay import *

import logging
LOG = logging.getLogger(__name__)
logging.basicConfig(
    filename= config.d['sys_args']['reward_log_path'],    # 日志文件名
    filemode='w',          # 文件模式：'a'为追加（默认），'w'为覆盖
    level=logging.INFO,   # 最低日志级别（DEBUG及以上均记录）
    format='%(asctime)s - %(levelname)s - %(message)s',  # 自定义格式
    datefmt='%Y-%m-%d %H:%M:%S'  # 时间格式
)

INF = 1e9
MEMORY_CAPACITY = 500                          # 记忆库容量
BATCH_SIZE = 8   
GET_SAMPLE = 1  # 0为随机获取sample， 1为获取最新n个sample
REWARD_UPDATE_METHOD = 0 # 0为更新成相同的reward 1为按比例减小reward
REWARD_UPDATE_COEF = 0.9
GAMMA = 0.9
LR = 0.01

class Agent(nn.Module):
    def __init__(self, eval_net, target_net, eps=0.2, device = 'cuda'):
        super().__init__()
        self.eval_net = eval_net
        self.target_net = target_net
        self.learn_step_counter = 0
        self.device = device
        self.eps = eps
        self.loss_func = nn.MSELoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(),lr=LR)
        if config.d['sys_args']['use_per']:
            self.er = PrioritizedReplayBuffer()
        else:
            self.er = NormalReplayBuffer()
        self.target_replace_iter = config.d['train_args']['target_replace_iter']

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
        probs = masked_logit.softmax(-1).detach().cpu().numpy()
        if np.random.uniform() < self.eps:
            action_idx = np.random.choice(range(0,len(probs)), 1, p=probs)[0]
        else :
            action_idx = np.argmax(probs)
        action = np.unravel_index(action_idx, mask_dim)
        assert mask[action[0]][action[1]] == True
        return action
    
    def store_transition(self, state, action, reward, next_state, is_done, next_mask, is_complete):
        self.er.push(state, action, reward, next_state, is_done, next_mask, is_complete)

    def learn(self):
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())   # 将评价网络的权重参数赋给目标网络
        self.learn_step_counter +=1                 # 目标函数的学习次数+1
        
        # 抽buffer中的数据学习
        data = self.re.sample()
        # if GET_SAMPLE == 0:
        #     data = self.ts.get_dataset_random(BATCH_SIZE)
        # elif GET_SAMPLE == 1:
        #     data = self.ts.get_dataset_lastn(BATCH_SIZE)
        length = len(data)
        mask_length = data[0][5].shape[0]
        # batch state
        b_s = [row[0] for row in data]
        # batch action
        b_a = [[row[1][0] *  mask_length + row[1][1]] for row in data]
        # batch reward
        b_r  = [row[2] for row in data]
        b_r = torch.tensor(b_r, dtype=torch.float32).to(self.device)
        # batch next state
        b_ns = [row[3] for row in data]
        b_d = [row[4] for row in data]
        # batch next mask
        b_nm = torch.empty((0,mask_length, mask_length), dtype=bool).to(self.device)
        for row in data:
            b_nm = torch.concat((b_nm, row[5].unsqueeze(0)))
        # batch complete mask
        b_cm = torch.empty(0, dtype=bool).to(self.device)
        for row in data:
            b_cm = torch.concat((b_cm, torch.tensor([row[6]], dtype = bool).to(self.device)))
        
        
        q_eval = self.eval_net(b_s).gather(1, torch.tensor(b_a).to(self.device)).view(-1)
        
        logit = self.target_net(b_ns)
        q_next = torch.where(b_nm.view(length,-1).to(logit.device),
                                   logit, torch.tensor(float("-inf")).to(logit.device)).detach()
        q_target_temp = b_r+GAMMA * q_next.max(1)[0]
        # 处理is_complete的情况，防止出现全是-inf的情况
        q_target = torch.where(b_cm,
                          b_r, q_target_temp)
        # q_target = b_r + GAMMA * q_next.max(1)[0]
        # q_next.max(1)[0]表示只返回每一行的最大值，不返回索引(长度为32的一维张量)；.view()表示把前面所得到的一维张量变成(BATCH_SIZE, 1)的形状；最终通过公式得到目标值
        loss = self.loss_func(q_eval, q_target)
        # 输入32个评估值和32个目标值，使用均方损失函数
        self.optimizer.zero_grad()                                      # 清空上一步的残余更新参数值
        loss.backward()                                                 # 误差反向传播, 计算参数更新值
        self.optimizer.step()                                           # 更新评估网络的所有参数

    def learn(self):
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())   # 将评价网络的权重参数赋给目标网络
        self.learn_step_counter +=1                 # 目标函数的学习次数+1
        
        # 抽buffer中的数据学习
        batch, indexes,weights = self.er.sample()
        b_s, b_a_orig, b_r, b_ns, _, b_nm_orig, b_cm_orig = batch
        length = len(b_s)
        mask_length = b_nm_orig[0].shape[0]
        # batch action
        b_a = [[row[0] * mask_length + row[1]]for row in b_a_orig]
        b_r = torch.tensor(b_r, dtype=torch.float32).to(self.device)
        # batch next mask
        b_nm = torch.empty((0,mask_length, mask_length), dtype=bool).to(self.device)
        for row in b_nm_orig:
            b_nm = torch.concat((b_nm, row.unsqueeze(0)))
        # batch complete mask
        b_cm = torch.empty(0, dtype=bool).to(self.device)
        for row in b_cm_orig:
            b_cm = torch.concat((b_cm, torch.tensor([row], dtype = bool).to(self.device)))
        
        
        q_eval = self.eval_net(b_s).gather(1, torch.tensor(b_a).to(self.device)).view(-1)
        
        logit = self.target_net(b_ns)
        q_next = torch.where(b_nm.view(length,-1).to(logit.device),
                                   logit, torch.tensor(float("-inf")).to(logit.device)).detach()
        q_target_temp = b_r+GAMMA * q_next.max(1)[0]
        # 处理is_complete的情况，防止出现全是-inf的情况
        q_target = torch.where(b_cm,
                          b_r, q_target_temp)
        # q_target = b_r + GAMMA * q_next.max(1)[0]
        # q_next.max(1)[0]表示只返回每一行的最大值，不返回索引(长度为32的一维张量)；.view()表示把前面所得到的一维张量变成(BATCH_SIZE, 1)的形状；最终通过公式得到目标值
        loss = (torch.FloatTensor(weights).to(self.device) * self.loss_func(q_eval, q_target)).mean()
        # 输入32个评估值和32个目标值，使用均方损失函数
        self.optimizer.zero_grad()                                      # 清空上一步的残余更新参数值
        loss.backward()                                                 # 误差反向传播, 计算参数更新值
        self.optimizer.step() 

        abs_errors = torch.abs(q_eval - q_target).detach().cpu().numpy().squeeze()
        self.er.update_error(indexes, abs_errors)  # 更新经验的优先级

        self.learn_step_counter += 1


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
        self.agent = Agent(self.env.eval_net,self.env.target_net, device = self.env.device)
        self.count = 0
    def run(self):
        for epoch in range(self.d['train_args']['epochs']):
        # 模拟实际情况，sql按批到来
            batches = random_batch_splitter(self.sql_names, config.d['sys_args']['sql_batch_size'])
            for batch_idx in range(len(batches)):
                batch = batches[batch_idx]
                rewards = []
                for ep in range(self.d['train_args']['episodes']):
                    total_reward = 0
                    self.env.reset(batch)
                    state = self.env.get_state()
                    while not self.env.is_complete():
                        
                        mask = self.env.get_mask()
                        # state中包含三个部分，分别是全局plan状态（以树形表示），当前plan的编码（以树形表示），当前batch的编码（以向量表示）
                        action = self.agent.predict(state, mask)   
                        next_state ,reward, is_done,  next_mask, is_complete= self.env.step(action)
                        self.agent.store_transition(state, action, reward, next_state, is_done, next_mask, is_complete) 
                        total_reward += reward
                        if is_done:
                            # 在plan的最终结果出来后更新前面的所有reward,update_count表示前面涉及到了几个join，总join数应是表数-1，出去最后一次，所以要-2
                            # update_count = len(self.env.global_plan.singlePlans[-1].alias_to_table)-1
                            # self.agent.ts.update_reward(reward, update_count)
                            self.agent.learn()
                            print(f'当前训练进度 epoch:{epoch} batch_idx:{batch_idx} episode:{ep} sql_name:{self.env.sql_names[self.env.plan_idx-1]}')

                            # ## mod by dhp
                            # sql_name = self.env.sql_names[self.env.plan_idx-1]
                            # safe_sql_name = re.sub(r'[\/:*?"<>|]', '_', sql_name)  # 替换非法文件名字符
                            # # 目标目录
                            # output_dir = "./output4dhp1"
                            # os.makedirs(output_dir, exist_ok=True)  # 自动创建目录
                            # # 目标文件路径
                            # file_path = os.path.join(output_dir, f"{safe_sql_name}.txt")
                            # with open(file_path, "a", encoding="utf-8") as f:  # "a" 表示追加模式
                            #     f.write(new_sql + "\n")  # 多个 SQL 之间留空行
                        
                        state = next_state
                        if is_complete:
                            logging.info(f'epoch:{epoch} batch_idx:{batch_idx} episode:{ep} total_reward:{total_reward} final_reward:{reward}')
                            rewards.append(total_reward)
                x = np.linspace(0,self.d['train_args']['episodes'] ,1)  # X轴数据（0-10，100个点）
                y = np.array(rewards)                # Y轴数据（正弦曲线）

                # 创建画布
                plt.figure(figsize=(10, 6))  # 设置画布尺寸（宽10英寸，高6英寸）

                # 绘制折线
                plt.plot(x, y, 
                        color='#1f77b4',   # 线条颜色（十六进制）
                        linestyle='-',     # 实线
                        linewidth=2,       # 线宽
                        label='正弦曲线')   # 图例标签 
                plt.savefig(f'reward{batch_idx}.png')

        

            
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = torch.zeros((size, size), dtype=bool).to(device)
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
