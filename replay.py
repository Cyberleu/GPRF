import numpy as np
import config
import random
class ReplayBuffer():
    def __init__(self):
        self.MAX_CAPACITY = config.d['train_args']['memory_capacity']
        self.NEED_REWARD_SHAPE = config.d['sys_args']['need_reward_shape']
        self.BATCH_SIZE = config.d['train_args']['batch_size']
        # store的transition先存在temp中，如果需要更新reward则在done时更新，并转储到buffer中
        self.temp = []
    def push(self,state, action, reward, next_state, is_done, next_mask, is_complete, error = 0):
        pass
    def update_reward(self):
        pass
    def sample(self):
        pass
    def update_error(self, indexes, error):
        pass


# 普通检验回放池，可选择最新的进行回放或进行uniform的sample
class NormalReplayBuffer(ReplayBuffer):
    def __init__(self):
        super().__init__()
        self.memory = []
        self.memory_idx = -1
        self.REWARD_UPDATE_METHOD = 0  # 0为更新成相同的reward 1为按比例减小reward
        self.REWARD_UPDATE_COEF = 0.9 
        self.GET_SAMPLE = 1 # 0为随机获取sample， 1为获取最新n个sample
    
    def push(self,state, action, reward, next_state, is_done, next_mask, is_complete, error = 0):
        transition = [state, action, reward, next_state, is_done, next_mask, is_complete]
        self.temp.append(transition)

        if is_done:
            if self.NEED_REWARD_SHAPE:
                reward = self.temp[-1][2]
                for i in range(len(self.temp)-2,-1,-1):
                    if self.REWARD_UPDATE_METHOD == 0:
                        self.temp[i][2] = reward
                    else:
                        self.temp[i][2] = reward * self.REWARD_UPDATE_COEF
                        reward = reward * self.REWARD_UPDATE_COEF
            for t in self.temp:
                self.push2buffer(t)
            self.temp.clear()

    def push2buffer(self,transition):
        if(len(self.memory) < config.d['train_args']['memory_capacity'] ):
            self.memory.append(transition)
            self.memory_idx += 1
        else:
            self.memory_idx = (self.memory_idx+1) % config.d['train_args']['memory_capacity']
            self.memory[self.memory_idx] = transition

    # 包括当前idx，往前update_count个更新成新的reward
    # def update_reward(self, reward, update_count):
    #     idx = self.memory_idx
    #     while(update_count):
    #         if self.REWARD_UPDATE_METHOD == 0:
    #             self.memory[idx][2] = reward
    #         elif self.REWARD_UPDATE_METHOD == 1:
    #             self.memory[idx][2] = reward
    #             reward = reward * self.REWARD_UPDATE_COEF
    #         update_count -= 1
    #         idx -= 1

    def sample(self):
        if self.GET_SAMPLE == 0:
            return self.get_dataset_random()
        else:
            return self.get_dataset_lastn()
    
    def update_error(self, indexes, error):
        pass


    def get_dataset_lastn(self):
        """Get last n trajectories"""
        batch_size = max(len(self.memory), self.BATCH_SIZE)
        start = self.memory_idx-batch_size
        dataset = []
        indexes = []
        if start < 0:
            dataset.extend(self.memory[start:])
            dataset.extend(self.memory[:self.memory_idx+1])
            indexes.extend(range(start, len(self.memory)))
            indexes.extend(range(0, self.memory_idx+1))
        else:
            dataset.append(self.memory[start:self.memory_idx+1])
            indexes.extend(range(start, self.memory_idx))
        return zip(*dataset),indexes, [1 for _ in range(len(indexes))]
    
    def get_dataset_random(self):
        batch_size = max(len(self.memory), self.BATCH_SIZE)
        indexes = np.random.choice(len(self.memory),batch_size)
        dataset = []
        for index in indexes:
            dataset.append(self.memory[index])
        return zip(*dataset), indexes, [1 for _ in range(indexes)]

# 优先经验回放，记录每个transition的TD error，error越大越容易被sample
class PrioritizedReplayBuffer(ReplayBuffer):#ReplayTree for the per(Prioritized Experience Replay) DQN. 
    def __init__(self):
        super().__init__()
        self.tree = SumTree(self.MAX_CAPACITY)  # 创建一个SumTree实例
        self.abs_err_upper = 1.  # 绝对误差上限
        self.epsilon = 0.01
        ## 用于计算重要性采样权重的超参数
        self.beta_increment_per_sampling = 0.001
        self.alpha = 0.6
        self.beta = 0.4 
        self.abs_err_upper = 1.

    def __len__(self):# 返回存储的样本数量
        return self.tree.total()

    def push(self, state, action, reward, next_state, is_done, next_mask, is_complete):#Push the sample into the replay according to the importance sampling weight
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        transition = (state, action, reward, next_state, is_done, next_mask, is_complete)
        self.tree.add(transition, max_p)         

    def sample(self):
        pri_segment = self.tree.total() / self.BATCH_SIZE
        priorities = []
        batch = []
        idxs = []
        is_weights = []
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(self.BATCH_SIZE):
            a = pri_segment * i
            b = pri_segment * (i+1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()
        return zip(*batch), idxs, is_weights
    
    def update_error(self, tree_idx, abs_errors):#Update the importance sampling weight
        abs_errors += self.epsilon
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
    


class SumTree:
    def __init__(self, capacity: int):
        # 初始化SumTree，设定容量
        self.capacity = capacity
        # 数据指针，指示下一个要存储数据的位置
        self.data_pointer = 0
        # 数据条目数
        self.n_entries = 0
        # 构建SumTree数组，长度为(2 * capacity - 1)，用于存储树结构
        self.tree = np.zeros(2 * capacity - 1)
        # 数据数组，用于存储实际数据
        self.data = np.zeros(capacity, dtype=object)

    def update(self, tree_idx, p):#更新采样权重
        # 计算权重变化
        change = p - self.tree[tree_idx]
        # 更新树中对应索引的权重
        self.tree[tree_idx] = p

        # 从更新的节点开始向上更新，直到根节点
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def add(self,data, p):#向SumTree中添加新数据
        # 计算数据存储在树中的索引
        tree_idx = self.data_pointer + self.capacity - 1
        # 存储数据到数据数组中
        self.data[self.data_pointer] = data
        # 更新对应索引的树节点权重
        self.update(tree_idx, p)

        # 移动数据指针，循环使用存储空间
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

        # 维护数据条目数
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def get_leaf(self, v):#采样数据
        # 从根节点开始向下搜索，直到找到叶子节点
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1
            cr_idx = cl_idx + 1
            # 如果左子节点超出范围，则当前节点为叶子节点
            if cl_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                # 根据采样值确定向左还是向右子节点移动
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        # 计算叶子节点在数据数组中的索引
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total(self):
        return int(self.tree[0])
