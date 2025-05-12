import numpy as np
import psycopg2
from gym.spaces import Discrete
import torch
from db_utils import *
from plan import SinglePlan
from plan import GlobalPlan
from net import Net
import re
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import numpy as np
import psycopg2
import torch
from torch_geometric.data import Data
import sys
import config
import random
from collections import Counter
from encoder import *

sys.setrecursionlimit(10000)
class Env():
    def __init__(self):
        self.conn = config.conn
        self.scheme = config.env_config['scheme']
        self.db_data = config.env_config['db_data']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rel_to_idx = {rel: i for i, rel in enumerate(
            self.scheme)}  # idx = obj[rel]
        self.rels = list(self.rel_to_idx.keys())
        self.N_rels = len(self.rels)
        self.col_to_idx = dict()  # idx = obj[rel][col]
        self.N_cols = 0
        self.cols = []
        for rel in self.scheme:
            self.col_to_idx[rel] = {}
            for col in self.scheme[rel]:
                self.col_to_idx[rel][col] = self.N_cols
                self.N_cols += 1
                self.cols.append((rel, col))

        self.actions = [(r1, r2) for r1 in range(self.N_rels)
                        for r2 in range(self.N_rels) if r1 != r2]
        self.action_ids = {a: i for i, a in enumerate(self.actions)}
        self.action_space = Discrete(len(self.actions))
        self.plans = []
        # 正在处理的plan下标
        self.plan_idx = -1
        self.plan = SinglePlan()
        self.global_plan = GlobalPlan()
        # 记录每个query的cost
        self.query_costs = {}
        self.view_costs = {}
        self.eval_net = Net(d_out = self.N_rels * self.N_rels).to(self.device)
        self.target_net = Net(d_out = self.N_rels * self.N_rels).to(self.device)
        self.plan_encoder = PlanEncoder(self)
        self.query_encoder = QueryEncoder(self)
        self.sql_names = []
        self.tables_encode = torch.zeros((0, self.N_rels)).to(self.device)
        # 记录batch执行的最短时间,在reset时不更新
        self.db_data = config.env_config['db_data']
        self.batch_cost = dict() # {(sql_name, sql_name...) : (total_cost, total_time)}
        self.batch_dict = {}

    def get_obs(self):
        """
        Constructs an observation based on the plan.
        """
        raise NotImplementedError

    def from_action(self, action):
        """
        Retrieve relations to join from action.
        """
        r1, r2 = action
        n1 = self.plan.rel_to_node(self.rels[r1])
        n2 = self.plan.rel_to_node(self.rels[r2])
        return n1, n2

    def valid_actions(self):
        """
        Computes valid actions.
        Any nodes in plan can be joined
        """
        relids = self.plan.query_tables
        valid_actions_ids = [
            self.action_ids[(r1, r2)] for r1 in relids for r2 in relids if r1 != r2]
        valid_actions = np.zeros(len(self.actions))
        valid_actions[valid_actions_ids] = 1.
        return valid_actions

    # 单各plan是否完成
    def is_done(self):
        return self.plan.is_complete
    
    # 总体是否完成
    def is_complete(self):
        return self.plan_idx == len(self.plans)
    
    def step2(self, action):
        self.current_step += 1
        table1, table2 = self.rels[action[0]], self.rels[action[1]]
        node1, node2 = self.plan.action_to_join((table1, table2))
        self.plan.join(node1, node2)
        is_done = self.is_done()
        if is_done:
            self.global_plan.merge(deepcopy(self.plan))
        reward = random.randint(-100, 100)
        if is_done:
            self.plan_idx += 1
            if self.plan_idx < len(self.plans):
                self.plan = self.plans[self.plan_idx]
        next_state = self.get_state()
        next_mask = self.get_mask()
        return next_state ,reward, is_done, next_mask, self.is_complete()

    def step(self, action):
        self.current_step += 1
        table1, table2 = self.rels[action[0]], self.rels[action[1]]
        node1, node2 = self.plan.action_to_join((table1, table2))
        self.plan.join(node1, node2)
        is_done = self.is_done()
        if is_done:
            self.global_plan.merge(deepcopy(self.plan))
        # reward = self.reward(exec_time = False)
        reward = self.reward_subtree()
        if is_done:
            self.plan_idx += 1
            if self.plan_idx < len(self.plans):
                self.plan = self.plans[self.plan_idx]
        next_state = self.get_state()
        next_mask = self.get_mask()
        # if(reward > 0):
        #     reward = reward *5
        return next_state ,reward, is_done, next_mask, self.is_complete()

    def reset(self, batch):
        self.global_plan.reset()
        self.sql_names = batch
        self.plans.clear()
        self.query_costs.clear()
        for sql_name in batch:
            self.plans.append(SinglePlan(*self.db_data[sql_name]))
        self.plan_idx = 0
        self.current_step = 0
        self.plan = self.plans[0]
        # 建立该batch的全局编码
        self.tables_encode = torch.zeros((0, self.N_rels)).to(self.device)
        for plan in self.plans:
            pos = torch.zeros(self.N_rels).to(self.device)
            for _, table in plan.alias_to_table.items():
                pos[self.rel_to_idx[table]] = pos[self.rel_to_idx[table]] + 1
            self.tables_encode = torch.vstack((self.tables_encode, pos))
  
        # 删除掉上一轮所有的物化视图（只有训练中使用，实际环境下可按照算法保留上一批的物化视图）
        drop_all_materialized_views()
        drop_all_views()
    
    # mask的shape是N_rels*N_rels,只当前状态下哪两个表之间可以连接就为1
    def get_mask(self):
        m = torch.zeros((self.N_rels, self.N_rels), dtype=bool).to(self.device)
        roots = self.plan.get_roots()
        for i, n1 in enumerate(roots):
            for j, n2 in enumerate(roots):
                if i != j:
                    join_list = self.plan.get_joinable_list(n1, n2)
                    for join in join_list:  
                        m[self.rel_to_idx[join[0]]][self.rel_to_idx[join[1]]] = 1
        if not m.any():
            print(1)
        return m
                           
    # def get_mask(self):
    #     def find_inner_join_actions(p):
    #         actions = []
    #         roots = p.get_roots()
    #         for i, n1 in enumerate(roots):
    #             for j, n2 in enumerate(roots):
    #                 if i != j and p.is_inner_join(n1, n2):
    #                     actions.append((i, j))
    #         return actions
    #     m = torch.zeros((self.N_rels, self.N_rels), dtype=bool)
    #     m[list(zip(*find_inner_join_actions(self.plan)))] = 1
    #     return m
    
    # 通过增量的方式获取GlobalPlan的cost,判断新加的plan是否能共享
    def get_cost(self, exec_time = False):
        root = self.global_plan.roots[-1]
        sql_name = self.global_plan.singlePlans[-1].sql_name
        shared_nodes = []
        # 因为已经merge，所以只需要获取所有包含该root的share_list即可。
        for node, node_data in self.global_plan.G.nodes.items():
            if node_data['type'] == 'Share' and root in node_data['share_list']:
                shared_nodes.append(node)
        if(len(shared_nodes) == 0):
            # 无共享节点， 则直接加上该sql的cost
            query_sql = self.global_plan.generate_sql(root, False)
            query_cost, query_time = get_cost_from_db(query_sql, conn=self.conn, exec_time=exec_time,is_view=False, baseline_cost=config.env_config['db_data'][sql_name][-2], baseline_time=config.env_config['db_data'][sql_name][-1])
            self.query_costs[root] = query_time if exec_time else query_cost
        else:
            # 可以进行共享，考虑1. 视图已经存在，无需再生成 2. 视图不存在
            # 当生成视图时可能会对前面已经生成的query有影响，因此要重新获取cost
            influcned_roots = set()
            for node in shared_nodes:
                share_list = self.global_plan.G.nodes[node]['share_list']
                assert len(share_list) >= 2
                if len(share_list) == 2:
                    view_sql = self.global_plan.generate_sql(node, True)
                    view_cost, view_time = get_cost_from_db(view_sql, self.conn, is_view = True, exec_time=True)
                    self.view_costs[node] = view_time if exec_time else view_cost
                    influcned_roots.add(share_list[0])
            influcned_roots.add(root)
            for node in list(influcned_roots):
                index = self.global_plan.roots.index(node)
                sql_name = self.global_plan.singlePlans[index].sql_name
                query_sql = self.global_plan.generate_sql(node, False)
                query_cost, query_time = get_cost_from_db(query_sql, self.conn, exec_time = exec_time,baseline_cost=config.env_config['db_data'][sql_name][-2], baseline_time=config.env_config['db_data'][sql_name][-1])
                self.query_costs[query_sql] = query_time if exec_time else query_cost
        # TODO:sum(list(self.view_costs.values())) 
        return sum(list(self.query_costs.values())), query_sql
    
    # def get_tables_encode(self):

    # 返回新增的plan带来的新增子树中的table数量
    def get_subtree_delta(self):
        root = self.global_plan.roots[-1]
        return self.global_plan.get_shared_node_entry_num(root)
    
    def get_state(self):
        return self.plan_encoder.encode(self.global_plan), self.plan_encoder.encode(self.plan), self.tables_encode
        
    def reward(self, exec_time = False):
        if not self.is_done():
            return 0
        else:
            cost, new_sql = self.get_cost(exec_time=exec_time)
            baseline_cost = 0
            batch_sql_names = frozenset(self.sql_names[:self.plan_idx+1])
            # batch_cost中保存的是经验池，如果不在batch_cost中则直接按db_data中的cost之和当作baseline
            if batch_sql_names not in self.batch_cost:
                for sql_name in batch_sql_names:
                    baseline_cost += self.db_data[sql_name][-1] if exec_time else self.db_data[sql_name][-2]
            else:
                baseline_cost = self.batch_cost[batch_sql_names][1] if exec_time else self.batch_cost[batch_sql_names][0]
            reward = - np.log(cost/baseline_cost)
            return reward, new_sql
    
    # 该reward只考虑公共子树的数量和子树的大小
    def reward_subtree(self):
        if not self.is_done():
            return 0
        count = self.get_subtree_delta()
        # 如果没有公共子树则惩罚
        if count == 0:
            return -5
        else:
            return count
            # # 将table转化为a-u的单个字符
            # batch = []
            # for plan in self.global_plan.singlePlans:
            #     sequence = ''
            #     for table in plan.query_tables:
            #         sequence += chr(ord('a') + self.rel_to_idx[table])
            #     batch.append(sequence)
            # pos_key = CanonicalBatch(batch)
            # # 当前key的count只有大于等于经验池中所有子集的count时才进行更新,惩罚量为与最优的子集的差值
            # flag = False
            # min_count = 999
            # for key in self.batch_dict:
            #     if key.is_subset_of(pos_key) and count <= self.batch_dict[key]:
            #         flag = True
            #         min_count = min(self.batch_dict[key], min_count)
            # if flag:
            #     return min_count-count
            # else:
            #     self.batch_dict[pos_key] = count
            #     return count
        

    def render(self):
        return self.plan.render()

    def find_cost(self, p):
        return list(p.G.nodes(data=True))[-1][-1]['cost'][0]




# 支持batch的等价性和子集判断
    
class CanonicalBatch:
    """支持哈希等价和子集判断的自定义字典键"""
    
    def __init__(self, batch):
        # 规范化步骤：每个字符串字符排序 + 整个batch排序
        self._sorted_batch = tuple(sorted(tuple(sorted(s)) for s in batch))
        
        # 预计算哈希值
        self._hash = hash(self._sorted_batch)
        
        # 生成字符计数器列表（用于子集判断）
        self._counters = [Counter(s) for s in self._sorted_batch]
    
    def __hash__(self):
        return self._hash
    
    def __eq__(self, other):
        return self._sorted_batch == other._sorted_batch
    
    def is_subset_of(self, other) -> bool:
        """判断当前batch是否是另一个batch的子集"""
        # 优化1：先按特征排序（字符数量降序，字符种类排序）
        self_sorted = sorted(self._counters, 
                           key=lambda c: (-sum(c.values()), sorted(c)))
        other_sorted = sorted(other._counters, 
                            key=lambda c: (-sum(c.values()), sorted(c)))
        
        # 优化2：使用贪心算法匹配最佳候选
        matched = [False] * len(other_sorted)
        for c_self in self_sorted:
            found = False
            # 优先匹配字符数量多的目标
            for i, c_other in enumerate(other_sorted):
                if not matched[i] and self._contains(c_other, c_self):
                    matched[i] = True
                    found = True
                    break
            if not found:
                return False
        return True
    
    @staticmethod
    def _contains(counter_a, counter_b) -> bool:
        """判断counter_a是否包含counter_b的所有字符（数量足够）"""
        for char, count in counter_b.items():
            if counter_a[char] < count:
                return False
        return True

    def __repr__(self):
        return f"BatchKey({self._sorted_batch})"
        


import json
if __name__ == '__main__':
    p = build_and_save_optimizer_plan("/data/homedata/lch/GPRF/1.sql")
    with open('/data/homedata/lch/GPRF/data/postgres_env_config.json', "r") as f:
        env_config = json.load(f)
    # pe = PlanEncoder(env_config)
    # encoded_p = pe.encode(p)
    qe = QueryEncoder(env_config)
    encoded_p = qe.encode(p)
    print(1)

