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

sys.setrecursionlimit(10000)
class Env():
    def __init__(self):
        self.conn = config.conn
        self.scheme = config.env_config['scheme']
        self.db_data = config.env_config['db_data']
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
        self.net = Net(d_out = self.N_rels * self.N_rels)
        self.plan_encoder = PlanEncoder(self)
        self.query_encoder = QueryEncoder(self)
        self.sql_names = []
        self.tables_encode = torch.tensor((0, self.N_rels))
        

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

    def step(self, action):
        self.current_step += 1
        table1, table2 = self.rels[action[0]], self.rels[action[1]]
        node1, node2 = self.plan.action_to_join((table1, table2))
        self.plan.join(node1, node2)
        im = self.plan.render()
        im.save(f'/data/homedata/lch/GPRF/im.png') 
        is_done = self.is_done()
        if is_done:
            self.global_plan.merge(self.plan)
        reward = self.reward()
        if is_done:
            self.plan_idx += 1
            if self.plan_idx < len(self.plans):
                self.plan = self.plans[self.plan_idx]
        return reward, is_done, self.is_complete()

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
        for plan in self.plans:
            pos = torch.zeros(self.N_rels)
            for _, table in plan.alias_to_tables.items():
                pos[table] = pos[table] + 1
            torch.vstack((self.tables_encode, pos))
  
        # 删除掉上一轮所有的物化视图（只有训练中使用，实际环境下可按照算法保留上一批的物化视图）
        drop_all_materialized_views()
    
    # mask的shape是N_rels*N_rels,只当前状态下哪两个表之间可以连接就为1
    def get_mask(self):
        m = torch.zeros((self.N_rels, self.N_rels), dtype=bool)
        roots = self.plan.get_roots()
        for i, n1 in enumerate(roots):
            for j, n2 in enumerate(roots):
                if i != j:
                    join_list = self.plan.get_joinable_list(n1, n2)
                    for join in join_list:  
                        m[self.rel_to_idx[join[0]]][self.rel_to_idx[join[1]]] = 1
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
        shared_nodes = self.global_plan.lookup_by_value_within_node(root, 'type', 'Share')
        if(len(shared_nodes) == 0):
            # 无法进行共享， 则直接加上该sql的cost
            query_sql = self.global_plan.generate_sql(root, False)
            query_cost = get_cost_from_db(query_sql, self.conn, exec_time)
            self.query_costs[root] = query_cost
        else:
            # 可以进行共享，考虑1. 视图已经存在，无需再生成 2. 视图不存在
            # 当生成视图时可能会对前面已经生成的query有影响，因此要重新获取cost
            influcned_roots = set()
            for node in shared_nodes:
                share_list = self.global_plan.G.nodes[node]['share_list']
                assert len(share_list) >= 2
                if len(share_list) == 2:
                    view_sql = self.global_plan.generate_sql(node, True)
                    view_cost = get_cost_from_db(view_sql, self.conn, is_view = True, exec_time=True)
                    self.view_costs[node] = view_cost
                    influcned_roots.add(share_list[0])
            influcned_roots.add(root)
            for node in list(influcned_roots):
                query_sql = self.global_plan.generate_sql(node, False)
                query_cost = get_cost_from_db(query_sql, self.conn, exec_time = exec_time)
                self.query_costs[query_sql] = query_cost
        return sum(list(self.view_costs.values())) + sum(list(self.query_costs.values()))
    
    def get_state(self):
        return self.plan_encoder.encode(self.global_plan), self.plan_encoder.encode(self.plan), self.tables_encode
        
    def reward(self, ):
        if not self.is_done():
            return None
        else:
            return self.get_cost()
        

    def render(self):
        return self.plan.render()

    def find_cost(self, p):
        return list(p.G.nodes(data=True))[-1][-1]['cost'][0]

# 主要是plan level和query level 的encoding
class PlanEncoder():
    def __init__(self, env):
        self.env = env
    
    # 返回矩阵和边信息，不是一个向量
    def encode(self, plan):
        
        if len(plan.roots) == 0:
            return None
        plan = self.node_encode(plan)
        # 获取所有节点并排序，确保顺序一致
        nodes = list(plan.G.nodes())
        node_idx = {node: idx for idx, node in enumerate(nodes)}
        node_vecs = np.zeros((0, list(plan.G.nodes.values())[0]['feature'].shape[0]))
        for node_index in nodes:
            node_vecs = np.vstack((node_vecs, plan.G.nodes[node_index]['feature'].reshape(1,-1)))
        
        # 处理边信息
        edges = list(plan.G.edges())
        sources = [node_idx[e[0]] for e in edges]
        targets = [node_idx[e[1]] for e in edges]

        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        x = torch.tensor(node_vecs, dtype=torch.float32)
        return Data(x=x, edge_index=edge_index)

    def node_encode(self, plan):
        feature_list = ['type', 'tables']
        for node_index, node_data in plan.G.nodes.items():
            node_data = plan.G.nodes[node_index]
            # 每个节点所需编码信息，type为one-hot编码，表示该节点的类型。tables表示该节点涉及到表（若为scan则为one-hot+0向量，join/share为one-hot+one-hot）
            if not "feature" in node_data:
                encoded_vec = np.array([])
                for feature in feature_list:
                    if feature == 'type':
                        vec = np.zeros(3)
                        if node_data['type'] == 'Scan':
                            vec[0] = 1
                        elif node_data['type'] == 'Join':
                            vec[1] = 1
                        elif node_data['type'] == 'Share':
                            vec[2] = 1
                        encoded_vec = np.concatenate((encoded_vec, vec), axis = 0)
                    elif feature == 'tables':
                        vec1 = np.zeros(len(self.env.rels))
                        vec2 = np.zeros(len(self.env.rels))
                        
                        if(node_data['type'] == 'Scan'):
                            vec1[self.env.rel_to_idx[plan.alias_to_table[list(node_data['table_entries'])[0]]]] = 1
                        else: 
                            for cond in node_data['conds']:
                                vec1[self.env.rel_to_idx[plan.alias_to_table[cond['left_entry_name']]]] = 1
                                vec2[self.env.rel_to_idx[plan.alias_to_table[cond['right_entry_name']]]] = 1
                        encoded_vec = np.concatenate((encoded_vec, vec1, vec2), axis = 0)
                plan.G.nodes[node_index]['feature'] = encoded_vec
        return plan        

        
    

class QueryEncoder():
    def __init__(self, env):
        self.query_encoding_size = int(env.N_rels * (env.N_rels - 1) / 2)
        self._initial_query = None
        self.cardinalities = config.env_config.get('cardinalities')
        self.condition_selectivity = config.env_config.get('condition_selectivity')
        self.tables_features = config.env_config.get('tables_features')
        self.index_features = config.env_config.get('index_features')

    def compute_cardinalities(env_config):
        # conn = config.(env_config["psycopg_connect_url"])
        conn = config.conn
        cursor = conn.cursor()
        env_config["cardinalities"] = {}
        for q, info in env_config["db_data"].items():
            card = defaultdict(int)
            d = defaultdict(list)
            for cond in info[2]:
                if len(cond['names']) == 1:
                    d[cond['names'][0]].append(cond['condition'])
            for table in info[1].keys():
                conds = d[table]
                if len(conds) == 0:
                    d[table] = 1
                    continue
                query1 = f"SELECT count(*) FROM {info[1][table]} AS {table} WHERE {' AND '.join(conds)};"
                query2 = f"SELECT count(*) FROM {info[1][table]}"
                cursor.execute(query1)
                num_after = cursor.fetchall()[0][0]
                # cursor.execute(query2)
                # num_before = cursor.fetchall()[0][0]
                # d[table] = num_after/(num_before+1e-10)
                card[table] = np.log(num_after)
            env_config["cardinalities"][info[3]] = card
        cursor.close()
        conn.close()

    def compute_condition_selectivities(env_config):
        conn = config.conn
        cursor = conn.cursor()
        env_config["condition_selectivity"] = {}
        for q, info in env_config["db_data"].items():
            condition_selectivity = defaultdict(int)
            for cond in info[2]:
                if len(cond['names']) == 1:
                    als = cond['names'][0]
                    quer1 = f"EXPLAIN (FORMAT JSON) SELECT * FROM {info[1][als]} AS {als}"
                    quer2 = f"EXPLAIN (FORMAT JSON) SELECT * FROM {info[1][als]} AS {als} WHERE {cond['condition']}"
                    cursor.execute(quer1)
                    num_rows_1 = cursor.fetchall()[0][0][0]['Plan']['Plan Rows']
                    cursor.execute(quer2)
                    num_rows_2 = cursor.fetchall()[0][0][0]['Plan']['Plan Rows']
                    condition_selectivity[cond['condition']] = num_rows_2/(num_rows_1 + 1e-6)
            env_config["condition_selectivity"][info[3]] = condition_selectivity
        cursor.close()
        conn.close()

    def get_predicates_encoding(self,plan):
        if self.condition_selectivity is None:
            return self.get_predicates_ohe(plan)
        else:
            return self.get_predicates_selectivity(plan)

    def get_predicates_selectivity(self, plan):
        column_preicates_vector = np.zeros(self.N_cols)
        selectivities = self.condition_selectivity[plan.initial_query]
        for aliases, conditions in plan._query_join_conditions.items():
            if len(aliases) != 1:
                continue
            alias = next(iter(aliases))
            tab_name = plan.alias_to_table[alias]
            for condition in conditions:
                col_name = self._parse_condition(condition)[0][-1]
                idx = self.col_to_idx[tab_name][col_name]
                # # one hot
                # column_preicates_vector[idx] = 1
                # selectivity
                column_preicates_vector[idx] += selectivities[condition]
        return column_preicates_vector


    def get_predicates_ohe(self, plan):
        column_preicates_vector = np.zeros(self.N_cols)
        for condition in plan.query_join_conditions:
            if len(condition['names']) == 1:
                tab_name = plan.alias_to_table[condition['names'][0]]
                idx = self.col_to_idx[tab_name][condition['condition']['col']]
                column_preicates_vector[idx] = 1
            else:
                tab_name1 = plan.alias_to_table[condition['names'][0]]
                idx = self.col_to_idx[tab_name1][condition['condition']['left_col_name']]
                column_preicates_vector[idx] = 1
                tab_name2 = plan.alias_to_table[condition['names'][1]]
                idx = self.col_to_idx[tab_name2][condition['condition']['right_col_name']]
                column_preicates_vector[idx] = 1
            # predicates = self._parse_condition(condition['condition'])
            # for alias, col_name in predicates:
            #     tab_name = plan.alias_to_table[alias]
            #     idx = self.col_to_idx[tab_name][col_name]
            #     column_preicates_vector[idx] = 1
        return column_preicates_vector


    def get_join_graph_encoding(self, plan):
        join_graph_matrix = np.zeros((self.N_rels, self.N_rels))
        for tabs in plan._query_join_conditions.keys():
            if len(tabs) == 2:
                tab1, tab2 = tabs
                tab1, tab2 = plan.alias_to_table[tab1], plan.alias_to_table[tab2]
                tab1_idx, tab2_idx = self.rel_to_idx[tab1], self.rel_to_idx[tab2]
                join_graph_matrix[[tab1_idx, tab2_idx],
                                  [tab2_idx, tab1_idx]] = 1
        return join_graph_matrix[np.triu_indices(self.N_rels, 1)]

    def encode(self, plan):
        if (plan.initial_query != self._initial_query):
            self._initial_query = plan.initial_query
            self.compute_query_enc(plan)
        features = [self.join_graph_encoding, self.predicate_ohe]
        if self.tables_features is not None:
            features.append(self.tables_features.flatten())
            features.append(self.index_features.flatten())
        return np.concatenate(features)

    def compute_query_enc(self, plan):
        self.predicate_ohe = self.get_predicates_encoding(plan)
        self.join_graph_encoding = self.get_join_graph_encoding(plan)

    def _parse_condition(self, condition: str) -> ((str, str), (str, str)):
        """Helper for parsing query condition
        Will return (table_name_1, column_name_1), (table_name_2, column_name_2)
        """
        return re.findall(r'([a-zA-Z][a-zA-Z0-9_-]*)\.{1}([a-zA-Z][a-zA-Z0-9_-]*)', condition)

    def compute_data_driven_features(env_config):
        conn = config.conn
        cursor = conn.cursor()
        cursor.execute('''
            SELECT seq_scan, seq_tup_read, n_tup_ins, n_tup_upd, n_tup_del,
            n_tup_hot_upd, n_live_tup, n_dead_tup, n_mod_since_analyze, vacuum_count, autovacuum_count
            FROM pg_stat_user_tables;
            ''')
        tables_features = cursor.fetchall()
        cursor.close()

        tables_features = np.array(tables_features)
        tables_features[tables_features == None] = 0

        dd_table_features_scaler = StandardScaler()
        dd_table_features_scaler.fit(tables_features)

        env_config['tables_features'] = dd_table_features_scaler.transform(
            tables_features
        )

        cursor = conn.cursor()
        cursor.execute("""
            SELECT idx_scan, idx_tup_read, idx_tup_fetch
            FROM pg_stat_user_indexes
            WHERE schemaname='public';
            """)
        index_features = cursor.fetchall()
        cursor.close()

        index_features = np.array(index_features)
        index_features[index_features == None] = 0

        dd_index_features_scaler = StandardScaler()
        dd_index_features_scaler.fit(index_features)

        env_config['index_features'] = dd_index_features_scaler.transform(
            index_features
        )
        conn.close()
        
class TrajectoryStorage():
    def __init__(self):
        self.episodes = []

    def set_env(self, env):
        self.env = env

    def split_trajectory(self, plan, reward):
        traj = []
        for i, (node, action) in enumerate(plan._joins[::-1]):
            plan.disjoin(node)
            obs = self.env.get_status(deepcopy(plan))
            traj.append(
                [obs, (action, self.env.get_mask(plan), i == 0, (i == 0)*reward)])
        return traj[::-1]

    def append(self, plan, final_reward):
        self.episodes.append(self.split_trajectory(
            deepcopy(plan), final_reward))

    def get_dataset(self, n=1000):
        """Get last n trajectories"""
        return self.episodes[-n:]


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

