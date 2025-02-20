import gym
import numpy as np
import psycopg2
from gym.spaces import Discrete
import torch

from plan import SinglePlan
from plan import GlobalPlan
import time
import sys
sys.setrecursionlimit(10000)
class Env():
    def __init__(self, env_config):
        self.scheme = env_config['scheme']
        self.db_data = env_config['db_data']
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
        self.plan_idx = 0
        self.plan = self.plans[self.plan_idx]
        self.global_plan = GlobalPlan()

    def get_obs(self):
        """
        Constructs an observation based on the plan.
        """
        raise NotImplementedError

    def from_action(self, action):
        """
        Retrieve relations to join from action.
        """
        r1, r2 = self.actions[action]
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

    @property
    def is_done(self):
        return self.plan.is_complete

    def step(self, action):
        self.current_step += 1
        tables_to_join = self.from_action(action)
        self.plan.join(*tables_to_join)
        return self.get_obs(), self.reward(), self.is_done, {}

    def reset(self, idxs=None):
        self.plans.clear()
        for idx in idxs:
            self.plans.append(SinglePlan(*self.db_data[self.query_id]))
        self.plan_idx = 0
        self.current_step = 0
    
    def done(self):
        return self.plan_idx == len(self.plans)
    
    def get_mask(self):
        def find_inner_join_actions(p):
            actions = []
            roots = p.get_roots()
            for i, n1 in enumerate(roots):
                for j, n2 in enumerate(roots):
                    if i != j and p.is_inner_join(n1, n2):
                        actions.append((i, j))
            return actions
        size = len(self.plan.get_roots())
        m = torch.zeros((size, size), dtype=bool)
        m[list(zip(*find_inner_join_actions(self.plan)))] = 1
        return m
    
    def reward(self):
        if not self.is_done():
            return -0.05
        

    def render(self):
        return self.plan.render()

    def find_cost(self, p):
        return list(p.G.nodes(data=True))[-1][-1]['cost'][0]

