import gym
import numpy as np
import psycopg2
from gym.spaces import Discrete

from plan import SinglePlan
import time
import sys
sys.setrecursionlimit(10000)
class DataBaseEnv():
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
        self.plan_idx = 0
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

    def reset(self, idx=None):
        self.batch_idx = 0
        
        if isinstance(idx, str):
            self.query_id = idx
        elif isinstance(idx, int):
            self.query_id = list(self.db_data.keys())[idx]
        else:
            self.query_id = np.random.choice(list(self.db_data.keys()))
        self.plan = SinglePlan(*self.db_data[self.query_id])
        self.current_step = 0
    
    def done(self):
        
    def get_mask(self):

    def render(self):
        return self.plan.render()

    def find_cost(self, p):
        return list(p.G.nodes(data=True))[-1][-1]['cost'][0]