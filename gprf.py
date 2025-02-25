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

import logging
LOG = logging.getLogger(__name__)

INF = 1e9

class Agent(nn.Module):
    def __init__(self, net, eps=1e-2, device='cpu'):
        super().__init__()
        self.net = net.to(device=device)
        self.device = device
        self.eps = eps
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

        logit = self.predict_net(self.net, inputs)
        dims = logit.shape
        masked_logit = torch.where(mask.view(dims).to(logit.device),
                                   logit, torch.tensor(float("-inf")).to(logit.device))
        probs = masked_logit.view(dims[0], -1).softmax(1).cpu().numpy()
        actions = [np.random.choice(len(prob), p=prob) for prob in probs]
        actions = list(zip(*np.unravel_index(actions, dims[1:])))
        return actions

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

    def predict_net(self, net, x_batch):
        with evaluation_mode(net):
            x_batch = to_device(x_batch, self.device)
            out = net(x_batch)
        return out


class GPRF():
    def __init__(self, baseline, batches):
        self.baseline = baseline
        self.batches = batches
        self.d = config.d
        self.env_config = config.env_config
        self.env = Env()
        self.agent = Agent(self.env.net)
    def run(self):
        for ep in range(self.d['train_args']['episodes']):
            total_reward = 0
            state = self.env.reset()
            while not self.env.done():
                mask = self.env.get_mask()
                action = self.agent.predict(state, mask)
                next_state, reward, done = self.env.step(action)  
                self.agent.store_transition(state, action, reward, next_state, done, next_mask)
                self.agent.train()
                
                state = next_state
                total_reward += reward
            

    def logger(self):
        writer = SummaryWriter(self.logdir)
        while self.episode.value < self.total_episodes * self.n_queries:
            r = self.log_q.get()
            if len(r) == 2:
                losses, ep = r
                pg_loss, value_loss, entropy_loss = losses
                writer.add_scalar('Loss/policy_loss', pg_loss, ep)
                writer.add_scalar('Loss/value_loss', value_loss, ep)
                writer.add_scalar('Loss/entropy_loss', entropy_loss, ep)
                torch.save(self.agent.net.state_dict(),
                           Path(self.logdir) / 'state_dict.pt')
            else:
                (n_plans, n_subplans), best_found_costs, generated_costs, baseline_costs, reward, step, episode = r
                writer.add_scalar(
                    'Experience size/complete unique plans', n_plans, episode)
                writer.add_scalar(
                    'Rewards', reward, step)

                for stat_type, costs in (('best_found', best_found_costs), ('generated', generated_costs)):
                    if costs.keys() >= self.env_config['db_data'].keys():
                        writer.add_scalar(f"Cost/{stat_type}/avg_cost:episode",
                                          np.mean(list(costs.values())), episode)
                        if baseline_costs.keys() >= costs.keys():
                            average_ratio = np.mean(
                                [costs[q]/baseline_costs[q] for q in costs])
                            writer.add_scalar(
                                f'Cost/{stat_type}/avg_baseline_ratio:episode', average_ratio, episode)
                            writer.add_scalar(
                                f'Cost/{stat_type}/avg_baseline_ratio:experience', average_ratio, n_plans)

    def runner_process(self):
        env = self.env_plan(self.env_config)
        is_done = True
        while True:
            if is_done:
                if self.episode.value >= self.total_episodes * self.n_queries:
                    return
                with self.episode.get_lock():
                    if self.episode.value % self.n_update == 0 and self.sync:
                        self.step_flag.clear()
                    query_num = self.episode.value % self.n_queries
                    if query_num == 0:
                        np.random.shuffle(self.query_ids)
                    query_idx = self.query_ids[query_num]
                    env.reset(query_idx)
                    self.episode.value += 1
                self.step_flag.wait()
            obs = [env.get_state()]
            mask = get_mask(env.plan)  # [N, max(ni), max(ni)]
            actions, _ = self.agent.predict(obs, mask)
            _, cost, is_done, _ = env.step(actions[0])
            with self.step.get_lock():
                self.step.value += 1
            if is_done:
                self.update_q.put(([env.plan], [cost], env.query_id))

    def update_process(self):
        env = self.env_plan(self.env_config)
        self.traj_storage.set_env(env)
        BASELINE_REWARD = 0.5
        for p in self.baseline_plans.values():
            self.traj_storage.append(p, BASELINE_REWARD)
        generated_costs = {}
        baseline_costs = {q: self.experience.get_cost(
            p, q) for q, p in self.baseline_plans.items()}
        episode = 0
        while True:
            if episode % self.n_update == 0:
                LOG.info(
                    f"Update started, step: {self.step.value}, episode: {episode}, time: {time.ctime()}")
                if episode == 0:
                    data = self.traj_storage.get_dataset(n=self.n_queries)
                else:
                    data = self.traj_storage.get_dataset(
                        n=self.n_train_episodes)
                train_data = data
                val_data = None
                # val_split = max(1, min(self.val_size, int(0.3*len(data))))
                # train_data, val_data = data[:-val_split], data[-val_split:]
                losses = self.agent.train_net(
                    train_data=train_data, val_data=val_data, val_steps=1, criterion=ac_loss, gamma=self.gamma, **self.train_args)
                LOG.info(
                    f"Update ended, step: {self.step.value}, episode: {episode}, time: {time.ctime()}")
                # allow exploring
                self.step_flag.set()
                self.log_q.put((losses, self.step.value))
                # save found plans
                path = Path(self.logdir) / 'plans'
                path.mkdir(parents=True, exist_ok=True)
                best_plans = self.experience.plans_for_queries()
                for q, p in best_plans.items():
                    p.save(path / f"{q}.json")
                LOG.info(
                    f"Best plans after {episode} episodes saved to {str(path)}")

            if (episode >= self.total_episodes * self.n_queries):
                return

            complete_plans, costs, query_id = self.update_q.get()

            for plan, cost in zip(complete_plans, costs):
                # reward = (
                #     baseline_costs[query_id] - cost)/baseline_costs[query_id]
                reward = - np.log(cost/baseline_costs[query_id])
                LOG.debug(
                    f"Completed plan for {query_id} query with cost = {cost}, reward = {reward}")
                self.experience.append(plan, cost, query_id)
                self.traj_storage.append(plan, reward)

            # update values for log
            average_generated_cost = self.experience.get_cost(
                complete_plans[0], query_id)
            if average_generated_cost is not None:
                generated_costs[query_id] = average_generated_cost
            baseline_costs[query_id] = self.experience.get_cost(
                self.baseline_plans[query_id], query_id)
            best_found_costs = self.experience.costs_for_queries()

            self.log_q.put((self.experience.size(), best_found_costs,
                            generated_costs, baseline_costs, reward, self.step.value, episode))

            if self.save_explored_plans and episode % (5 * self.n_queries) == 0:
                for i, (p, q) in enumerate(self.experience.complete_plans.keys()):
                    save_path = Path(self.logdir) / 'all_plans' / str(q)
                    save_path.mkdir(parents=True, exist_ok=True)
                    p.save(save_path / f"{i}.json")

            episode += 1
            
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
