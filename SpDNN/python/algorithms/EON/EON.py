import argparse
import json
import numpy as np
import networkx as nx
import time
import torch
from torch import nn
from torch.optim import Adam

from . import network
from . import environment

# Adapted from: https://github.com/ericyangyu/PPO-for-Beginners

class PPO:
    def __init__(self, graph, training_log_path="training.log"):
        self._init_hyperparameters(graph)
        self.env = environment.SparseNetowrkEnv(graph, max_frontier_len=100)
        self.actor_critic = network.EdgeOrderingNetwork(graph)
        self.actor_critic_optim = Adam(self.actor_critic.parameters(), lr=self.lr)

        self.best_order = None
        self.best_reward = -float('inf')

        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,          # timesteps so far
            'i_so_far': 0,          # iterations so far
            'batch_lens': [],       # episodic lengths in batch
            'batch_rews': [],       # episodic returns in batch
            'actor_losses': [],     # losses of actor network in current iteration
        }

        self.training_log_path = training_log_path
        self.alpha = 0.1

    def _init_hyperparameters(self, graph):
        self.max_timesteps_per_episode = graph.number_of_edges()
        self.timesteps_per_batch = self.max_timesteps_per_episode * 8
        self.n_updates_per_iteration = 25
        self.gamma = 0.95
        self.clip = 0.2
        self.lr = 0.0005
        self.alpha_decay = 0.002

        self.save_freq = 10

    def compute_rtgs(self, batch_rews):
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []
        # Iterate through each episode backwards to maintain same order
        # in batch_rtgs
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0 # The discounted reward so far
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs

    def get_action(self, cache, frontier):
        dist, _ = self.actor_critic(frontier, cache)
        dist = dist.squeeze(0)

        idx = torch.argmin(frontier[:, 0])
        if frontier[idx, 0] >= 0:
            idx = frontier.shape(0)
        
        if np.random.rand() < self.alpha:
            index = np.random.randint(0, idx)
        else:
            index = torch.argmax(dist[:idx])

        action = frontier[index].detach().tolist()
        return action, index, torch.log(dist[index])

    def evaluate(self, batch_obs, batch_acts):
        cache = torch.LongTensor([o["cache"] for o in batch_obs])
        frontier = torch.LongTensor([o["frontier"] for o in batch_obs])
        dist, V = self.actor_critic(frontier, cache)
        probs = torch.take_along_dim(dist, batch_acts.unsqueeze(0), 1).squeeze()

        return V, torch.log(probs)

    def rollout(self):
        # Batch data
        batch_obs = []             # batch observations
        batch_acts = []            # batch actions
        batch_log_probs = []       # log probs of each action
        batch_rews = []            # batch rewards
        batch_rtgs = []            # batch rewards-to-go
        batch_lens = []            # episodic lengths in batch

        # Number of timesteps run so far this batch
        t = 0 
        while t < self.timesteps_per_batch:
            # Rewards this episode
            ep_rews = []
            obs = self.env.reset()
            done = False
            order = []
            for ep_t in range(self.max_timesteps_per_episode):
                # Increment timesteps ran this batch so far
                t += 1
                # Collect observation
                batch_obs.append(obs)
                cache = torch.LongTensor(obs["cache"])
                frontier = torch.LongTensor(obs["frontier"])
                action, action_idx, log_prob = self.get_action(cache, frontier)
                order.append(tuple(action))
                obs, rew, done = self.env.step(action)
            
                # Collect reward, action, and log prob
                ep_rews.append(rew)
                batch_acts.append(action_idx)
                batch_log_probs.append(log_prob)

                if done:
                    break
            
            # Collect episodic length and rewards
            batch_lens.append(ep_t + 1) # plus 1 because timestep starts at 0
            batch_rews.append(ep_rews)

            if np.sum(ep_rews) > self.best_reward:
                self.best_order = order
                self.best_reward = sum(ep_rews)   
        
        batch_acts = torch.LongTensor(batch_acts)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs = self.compute_rtgs(batch_rews)

        # Log the episodic returns and episodic lengths in this batch.
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def learn(self, total_timesteps):
        i_so_far = 0
        t_so_far = 0
        while t_so_far < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            t_so_far += np.sum(batch_lens)
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # Calculate V_{phi, k}
            V, _ = self.evaluate(batch_obs, batch_acts)

            # Calculate advantage
            A_k = batch_rtgs - V.detach()
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            old_probs = batch_log_probs

            for _ in range(self.n_updates_per_iteration):
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                
                ratios = torch.exp(curr_log_probs - batch_log_probs)
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V.squeeze(), batch_rtgs)
                loss = actor_loss  + 0.5 * critic_loss

                self.actor_critic_optim.zero_grad()
                loss.backward()
                self.actor_critic_optim.step()

                self.logger['actor_losses'].append(actor_loss.detach())
                old_probs = curr_log_probs

            self._log_summary()

            self.alpha = max(0, self.alpha - self.alpha_decay)

            if i_so_far % self.save_freq == 0:
                torch.save(self.actor_critic.state_dict(), "./ppo_actor_critic.pth")

    def _log_summary(self):
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        actor_losses = [losses.float().mean().item() for losses in self.logger['actor_losses']]
        avg_actor_loss = np.mean(actor_losses)

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []

        entry = {
            "iteration": i_so_far,
            "avg_ep_rews": avg_ep_rews,
            "average_eps_loss": avg_actor_loss,
            "losses": actor_losses,
            "best_reward": self.best_reward
        }

        with open(self.training_log_path, "a") as fp:
            fp.write(json.dumps(entry) + "\n")

    def generate_order(self):
        return self.best_order

def reorder_edges(graph: nx.DiGraph, n:int = 5000, training_log_path:str = "training.log") -> list[tuple]:
    model = PPO(graph, training_log_path=training_log_path)
    model.learn(n)
    order = model.generate_order()
    order = [(u, v, graph.edges[u,v][2]) for (u,v) in order]    
    return order
