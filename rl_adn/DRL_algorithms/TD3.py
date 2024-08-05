import torch
import numpy as np
import torch.onnx
import torch.nn as nn
import copy as cp
from copy import deepcopy
import os
from torch import nn, Tensor
from typing import Tuple, Union
from utility import Config, ReplayBuffer, SumTree, build_mlp, get_episode_return, get_optim_param
from Agent import AgentBase


class CriticTwin(nn.Module):
    """
    Twin Critic network for algorithms like SAC and TD3.

    Attributes:
        enc_sa (nn.Module): Encoder network for state and action input.
        dec_q1 (nn.Module): Decoder network for the first Q-value output.
        dec_q2 (nn.Module): Decoder network for the second Q-value output.

    Methods:
        forward(value): Computes the Q-value for a given state-action pair.
        get_q1_q2(value): Computes both Q-values for a given state-action pair.
    """

    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__()
        self.enc_sa = build_mlp(dims=[state_dim + action_dim, *dims])  # encoder of state and action
        self.dec_q1 = build_mlp(dims=[dims[-1], 1])  # decoder of Q value 1
        self.dec_q2 = build_mlp(dims=[dims[-1], 1])  # decoder of Q value 2

    def forward(self, value: Tensor) -> Tensor:
        sa_tmp = self.enc_sa(value)
        return self.dec_q1(sa_tmp)  # Q value

    def get_q1_q2(self, value):
        sa_tmp = self.enc_sa(value)
        return self.dec_q1(sa_tmp), self.dec_q2(sa_tmp)  # two Q values
class Actor_TD3(nn.Module):
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__()
        self.net = build_mlp(dims=[state_dim, *dims, action_dim])
        self.explore_noise_std = None  # standard deviation of exploration action noise
    def forward(self, state: Tensor) -> Tensor:
        return self.net(state).tanh()  # action.tanh()
    def get_action(self, state: Tensor) -> Tensor:  # for exploration
        action = self.net(state).tanh()
        noise = (torch.randn_like(action) * self.explore_noise_std).clamp(-0.5, 0.5)
        return (action + noise).clamp(-1.0, 1.0)
    def get_action_noise(self, state: Tensor, action_std: float) -> Tensor:
        action = self.net(state).tanh()
        noise = (torch.randn_like(action) * action_std).clamp(-0.5, 0.5)
        return (action + noise).clamp(-1.0, 1.0)


class AgentTD3(AgentBase):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) agent implementation.

    Attributes:
        act_class (type): Class type for the actor network.
        cri_class (type): Class type for the critic network.
        act_target (nn.Module): Target actor network for stable training.
        cri_target (nn.Module): Target critic network for stable training.
        explore_noise_std (float): Standard deviation for exploration noise.
        policy_noise_std (float): Standard deviation for policy noise.
        update_freq (int): Frequency of policy updates.

    Methods:
        update_net(buffer): Updates the networks using the given replay buffer.
        get_obj_critic_raw(buffer, batch_size): Computes the raw objective for the critic.
        get_obj_critic_per(buffer, batch_size): Computes the PER-adjusted objective for the critic.
        explore_one_env(env, horizon_len, if_random): Explores an environment for a given horizon length.
    """

    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.act_class = getattr(self, 'act_class', Actor_TD3)
        self.cri_class = getattr(self, 'cri_class', CriticTwin)
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)
        self.act_target = deepcopy(self.act)
        self.cri_target = deepcopy(self.cri)

        self.explore_noise_std = getattr(args, 'explore_noise_std', 0.05)  # standard deviation of exploration noise
        self.policy_noise_std = getattr(args, 'policy_noise_std', 0.10)  # standard deviation of exploration noise
        self.update_freq = getattr(args, 'update_freq', 2)  # delay update frequency

        self.act.explore_noise_std = self.explore_noise_std  # assign explore_noise_std for agent.act.get_action(state)

    def update_net(self, buffer: ReplayBuffer) -> Tuple[float, ...]:
        """
        Updates the networks (actor and critic) using experiences from the replay buffer.

        This method performs the core updates for the TD3 algorithm, including updating the critic network and the actor network with a delayed policy update.

        Args:
            buffer (ReplayBuffer): The replay buffer containing experiences for training.

        Returns:
            Tuple[float, float]: A tuple containing the average objective values for the critic and actor updates.
        """

        obj_critics = 0.0
        obj_actors = 0.0
        update_times = int(buffer.cur_size * self.repeat_times/self.batch_size)
        assert update_times >= 1
        for update_c in range(update_times):
            obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
            obj_critics += obj_critic.item()
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            if update_c % self.update_freq == 0:  # delay update
                action_pg = self.act(state)  # policy gradient
                obj_actor = self.cri_target(torch.cat((state, action_pg), dim=1)).mean()  # use cri_target is more stable than cri
                obj_actors += obj_actor.item()
                self.optimizer_update(self.act_optimizer, -obj_actor)
                self.soft_update(self.act_target, self.act, self.soft_update_tau)
        return obj_critics / update_times, obj_actors / update_times

    def get_obj_critic_raw(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            states, actions, rewards, undones, next_ss = buffer.sample(batch_size)  # next_ss: next states
            if rewards.dim() != states.dim():
                rewards = rewards.unsqueeze(-1)
            if undones.dim() != states.dim():
                undones = undones.unsqueeze(-1)
            next_as = self.act_target.get_action_noise(next_ss, self.policy_noise_std)  # next actions
            next_qs = torch.min(*self.cri_target.get_q1_q2(torch.cat((next_ss, next_as),dim=1)))

            q_labels = rewards + undones * self.gamma * next_qs

        q1, q2 = self.cri.get_q1_q2(torch.cat((states, actions),dim=1))
        obj_critic = self.criterion(q1, q_labels) + self.criterion(q2, q_labels)  # twin critics
        return obj_critic, states

    def get_obj_critic_per(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            states, actions, rewards, undones, next_ss, is_weights, is_indices = buffer.sample_for_per(batch_size)
            if rewards.dim() != states.dim():
                rewards = rewards.unsqueeze(-1)
            if undones.dim() != states.dim():
                undones = undones.unsqueeze(-1)
            next_as = self.act_target.get_action_noise(next_ss, self.policy_noise_std)
            next_qs = torch.min(*self.cri_target.get_q1_q2(torch.cat((next_ss, next_as),dim=1)))
            q_labels = rewards + undones * self.gamma * next_qs

        q1, q2 = self.cri.get_q1_q2(torch.cat((states, actions),dim=1))
        td_errors = self.criterion(q1, q_labels) + self.criterion(q2, q_labels)
        obj_critic = (td_errors * is_weights).mean()

        buffer.td_error_update_for_per(is_indices.detach(), td_errors.detach())
        return obj_critic, states
    def explore_one_env(self, env, horizon_len: int, if_random: bool = False) -> [Tensor]:
        """
        Explores a given environment for a specified horizon length.

        This method is used for collecting experiences by interacting with the environment. It can operate in either a random action mode or a policy-based action mode, with an additional noise for exploration in TD3.

        Args:
            env: The environment to be explored.
            horizon_len (int): The number of steps to explore the environment.
            if_random (bool): If True, actions are chosen randomly. If False, actions are chosen based on the current policy with added noise for exploration.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: A tuple containing states, actions, rewards, and undones (indicating whether the episode has ended) collected during the exploration.
        """

        states = torch.zeros((horizon_len, self.num_envs, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.num_envs, self.action_dim), dtype=torch.float32).to(self.device)
        rewards = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)
        dones = torch.zeros((horizon_len, self.num_envs), dtype=torch.bool).to(self.device)

        ary_state =env.reset()
        get_action = self.act.get_action
        for i in range(horizon_len):
            state = torch.as_tensor(ary_state, dtype=torch.float32, device=self.device)
            action = torch.rand(self.action_dim) * 2 - 1.0 if if_random else get_action(state.unsqueeze(0))[0]

            states[i] = state
            actions[i] = action

            ary_action = action.detach().cpu().numpy()
            next_state, reward, done,_ = env.step(ary_action)
            ary_state = env.reset() if done else next_state

            rewards[i] = reward
            dones[i] = done

        # rewards = rewards.unsqueeze(1)
        undones = 1.0 - dones.type(torch.float32)
        # undones = (1.0 - dones.type(torch.float32)).unsqueeze(1)
        return states, actions, rewards, undones


