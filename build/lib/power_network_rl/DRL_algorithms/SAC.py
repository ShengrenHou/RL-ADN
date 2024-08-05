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


## define net
class ActorSAC(nn.Module):
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__()
        self.enc_s = build_mlp(dims=[state_dim, *dims])  # encoder of state
        self.dec_a_avg = build_mlp(dims=[dims[-1], action_dim])  # decoder of action mean
        self.dec_a_std = build_mlp(dims=[dims[-1], action_dim])  # decoder of action log_std
        self.log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        self.soft_plus = nn.Softplus()

    def forward(self, state: Tensor) -> Tensor:
        state_tmp = self.enc_s(state)  # temporary tensor of state
        return self.dec_a_avg(state_tmp).tanh()  # action

    def get_action(self, state: Tensor) -> Tensor:  # for exploration
        state_tmp = self.enc_s(state)  # temporary tensor of state
        action_avg = self.dec_a_avg(state_tmp)
        action_std = self.dec_a_std(state_tmp).clamp(-20, 2).exp()

        noise = torch.randn_like(action_avg, requires_grad=True)
        action = action_avg + action_std * noise
        return action.clip(-1.0, 1.0)  # action (re-parameterize)

    def get_action_logprob(self, state: Tensor) -> [Tensor, Tensor]:
        state_tmp = self.enc_s(state)  # temporary tensor of state
        action_log_std = self.dec_a_std(state_tmp).clamp(-20, 2)
        action_std = action_log_std.exp()
        action_avg = self.dec_a_avg(state_tmp)

        '''add noise to a_noise in stochastic policy'''
        noise = torch.randn_like(action_avg, requires_grad=True)
        a_noise = action_avg + action_std * noise

        '''compute log_prob according to mean and std of a_noise (stochastic policy)'''
        # self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))
        log_prob = -action_log_std - noise.pow(2) * 0.5 - np.log(np.sqrt(2 * np.pi))

        '''fix logprob by adding the derivative of y=tanh(x)'''
        log_prob -= (np.log(2.) - a_noise - self.soft_plus(-2. * a_noise)) * 2.
        # logprob -= (1.000001 - action.tanh().pow(2)).log()
        return a_noise.tanh(), log_prob.sum(1, keepdim=True)
class CriticTwin(nn.Module):
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

class AgentSAC(AgentBase):
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.act_class = getattr(self, 'act_class', ActorSAC)  # get the attribute of object `self`
        self.cri_class = getattr(self, 'cri_class', CriticTwin)  # get the attribute of object `self`
        super().__init__(net_dims, state_dim, action_dim, gpu_id, args)
        self.cri_target = deepcopy(self.cri)

        self.alpha_log = torch.tensor((-1,), dtype=torch.float32, requires_grad=True, device=self.device)  # trainable
        self.alpha_optimizer = torch.optim.Adam((self.alpha_log,), lr=self.learning_rate)
        # here target entropy can be negaticve
        self.target_entropy = np.log(action_dim)

    def explore_one_env(self, env, horizon_len: int, if_random: bool = False) -> [Tensor]:
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

    def update_net(self, buffer: ReplayBuffer) -> Tuple[float, ...]:
        '''update network'''
        obj_critics = 0.0
        obj_actors = 0.0
        alphas = 0.0

        update_times = int(buffer.cur_size * self.repeat_times/self.batch_size)
        assert update_times >= 1
        for _ in range(update_times):
            '''objective of critic (loss function of critic)'''
            obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
            obj_critics += obj_critic.item()
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            '''objective of alpha (temperature parameter automatic adjustment)'''
            action_pg, log_prob = self.act.get_action_logprob(state)  # policy gradient
            obj_alpha = (self.alpha_log * (-log_prob + self.target_entropy).detach()).mean()
            self.optimizer_update(self.alpha_optimizer, obj_alpha)

            '''objective of actor'''
            alpha = self.alpha_log.exp().detach()
            alphas += alpha.item()
            # # keep the log into clip domains.
            # with torch.no_grad():
            #     self.alpha_log[:] = self.alpha_log.clamp(-16, 2)
            '''here is the objective of actor changes'''
            obj_actor = (self.cri(torch.cat((state, action_pg), dim=1)) - log_prob * alpha).mean()
            obj_actors += obj_actor.item()
            self.optimizer_update(self.act_optimizer, -obj_actor)
        return obj_critics / update_times, obj_actors / update_times, alphas / update_times


    def get_obj_critic_raw(self, buffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            states, actions, rewards, undones, next_ss = buffer.sample(batch_size)  # next_ss: next states
            # Check if rewards and undones have the same dimensions as states and actions
            if rewards.dim() != states.dim():
                rewards = rewards.unsqueeze(-1)
            if undones.dim() != states.dim():
                undones = undones.unsqueeze(-1)
            next_as, next_logprobs = self.act.get_action_logprob(next_ss)  # next actions
            '''here is how to calculate the next qs and q labels'''
            next_qs = torch.min(*self.cri_target.get_q1_q2(torch.cat((next_ss, next_as),dim=1)))
            alpha = self.alpha_log.exp()
            q_labels = rewards + undones * self.gamma * (next_qs - next_logprobs * alpha)

        q1, q2 = self.cri.get_q1_q2(torch.cat((states, actions),dim=1))

        obj_critic = self.criterion(q1, q_labels) + self.criterion(q2, q_labels)/2.  # twin critics
        return obj_critic, states

    def get_obj_critic_per(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            states, actions, rewards, undones, next_ss, is_weights, is_indices = buffer.sample_for_per(batch_size)
            # Check if rewards and undones have the same dimensions as states and actions
            if rewards.dim() != states.dim():
                rewards = rewards.unsqueeze(-1)
            if undones.dim() != states.dim():
                undones = undones.unsqueeze(-1)
            next_as, next_logprobs = self.act.get_action_logprob(next_ss)
            next_qs = torch.min(*self.cri_target.get_q1_q2(torch.cat((next_ss, next_as),dim=1)))
            alpha = self.alpha_log.exp()
            q_labels = rewards + undones * self.gamma * (next_qs - next_logprobs * alpha)

        q1, q2 = self.cri.get_q1_q2(torch.cat((states, actions),dim=1))

        td_errors = (self.criterion(q1, q_labels) + self.criterion(q2, q_labels))/2.
        obj_critic = (td_errors * is_weights).mean()

        buffer.td_error_update_for_per(is_indices.detach(), td_errors.detach())
        return obj_critic, states

