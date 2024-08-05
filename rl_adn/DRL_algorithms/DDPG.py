import torch
import torch.onnx
import copy as cp
from copy import deepcopy
import os
from torch import nn, Tensor
from typing import Tuple, Union
from utility import Config, ReplayBuffer, SumTree, build_mlp, get_episode_return, get_optim_param
from Agent import AgentBase
from rl_adn.environments.env import PowerNetEnv, env_config
import time
class Actor(nn.Module):
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        """
        Initializes the Actor network for the DDPG algorithm.

        Args:
            dims ([int]): List of integers defining the dimensions of the hidden layers.
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.

        Attributes:
            net: Neural network created using the specified dimensions.
            explore_noise_std: Standard deviation of exploration action noise, initialized as None.
        """
        super().__init__()
        self.net = build_mlp(dims=[state_dim, *dims, action_dim])
        self.explore_noise_std = None  # standard deviation of exploration action noise
    def forward(self, state: Tensor) -> Tensor:
        """
        Defines the forward pass of the Actor network.

        Args:
            state (Tensor): The input state tensor.

        Returns:
            Tensor: The output action tensor after applying the tanh activation function.
        """
        return self.net(state).tanh()  # action.tanh()
    def get_action(self, state: Tensor) -> Tensor:
        """
        Computes the action for a given state with added exploration noise.

        Args:
            state (Tensor): The input state tensor.

        Returns:
            Tensor: The action tensor with added exploration noise.
        """
        action = self.net(state).tanh()
        noise = (torch.randn_like(action) * self.explore_noise_std).clamp(-0.5, 0.5)
        return (action + noise).clamp(-1.0, 1.0)
    def get_action_noise(self, state: Tensor, action_std: float) -> Tensor:
        """
        Computes the action for a given state with specified exploration noise.

        Args:
            state (Tensor): The input state tensor.
            action_std (float): Standard deviation for the exploration noise.

        Returns:
            Tensor: The action tensor with specified exploration noise.
        """
        action = self.net(state).tanh()
        noise = (torch.randn_like(action) * action_std).clamp(-0.5, 0.5)
        return (action + noise).clamp(-1.0, 1.0)
class Critic(nn.Module):
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        """
        Initializes the Critic network for the DDPG algorithm.

        Args:
            dims ([int]): List of integers defining the dimensions of the hidden layers.
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.

        Attributes:
            net: Neural network created using the specified dimensions.
        """
        super().__init__()
        self.net = build_mlp(dims=[state_dim + action_dim, *dims, 1])
    def forward(self, value: Tensor) -> Tensor:
        """
        Defines the forward pass of the Critic network.

        Args:
            value (Tensor): The input tensor combining state and action.

        Returns:
            Tensor: The output Q-value tensor.
        """
        return self.net(value)  # Q value
class AgentDDPG(AgentBase):
    """
    Implements the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm.

    This class is responsible for the overall management of the actor and critic networks, including their initialization, updates, and interactions with the environment.

    Attributes:
        act_class: Actor class for creating the actor network.
        cri_class: Critic class for creating the critic network.
        act_target: Target actor network for stable training.
        cri_target: Target critic network for stable training.
        explore_noise_std: Standard deviation of exploration noise.
    """

    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        """
        Initializes the AgentDDPG with the specified network dimensions, state and action dimensions, and other configurations.

        Args:
            net_dims ([int]): List of integers defining the dimensions of the hidden layers for the actor and critic networks.
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            gpu_id (int): GPU ID for running the networks. Defaults to 0.
            args (Config): Configuration object with additional settings.
        """
        self.act_class = getattr(self, 'act_class', Actor)
        self.cri_class = getattr(self, 'cri_class', Critic)
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)
        self.act_target = deepcopy(self.act)
        self.cri_target = deepcopy(self.cri)
        '''comapre to TD3, there is no policy noise'''
        self.explore_noise_std = getattr(args, 'explore_noise_std', 0.05)  # standard deviation of exploration noise
        self.act.explore_noise_std = self.explore_noise_std  # assign explore_noise_std for agent.act.get_action(state)

    def update_net(self, buffer: ReplayBuffer) -> Tuple[float, ...]:
        """
        Updates the networks (actor and critic) using the given replay buffer.

        Args:
            buffer (ReplayBuffer): The replay buffer containing experience tuples.

        Returns:
            Tuple[float, ...]: A tuple containing the average objective values for the critic and actor updates.
        """
        obj_critics = 0.0
        obj_actors = 0.0
        # update_times = int(buffer.add_size * self.repeat_times)
        update_times = int(buffer.cur_size * self.repeat_times/self.batch_size)
        assert update_times >= 1
        for update_c in range(update_times):
            obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
            obj_critics += obj_critic.item()
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
            '''compare with TD3, DDPG no policy delay update'''
            action_pg = self.act(state)  # policy gradient
            obj_actor = self.cri_target(torch.cat((state, action_pg), dim=1)).mean()  # use cri_target is more stable than cri
            obj_actors += obj_actor.item()
            self.optimizer_update(self.act_optimizer, -obj_actor)
            self.soft_update(self.act_target, self.act, self.soft_update_tau)
        return obj_critics / update_times, obj_actors / update_times

    def get_obj_critic_raw(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        """
        Computes the objective for the critic network using raw experiences from the buffer.

        Args:
            buffer (ReplayBuffer): The replay buffer containing experience tuples.
            batch_size (int): The size of the batch to sample from the buffer.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the critic objective and the sampled states.
        """

        with torch.no_grad():
            states, actions, rewards, undones, next_ss = buffer.sample(batch_size)  # next_ss: next states
            # Check if rewards and undones have the same dimensions as states and actions
            if rewards.dim() != states.dim():
                rewards = rewards.unsqueeze(-1)
            if undones.dim() != states.dim():
                undones = undones.unsqueeze(-1)
            '''compare with TD3 no policy noise'''
            next_as = self.act_target(next_ss) # next actions
            next_qs = self.cri_target(torch.cat((next_ss, next_as),dim=1))  # next q values
            q_labels = rewards + undones * self.gamma * next_qs

        q_values = self.cri(torch.cat((states, actions),dim=1))
        obj_critic = self.criterion(q_values,q_labels)
        return obj_critic, states

    def get_obj_critic_per(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        """
        Computes the objective for the critic network using prioritized experiences from the buffer.

        Args:
            buffer (ReplayBuffer): The replay buffer containing experience tuples.
            batch_size (int): The size of the batch to sample from the buffer.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the critic objective and the sampled states.
        """
        with torch.no_grad():
            states, actions, rewards, undones, next_ss, is_weights, is_indices = buffer.sample_for_per(batch_size)
            # is_weights, is_indices: important sampling `weights, indices` by Prioritized Experience Replay (PER)
            # Check if rewards and undones have the same dimensions as states and actions
            if rewards.dim() != states.dim():
                rewards = rewards.unsqueeze(-1)
            if undones.dim() != states.dim():
                undones = undones.unsqueeze(-1)
            next_as = self.act_target(next_ss)

            next_qs = self.cri_target(torch.cat((next_ss, next_as),dim=1))  # next q values

            q_labels = rewards + undones * self.gamma * next_qs

        q_values = self.cri(torch.cat((states, actions),dim=1))
        td_errors = self.criterion(q_values, q_labels)
        obj_critic = (td_errors * is_weights).mean()

        buffer.td_error_update_for_per(is_indices.detach(), td_errors.detach())
        return obj_critic, states
    def explore_one_env(self, env, horizon_len: int, if_random: bool = False) -> [Tensor]:
        """
        Explores the environment for a given number of steps.

        Args:
            env: The environment to be explored.
            horizon_len (int): The number of steps to explore.
            if_random (bool): Flag to determine if actions should be random. Defaults to False.

        Returns:
            [Tensor]: A list of tensors containing states, actions, rewards, and undones (not done flags).
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
