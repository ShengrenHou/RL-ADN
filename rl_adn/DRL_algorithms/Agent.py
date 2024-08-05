from torch import nn, Tensor
from typing import Tuple, Union
import torch
import os
from copy import deepcopy
import numpy as np
from rl_adn.DRL_algorithms.utility import Config, ReplayBuffer, build_mlp, \
    get_optim_param
from torch.nn.utils import clip_grad_norm_
class AgentBase:
    """
        Base Agent class for handling basic agent functions.

        Args:
            net_dims (list): Network dimensions.
            state_dim (int): Dimensionality of the state space.
            action_dim (int): Dimensionality of the action space.
            gpu_id (int): GPU ID. Default is 0.
            args (Config): Configuration arguments. Default is `Config()`.

        Attributes:
            gamma (float): Discount factor of future rewards.
            num_envs (int): Number of sub-environments in a vectorized environment.
            batch_size (int): Number of transitions sampled from replay buffer.
            ...
            save_attr_names (set): Attributes to be saved or loaded.

        Example:
            >>> agent = AgentBase(net_dims=[64, 64], state_dim=10, action_dim=2)
        """
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.gamma = args.gamma  # discount factor of future rewards
        self.num_envs = args.num_envs  # the number of sub envs in vectorized env. `num_envs=1` in single env.
        self.batch_size = args.batch_size  # num of transitions sampled from replay buffer.
        self.repeat_times = args.repeat_times  # repeatedly update network using ReplayBuffer
        self.reward_scale = args.reward_scale  # an approximate target reward usually be closed to 256
        self.learning_rate = args.learning_rate  # the learning rate for network updating
        self.if_off_policy = args.if_off_policy  # whether off-policy or on-policy of DRL algorithm
        self.clip_grad_norm = args.clip_grad_norm  # clip the gradient after normalization
        self.soft_update_tau = args.soft_update_tau  # the tau of soft target update `net = (1-tau)*net + net1`
        self.state_value_tau = args.state_value_tau  # the tau of normalize for value and state

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.last_state = None  # last state of the trajectory for training. last_state.shape == (num_envs, state_dim)
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        '''network'''
        act_class = getattr(self, "act_class", None)
        cri_class = getattr(self, "cri_class", None)
        self.act = self.act_target = act_class(net_dims, state_dim, action_dim).to(self.device)
        self.cri = self.cri_target = cri_class(net_dims, state_dim, action_dim).to(self.device) \
            if cri_class else self.act

        '''optimizer'''
        self.act_optimizer = torch.optim.AdamW(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = torch.optim.AdamW(self.cri.parameters(), self.learning_rate) \
            if cri_class else self.act_optimizer
        from types import MethodType  # built-in package of Python3
        self.act_optimizer.parameters = MethodType(get_optim_param, self.act_optimizer)
        self.cri_optimizer.parameters = MethodType(get_optim_param, self.cri_optimizer)

        """attribute"""
        if self.num_envs == 1:
            self.explore_env = self.explore_one_env
        else:
            self.explore_env = self.explore_vec_env

        self.if_use_per = getattr(args, 'if_use_per', None)  # use PER (Prioritized Experience Replay)
        if self.if_use_per:
            self.criterion = torch.nn.SmoothL1Loss(reduction="none")
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.SmoothL1Loss(reduction="mean")
            self.get_obj_critic = self.get_obj_critic_raw

        """save and load"""
        self.save_attr_names = {'act', 'act_target', 'act_optimizer', 'cri', 'cri_target', 'cri_optimizer'}

    def explore_one_env(self, env, horizon_len: int, if_random: bool = False) -> Tuple[Tensor, ...]:
        """
        Collect trajectories through actor-environment interaction for a single environment.

        Args:
            env: The Reinforcement Learning environment.
            horizon_len (int): Number of steps for exploration.
            if_random (bool, optional): Whether to use random actions for exploration. Defaults to False.

        Returns:
            Tuple[Tensor, ...]: A tuple containing states, actions, rewards, and undones.
        """

        states = torch.zeros((horizon_len, self.num_envs, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.num_envs, self.action_dim), dtype=torch.float32).to(self.device)
        rewards = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)
        dones = torch.zeros((horizon_len, self.num_envs), dtype=torch.bool).to(self.device)

        state = self.last_state  # state.shape == (1, state_dim) for a single env.

        get_action = self.act.get_action
        for t in range(horizon_len):
            action = torch.rand(1, self.action_dim) * 2 - 1.0 if if_random else get_action(state)
            states[t] = state

            ary_action = action[0].detach().cpu().numpy()
            ary_state, reward, done, _ = env.step(ary_action)  # next_state
            ary_state = env.reset() if done else ary_state  # ary_state.shape == (state_dim, )
            state = torch.as_tensor(ary_state, dtype=torch.float32, device=self.device).unsqueeze(0)

            actions[t] = action
            rewards[t] = reward
            dones[t] = done

        self.last_state = state  # state.shape == (1, state_dim) for a single env.

        rewards *= self.reward_scale
        undones = 1.0 - dones.type(torch.float32)
        return states, actions, rewards, undones

    def explore_vec_env(self, env, horizon_len: int, if_random: bool = False) -> Tuple[Tensor, ...]:
        """
        Collect trajectories through actor-environment interaction for a vectorized environment.

        Args:
            env: The Reinforcement Learning environment supporting vectorized operations.
            horizon_len (int): Number of steps for exploration.
            if_random (bool, optional): Whether to use random actions for exploration. Defaults to False.

        Returns:
            Tuple[Tensor, ...]: A tuple containing states, actions, rewards, and undones.
        """

        states = torch.zeros((horizon_len, self.num_envs, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.num_envs, self.action_dim), dtype=torch.float32).to(self.device)
        rewards = torch.zeros((horizon_len, self.num_envs), dtype=torch.float32).to(self.device)
        dones = torch.zeros((horizon_len, self.num_envs), dtype=torch.bool).to(self.device)

        state = self.last_state  # last_state.shape == (num_envs, state_dim)
        get_action = self.act.get_action
        for t in range(horizon_len):
            action = torch.rand(self.num_envs, self.action_dim) * 2 - 1.0 if if_random \
                else get_action(state).detach()
            states[t] = state  # state.shape == (num_envs, state_dim)

            state, reward, done, _ = env.step(action)  # next_state
            actions[t] = action
            rewards[t] = reward
            dones[t] = done

        self.last_state = state

        rewards *= self.reward_scale
        undones = 1.0 - dones.type(torch.float32)
        return states, actions, rewards, undones

    def update_net(self, buffer: Union[ReplayBuffer, tuple]) -> Tuple[float, ...]:
        """
        Update the neural network models based on the data in the replay buffer.

        Args:
            buffer (Union[ReplayBuffer, tuple]): The replay buffer or a tuple of experiences.

        Returns:
            Tuple[float, ...]: Objectives of the critic and actor updates.
        """

        obj_critic = 0.0  # criterion(q_value, q_label).mean().item()
        obj_actor = 0.0  # q_value.mean().item()
        assert isinstance(buffer, ReplayBuffer) or isinstance(buffer, tuple)
        assert isinstance(self.batch_size, int)
        assert isinstance(self.repeat_times, int)
        assert isinstance(self.reward_scale, float)
        return obj_critic, obj_actor

    def get_obj_critic_raw(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            states, actions, rewards, undones, next_ss = buffer.sample(batch_size)  # next_ss: next states
            next_as = self.act_target(next_ss)  # next actions
            next_qs = self.cri_target(next_ss, next_as)  # next q values
            q_labels = rewards + undones * self.gamma * next_qs

        q_values = self.cri(states, actions)
        obj_critic = self.criterion(q_values, q_labels)
        return obj_critic, states

    def get_obj_critic_per(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            states, actions, rewards, undones, next_ss, is_weights, is_indices = buffer.sample_for_per(batch_size)
            # is_weights, is_indices: important sampling `weights, indices` by Prioritized Experience Replay (PER)

            next_as = self.act_target(next_ss)
            next_qs = self.cri_target(next_ss, next_as)
            q_labels = rewards + undones * self.gamma * next_qs

        q_values = self.cri(states, actions)
        td_errors = self.criterion(q_values, q_labels)
        obj_critic = (td_errors * is_weights).mean()

        buffer.td_error_update_for_per(is_indices.detach(), td_errors.detach())
        return obj_critic, states

    def get_cumulative_rewards(self, rewards: Tensor, undones: Tensor) -> Tensor:
        returns = torch.empty_like(rewards)

        masks = undones * self.gamma
        horizon_len = rewards.shape[0]

        last_state = self.last_state
        next_action = self.act_target(last_state)
        next_value = self.cri_target(last_state, next_action).detach()
        for t in range(horizon_len - 1, -1, -1):
            returns[t] = next_value = rewards[t] + masks[t] * next_value
        return returns

    def optimizer_update(self, optimizer: torch.optim, objective: Tensor):
        """minimize the optimization objective via update the network parameters

        optimizer: `optimizer = torch.optim.SGD(net.parameters(), learning_rate)`
        objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
        """
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
        optimizer.step()

    def optimizer_update_amp(self, optimizer: torch.optim, objective: Tensor):  # automatic mixed precision
        """minimize the optimization objective via update the network parameters

        amp: Automatic Mixed Precision

        optimizer: `optimizer = torch.optim.SGD(net.parameters(), learning_rate)`
        objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
        """
        amp_scale = torch.cuda.amp.GradScaler()  # write in __init__()

        optimizer.zero_grad()
        amp_scale.scale(objective).backward()  # loss.backward()
        amp_scale.unscale_(optimizer)  # amp

        # from torch.nn.utils import clip_grad_norm_
        clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
        amp_scale.step(optimizer)  # optimizer.step()
        amp_scale.update()  # optimizer.step()

    def update_avg_std_for_normalization(self, states: Tensor, returns: Tensor):
        tau = self.state_value_tau
        if tau == 0:
            return

        state_avg = states.mean(dim=0, keepdim=True)
        state_std = states.std(dim=0, keepdim=True)
        self.act.state_avg[:] = self.act.state_avg * (1 - tau) + state_avg * tau
        self.act.state_std[:] = self.cri.state_std * (1 - tau) + state_std * tau + 1e-4
        self.cri.state_avg[:] = self.act.state_avg
        self.cri.state_std[:] = self.cri.state_std

        returns_avg = returns.mean(dim=0)
        returns_std = returns.std(dim=0)
        self.cri.value_avg[:] = self.cri.value_avg * (1 - tau) + returns_avg * tau
        self.cri.value_std[:] = self.cri.value_std * (1 - tau) + returns_std * tau + 1e-4

    @staticmethod
    def soft_update(target_net: torch.nn.Module, current_net: torch.nn.Module, tau: float):
        """soft update target network via current network

        target_net: update target network via current network to make training more stable.
        current_net: current network update via an optimizer
        tau: tau of soft target update: `target_net = target_net * (1-tau) + current_net * tau`
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def save_or_load_agent(self, cwd: str, if_save: bool):
        """save or load training files for Agent

        cwd: Current Working Directory. ElegantRL save training files in CWD.
        if_save: True: save files. False: load files.
        """
        assert self.save_attr_names.issuperset({'act', 'act_target', 'act_optimizer'})

        for attr_name in self.save_attr_names:
            file_path = f"{cwd}/{attr_name}.pth"
            print(file_path)
            if if_save:
                torch.save(getattr(self, attr_name), file_path)
            elif os.path.isfile(file_path):
                setattr(self, attr_name, torch.load(file_path, map_location=self.device))

class Actor(nn.Module):
    """
    Actor network for policy learning in Actor-Critic models.

    Args:
        dims (list): Network dimensions.
        state_dim (int): Dimensionality of the state space.
        action_dim (int): Dimensionality of the action space.
    """
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
class Critic(nn.Module):
    """
    Critic network for evaluating the value of taking a particular action in a given state.

    Args:
        dims (list): Network dimensions.
        state_dim (int): Dimensionality of the state space.
        action_dim (int): Dimensionality of the action space.

    Example:
        >>> critic = Critic(dims=[64, 64], state_dim=10, action_dim=2)
    """
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__()
        self.net = build_mlp(dims=[state_dim + action_dim, *dims, 1])
    def forward(self, value: Tensor) -> Tensor:
        return self.net(value)  # Q value
class AgentDDPG(AgentBase):
    """
    Twin Delayed DDPG (Deep Deterministic Policy Gradient) algorithm agent.

    Args:
        net_dims (list): Network dimensions.
        state_dim (int): Dimensionality of the state space.
        action_dim (int): Dimensionality of the action space.
        gpu_id (int): GPU ID. Default is 0.
        args (Config): Configuration arguments. Default is `Config()`.

    Attributes:
        act_class (nn.Module): Actor network class.
        cri_class (nn.Module): Critic network class.
        ...
        explore_noise_std (float): Standard deviation of exploration noise.

    Example:
        >>> agent_ddpg = AgentMIPDDPG(net_dims=[64, 64], state_dim=10, action_dim=2)
    """

    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
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
        Update the neural networks of the agent using the experiences stored in the replay buffer.

        This method performs the optimization of both the actor and critic networks using the experiences
        sampled from the replay buffer. It applies the Twin Delayed DDPG algorithm for updates.

        Args:
            buffer (ReplayBuffer): The replay buffer containing experiences for training.

        Returns:
            Tuple[float, ...]: A tuple containing the average objectives (losses) of the critic and actor updates.
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
        Compute the objective (loss) for the critic network using raw experiences from the replay buffer.

        This method calculates the critic loss using the experiences sampled from the replay buffer. It is used
        when not employing Prioritized Experience Replay (PER).

        Args:
            buffer (ReplayBuffer): The replay buffer from which to sample experiences.
            batch_size (int): The number of samples to draw from the buffer.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the critic loss and the sampled states.
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
        Compute the objective (loss) for the critic network using prioritized experiences from the replay buffer.

        This method calculates the critic loss using the experiences sampled from the replay buffer with
        Prioritized Experience Replay (PER). It considers the importance sampling weights in the loss calculation.

        Args:
            buffer (ReplayBuffer): The replay buffer from which to sample experiences.
            batch_size (int): The number of samples to draw from the buffer.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the weighted critic loss and the sampled states.
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
        Explore the environment for a given number of steps and collect experiences.

        This method is used for collecting experiences from a single environment instance. It can operate
        in either random action mode or based on the current policy of the agent.

        Args:
            env: The environment to interact with.
            horizon_len (int): The number of steps to explore.
            if_random (bool, optional): Flag to determine if actions should be random. Defaults to False.

        Returns:
            [Tensor]: A list of tensors containing states, actions, rewards, and undones collected during exploration.
        """
        # Method implementation...

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

class ActorSAC(nn.Module):
    """
    Actor network for Soft Actor-Critic (SAC) algorithm.

    Attributes:
        enc_s (nn.Module): Encoder network for state input.
        dec_a_avg (nn.Module): Decoder network for action mean.
        dec_a_std (nn.Module): Decoder network for action log standard deviation.
        log_sqrt_2pi (float): Logarithm of the square root of 2π, a constant used in calculations.
        soft_plus (nn.Softplus): Softplus activation function.

    Methods:
        forward(state): Computes the action for a given state.
        get_action(state): Computes the action for exploration.
        get_action_logprob(state): Computes the action and its log probability for a given state.
    """

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
        return self.dec_q1(sa_tmp), self.dec_q2(sa_tmp)  # two Q valuesplease show me all
class AgentSAC(AgentBase):
    """
    Soft Actor-Critic (SAC) agent implementation.

    Attributes:
        act_class (type): Class type for the actor network.
        cri_class (type): Class type for the critic network.
        cri_target (nn.Module): Target critic network for stable training.
        alpha_log (Tensor): Logarithm of the temperature parameter alpha.
        alpha_optimizer (torch.optim.Optimizer): Optimizer for alpha.
        target_entropy (float): Target entropy for policy optimization.

    Methods:
        explore_one_env(env, horizon_len, if_random): Explores an environment for a given horizon length.
        update_net(buffer): Updates the networks using the given replay buffer.
        get_obj_critic_raw(buffer, batch_size): Computes the raw objective for the critic.
        get_obj_critic_per(buffer, batch_size): Computes the PER-adjusted objective for the critic.
    """

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
        """
        Explores a given environment for a specified horizon length.

        This method is used for collecting experiences by interacting with the environment. It can operate in either a random action mode or a policy-based action mode.

        Args:
            env: The environment to be explored.
            horizon_len (int): The number of steps to explore the environment.
            if_random (bool): If True, actions are chosen randomly. If False, actions are chosen based on the current policy.

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

    def update_net(self, buffer: ReplayBuffer) -> Tuple[float, ...]:
        """
        Updates the networks (actor, critic, and temperature parameter) using experiences from the replay buffer.

        This method performs the core updates for the SAC algorithm, including updating the critic network, the temperature parameter for entropy maximization, and the actor network.

        Args:
            buffer (ReplayBuffer): The replay buffer containing experiences for training.

        Returns:
            Tuple[float, float, float]: A tuple containing the average objective values for the critic, actor, and alpha (temperature parameter) updates.
        """

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
        # update_times = int(buffer.add_size * self.repeat_times)
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
            # Check if rewards and undones have the same dimensions as states and actions
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
            # is_weights, is_indices: important sampling `weights, indices` by Prioritized Experience Replay (PER)
            # Check if rewards and undones have the same dimensions as states and actions
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