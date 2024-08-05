import numpy as np
import torch
import os
from torch import nn, Tensor
from typing import Tuple
import math
from torch.nn.utils import clip_grad_norm_

class Config:
    """
    Configuration class for setting up and managing parameters for the agent and environment.

    Attributes:
        num_envs (int): Number of environments.
        agent_class (class): Class of the agent.
        if_off_policy (bool): Indicates whether the DRL algorithm is off-policy or on-policy.
        env_class (class): Class of the environment.
        env_args (dict): Arguments for the environment.
        env_name (str): Name of the environment.
        max_step (int): Maximum number of steps in an episode.
        state_dim (int): Dimension of the state vector.
        action_dim (int): Dimension of the action vector.
        if_discrete (bool): Indicates if the action space is discrete.
        gamma (float): Discount factor for future rewards.
        reward_scale (float): Scale of the reward.
        net_dims (tuple): Dimensions of the MLP layers.
        learning_rate (float): Learning rate for network updates.
        clip_grad_norm (float): Gradient clipping norm.
        state_value_tau (float): Tau for normalizing state and value.
        soft_update_tau (float): Tau for soft target update.
        batch_size (int): Batch size for training.
        target_step (int): Number of steps for target update.
        buffer_size (int): Size of the replay buffer.
        repeat_times (float): Number of times to update the network with the replay buffer.
        if_use_per (bool): Indicates if PER (Prioritized Experience Replay) is used.
        if_use_vtrace (bool): Indicates if V-trace is used.
        random_seed (int): Random seed for reproducibility.
        num_episode (int): Number of episodes for training.
        gpu_id (int): GPU ID for training.
        num_workers (int): Number of workers for data collection.
        num_threads (int): Number of threads for PyTorch.
        learner_gpus (int): GPU ID for the learner.
        run_name (str): Name of the run for data storage.
        cwd (str): Current working directory.
        if_remove (bool): Flag to remove the current working directory.
        train (bool): Flag to indicate training mode.

    Methods:
        init_before_training(): Initializes settings before training starts.
        get_if_off_policy(): Determines if the agent is off-policy based on its name.
        print(): Prints the configuration in a readable format.
        to_dict(): Converts the configuration to a dictionary.
    """
    def __init__(self, agent_class=None, env_class=None, env_args=None):
        self.num_envs = None
        self.agent_class = agent_class  # agent = agent_class(...)
        self.if_off_policy = self.get_if_off_policy()  # whether off-policy or on-policy of DRL algorithm

        '''Argument of environment'''
        self.env_class = env_class  # env = env_class(**env_args)
        self.env_args = env_args  # env = env_class(**env_args)
        if env_args is None:  # dummy env_args
            env_args = {'env_name': None,
                        'num_envs': 1,
                        'max_step': 96,
                        'state_dim': None,
                        'action_dim': None,
                        'if_discrete': None, }
        env_args.setdefault('num_envs', 1)  # `num_envs=1` in default in single env.
        env_args.setdefault('max_step', 96)  # `max_step=12345` in default, which is a large enough value.
        self.env_name = env_args['env_name']  # the name of environment. Be used to set 'cwd'.
        self.num_envs = env_args['num_envs']  # the number of sub envs in vectorized env. `num_envs=1` in single env.
        self.max_step = env_args['max_step']  # the max step number of an episode. 'set as 12345 in default.
        self.state_dim = env_args['state_dim']  # vector dimension (feature number) of state
        self.action_dim = env_args['action_dim']  # vector dimension (feature number) of action
        self.if_discrete = env_args['if_discrete']  # discrete or continuous action space
        '''Arguments for reward shaping'''
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256

        '''Arguments for training'''
        self.net_dims = (64, 32)  # the middle layer dimension of MLP (MultiLayer Perceptron)
        self.learning_rate = 6e-5  # the learning rate for network updating
        self.clip_grad_norm = 3.0  # 0.1 ~ 4.0, clip the gradient after normalization
        self.state_value_tau = 0  # the tau of normalize for value and state `std = (1-std)*std + tau*std`
        self.soft_update_tau = 5e-3  # 2 ** -8 ~= 5e-3. the tau of soft target update `net = (1-tau)*net + tau*net1`
        if self.if_off_policy:  # off-policy
            self.batch_size = int(64)  # num of transitions sampled from replay buffer.
            self.target_step = int(512)  # collect horizon_len step while exploring, then update networks
            self.buffer_size = int(1e6)  # ReplayBuffer size. First in first out for off-policy.
            self.repeat_times = 1.0  # repeatedly update network using ReplayBuffer to keep critic's loss small
            self.if_use_per = False  # use PER (Prioritized Experience Replay) for sparse reward
        else:  # on-policy
            self.batch_size = int(128)  # num of transitions sampled from replay buffer.
            self.target_step = int(2048)  # collect horizon_len step while exploring, then update network
            self.buffer_size = None  # ReplayBuffer size. Empty the ReplayBuffer for on-policy.
            self.repeat_times = 8.0  # repeatedly update network using ReplayBuffer to keep critic's loss small
            self.if_use_vtrace = False  # use V-trace + GAE (Generalized Advantage Estimation) for sparse reward
        self.random_seed=521
        self.num_episode=2000
        self.buffer_size = 500000  # capacity of replay buffer
        '''Arguments for device'''
        self.gpu_id = int(0)  # `int` means the ID of single GPU, -1 means CPU
        self.num_workers = 2  # rollout workers number pre GPU (adjust it to get high GPU usage)
        self.num_threads = 8  # cpu_num for pytorch, `torch.set_num_threads(self.num_threads)`
        self.random_seed = 0  # initialize random seed in self.init_before_training()
        self.learner_gpus = 0  # `int` means the ID of single GPU, -1 means CPU

        '''arguments for creating data storage directory'''

        self.run_name=None
        '''Arguments for save and plot issues'''
        self.cwd = None  # current work directory. None means set automatically
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.train=True
    def init_before_training(self):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)
        if self.cwd is None:
            agent_name = self.agent_class.__name__[5:]
            self.cwd = f'./{agent_name}/{self.run_name}'

        '''remove history'''
        if self.if_remove is None:
            self.if_remove = bool(input(f"| Arguments PRESS 'y' to REMOVE: {self.cwd}? ") == 'y')
        if self.if_remove:
            import shutil
            shutil.rmtree(self.cwd, ignore_errors=True)
            print(f"| Arguments Remove cwd: {self.cwd}")
        else:
            print(f"| Arguments Keep cwd: {self.cwd}")
        os.makedirs(self.cwd, exist_ok=True)
    def get_if_off_policy(self) -> bool:
        agent_name = self.agent_class.__name__ if self.agent_class else ''
        on_policy_names = ('SARSA', 'VPG', 'A2C', 'A3C', 'TRPO', 'PPO', 'MPO')
        return all([agent_name.find(s) == -1 for s in on_policy_names])

    def print(self):
        from pprint import pprint
        pprint(vars(self))  # prints out args in a neat, readable format

    def to_dict(self):
        return vars(self)
def get_optim_param(optimizer: torch.optim) -> list:  # backup
    """
    Extracts parameters from the optimizer state.

    Args:
        optimizer (torch.optim): The optimizer from which to extract parameters.

    Returns:
        list: A list of parameters extracted from the optimizer state.
    """
    params_list = []
    for params_dict in optimizer.state_dict()["state"].values():
        params_list.extend([t for t in params_dict.values() if isinstance(t, torch.Tensor)])
    return params_list
def build_mlp(dims: [int]) -> nn.Sequential:  # MLP (MultiLayer Perceptron)
    """
    Builds a Multi-Layer Perceptron (MLP) network.

    Args:
        dims (list of int): A list containing the dimensions of each layer in the MLP.

    Returns:
        nn.Sequential: The constructed MLP network.
    """
    net_list = list()
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
    del net_list[-1]  # remove the activation of output layer
    return nn.Sequential(*net_list)
class ReplayBuffer:  # for off-policy
    """
    Replay Buffer for storing and sampling experiences for off-policy reinforcement learning algorithms.

    Attributes:
        max_size (int): Maximum size of the buffer.
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        gpu_id (int): GPU ID for storing the buffer.
        num_seqs (int): Number of sequences in the buffer.
        if_use_per (bool): Flag to use Prioritized Experience Replay.
        args (Config): Configuration object with additional parameters.

    Methods:
        update(items): Updates the buffer with new experiences.
        sample(batch_size): Samples a batch of experiences from the buffer.
        sample_for_per(batch_size): Samples a batch with prioritization.
        td_error_update_for_per(is_indices, td_error): Updates the priorities based on TD error.
        save_or_load_history(cwd, if_save): Saves or loads the buffer history.
    """
    def __init__(self,
                 max_size: int,
                 state_dim: int,
                 action_dim: int,
                 gpu_id: int = 0,
                 num_seqs: int = 1,
                 if_use_per: bool = False,
                 args: Config = Config()):
        self.p = 0  # pointer
        self.if_full = False
        self.cur_size = 0
        self.add_size = 0
        self.add_item = None
        self.max_size = max_size
        self.num_seqs = num_seqs
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        self.args=args
        """The struction of ReplayBuffer (for examples, num_seqs = num_workers * num_envs == 2*4 = 8
        ReplayBuffer:
        worker0 for env0:   sequence of sub_env0.0  self.states  = Tensor[s, s, ..., s, ..., s]     
                                                    self.actions = Tensor[a, a, ..., a, ..., a]   
                                                    self.rewards = Tensor[r, r, ..., r, ..., r]   
                                                    self.undones = Tensor[d, d, ..., d, ..., d]
                                                                          <-----max_size----->
                                                                          <-cur_size->
                                                                                     ↑ pointer
                            sequence of sub_env0.1  s, s, ..., s    a, a, ..., a    r, r, ..., r    d, d, ..., d
                            sequence of sub_env0.2  s, s, ..., s    a, a, ..., a    r, r, ..., r    d, d, ..., d
                            sequence of sub_env0.3  s, s, ..., s    a, a, ..., a    r, r, ..., r    d, d, ..., d
        worker1 for env1:   sequence of sub_env1.0  s, s, ..., s    a, a, ..., a    r, r, ..., r    d, d, ..., d
                            sequence of sub_env1.1  s, s, ..., s    a, a, ..., a    r, r, ..., r    d, d, ..., d
                            sequence of sub_env1.2  s, s, ..., s    a, a, ..., a    r, r, ..., r    d, d, ..., d
                            sequence of sub_env1.3  s, s, ..., s    a, a, ..., a    r, r, ..., r    d, d, ..., d

        D: done=True
        d: done=False
        sequence of transition: s-a-r-d, s-a-r-d, s-a-r-D  s-a-r-d, s-a-r-d, s-a-r-d, s-a-r-d, s-a-r-D  s-a-r-d, ...
                                <------trajectory------->  <----------trajectory--------------------->  <-----------
        """
        self.states = torch.empty((max_size, num_seqs, state_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.empty((max_size, num_seqs, action_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.empty((max_size, num_seqs), dtype=torch.float32, device=self.device)
        self.undones = torch.empty((max_size, num_seqs), dtype=torch.float32, device=self.device)

        self.if_use_per = if_use_per
        if if_use_per:
            self.sum_trees = [SumTree(buf_len=max_size) for _ in range(num_seqs)]
            self.per_alpha = getattr(args, 'per_alpha', 0.6)  # alpha = (Uniform:0, Greedy:1)
            self.per_beta = getattr(args, 'per_beta', 0.4)  # alpha = (Uniform:0, Greedy:1)
            """PER.  Prioritized Experience Replay. Section 4
            alpha, beta = 0.7, 0.5 for rank-based variant
            alpha, beta = 0.6, 0.4 for proportional variant
            """
        else:
            self.sum_trees = None
            self.per_alpha = None
            self.per_beta = None

    def update(self, items: Tuple[Tensor, ...]):
        """
        Updates the replay buffer with new experience tuples.

        Args:
            items (Tuple[Tensor, ...]): A tuple containing tensors of states, actions, rewards, and undones.
                                        Each tensor should have a shape that matches the expected dimensions
                                        for states, actions, rewards, and undones respectively.

        Description:
            This method updates the replay buffer with new experiences. It handles the buffer's internal
            pointers and ensures that new data is added correctly, even when the buffer is full. If the buffer
            is full, it starts overwriting the oldest data. In case of using Prioritized Experience Replay (PER),
            it updates the sum trees with new priorities.
        """
        self.add_item = items
        states, actions, rewards, undones = items
        assert states.shape[1:] == (self.args.num_envs, self.args.state_dim)
        assert actions.shape[1:] == (self.args.num_envs, self.args.action_dim)
        assert rewards.shape[1:] == (self.args.num_envs,)
        assert undones.shape[1:] == (self.args.num_envs,)
        self.add_size = rewards.shape[0]

        p = self.p + self.add_size  # pointer
        if p > self.max_size:
            self.if_full = True
            p0 = self.p
            p1 = self.max_size
            p2 = self.max_size - self.p
            p = p - self.max_size

            self.states[p0:p1], self.states[0:p] = states[:p2], states[-p:]
            self.actions[p0:p1], self.actions[0:p] = actions[:p2], actions[-p:]
            self.rewards[p0:p1], self.rewards[0:p] = rewards[:p2], rewards[-p:]
            self.undones[p0:p1], self.undones[0:p] = undones[:p2], undones[-p:]
        else:
            self.states[self.p:p] = states
            self.actions[self.p:p] = actions
            self.rewards[self.p:p] = rewards
            self.undones[self.p:p] = undones

        if self.if_use_per:
            '''data_ids for single env'''
            data_ids = torch.arange(self.p, p, dtype=torch.long, device=self.device)
            if p > self.max_size:
                data_ids = torch.fmod(data_ids, self.max_size)

            '''apply data_ids for vectorized env'''
            for sum_tree in self.sum_trees:
                sum_tree.update_ids(data_ids=data_ids.cpu(), prob=10.)

        self.p = p
        self.cur_size = self.max_size if self.if_full else self.p

    def sample(self, batch_size: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Samples a batch of experiences from the replay buffer.

        Args:
            batch_size (int): The size of the batch to sample.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: A tuple containing batches of states, actions, rewards,
                                                           undones, and next_states. Each tensor in the tuple has
                                                           dimensions corresponding to the batch size.

        Description:
            This method randomly samples a batch of experiences from the replay buffer. It is typically used in
            off-policy algorithms where random sampling of experiences is required for training the agent.
        """

        sample_len = self.cur_size - 1

        ids = torch.randint(sample_len * self.num_seqs, size=(batch_size,), requires_grad=False)
        ids0 = torch.fmod(ids, sample_len)  # ids % sample_len
        ids1 = torch.div(ids, sample_len, rounding_mode='floor')  # ids // sample_len

        return (self.states[ids0, ids1],
                self.actions[ids0, ids1],
                self.rewards[ids0, ids1],
                self.undones[ids0, ids1],
                self.states[ids0 + 1, ids1],)  # next_state

    def sample_for_per(self, batch_size: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Samples a batch of experiences using Prioritized Experience Replay.

        Args:
            batch_size (int): The size of the batch to sample.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]: A tuple containing batches of states,
                                                                           actions, rewards, undones, next_states,
                                                                           importance sampling weights, and indices.
                                                                           Each tensor in the tuple has dimensions
                                                                           corresponding to the batch size.

        Description:
            This method samples experiences using Prioritized Experience Replay (PER). It uses importance sampling
            to give more priority to experiences with higher expected learning value. This is particularly useful
            in scenarios where some experiences may be more significant than others for learning.
        """
        beg = -self.max_size
        end = (self.cur_size - self.max_size) if (self.cur_size < self.max_size) else -1

        '''get is_indices, is_weights'''
        is_indices: list = []
        is_weights: list = []

        assert batch_size % self.num_seqs == 0
        sub_batch_size = batch_size // self.num_seqs
        for env_i in range(self.num_seqs):
            sum_tree = self.sum_trees[env_i]
            _is_indices, _is_weights = sum_tree.important_sampling(batch_size, beg, end, self.per_beta)
            is_indices.append(_is_indices + sub_batch_size * env_i)
            is_weights.append(_is_weights)

        is_indices: Tensor = torch.hstack(is_indices).to(self.device)
        is_weights: Tensor = torch.hstack(is_weights).to(self.device)

        ids0 = torch.fmod(is_indices, self.cur_size)  # is_indices % sample_len
        ids1 = torch.div(is_indices, self.cur_size, rounding_mode='floor')  # is_indices // sample_len
        return (
            self.states[ids0, ids1],
            self.actions[ids0, ids1],
            self.rewards[ids0, ids1],
            self.undones[ids0, ids1],
            self.states[ids0 + 1, ids1],  # next_state
            is_weights,  # important sampling weights
            is_indices,  # important sampling indices
        )

    def td_error_update_for_per(self, is_indices: Tensor, td_error: Tensor):  # td_error = (q-q).detach_().abs()
        """
        Updates the priorities in the sum trees based on the TD error.

        Args:
            is_indices (Tensor): Tensor containing indices of sampled experiences.
            td_error (Tensor): Tensor containing the Temporal Difference (TD) error for each sampled experience.

        Description:
            This method updates the priorities in the sum trees for each experience based on the provided TD error.
            It is an essential part of the Prioritized Experience Replay mechanism, ensuring that experiences
            with higher TD error (and thus potentially higher learning value) have a higher chance of being sampled.
        """
        prob = td_error.clamp(1e-8, 10).pow(self.per_alpha).squeeze(-1)

        # self.sum_tree.update_ids(is_indices.cpu(), prob.cpu())
        batch_size = td_error.shape[0]
        sub_batch_size = batch_size // self.num_seqs
        for env_i in range(self.num_seqs):
            sum_tree = self.sum_trees[env_i]
            slice_i = env_i * sub_batch_size
            slice_j = slice_i + sub_batch_size

            sum_tree.update_ids(is_indices[slice_i:slice_j].cpu(), prob[slice_i:slice_j].cpu())

    def save_or_load_history(self, cwd: str, if_save: bool):
        """
        Saves or loads the replay buffer history to/from disk.

        Args:
            cwd (str): The current working directory where the buffer history will be saved or loaded from.
            if_save (bool): A flag indicating whether to save (True) or load (False) the buffer history.

        Description:
            This method either saves the current state of the replay buffer to disk or loads it from disk.
            This is useful for persisting the replay buffer across different training sessions or for
            transferring the buffer state between different instances.
        """
        item_names = (
            (self.states, "states"),
            (self.actions, "actions"),
            (self.rewards, "rewards"),
            (self.undones, "undones"),
        )

        if if_save:
            for item, name in item_names:
                if self.cur_size == self.p:
                    buf_item = item[:self.cur_size]
                else:
                    buf_item = torch.vstack((item[self.p:self.cur_size], item[0:self.p]))
                file_path = f"{cwd}/replay_buffer_{name}.pth"
                print(f"| buffer.save_or_load_history(): Save {file_path}")
                torch.save(buf_item, file_path)

        elif all([os.path.isfile(f"{cwd}/replay_buffer_{name}.pth") for item, name in item_names]):
            max_sizes = []
            for item, name in item_names:
                file_path = f"{cwd}/replay_buffer_{name}.pth"
                print(f"| buffer.save_or_load_history(): Load {file_path}")
                buf_item = torch.load(file_path)

                max_size = buf_item.shape[0]
                item[:max_size] = buf_item
                max_sizes.append(max_size)
            assert all([max_size == max_sizes[0] for max_size in max_sizes])
            self.cur_size = self.p = max_sizes[0]
            self.if_full = self.cur_size == self.max_size
class SumTree:
    """
    Binary Search Tree for efficient sampling in Prioritized Experience Replay.

    Attributes:
        buf_len (int): Length of the buffer.
        max_len (int): Maximum length of the tree.
        depth (int): Depth of the tree.
        tree (Tensor): Tensor representing the tree structure.

    Methods:
        update_id(data_id, prob): Updates a single node in the tree.
        update_ids(data_ids, prob): Updates multiple nodes in the tree.
        get_leaf_id_and_value(v): Retrieves the leaf ID and value for a given value.
        important_sampling(batch_size, beg, end, per_beta): Performs important sampling for a batch.
    """
    def __init__(self, buf_len: int):
        """
        Initializes the SumTree object.

        Args:
            buf_len (int): The length of the buffer for which this SumTree is being used.

        Description:
            This method initializes a SumTree data structure. The SumTree is a binary tree where each node's
            value is the sum of its children's values. This structure is particularly useful for efficiently
            implementing Prioritized Experience Replay (PER) in reinforcement learning.
        """
        self.buf_len = buf_len  # replay buffer len
        self.max_len = (buf_len - 1) + buf_len  # parent_nodes_num + leaf_nodes_num
        self.depth = math.ceil(math.log2(self.max_len))

        self.tree = torch.zeros(self.max_len, dtype=torch.float32)

    def update_id(self, data_id: int, prob=10):  # 10 is max_prob
        """
         Updates the priority of a single data point in the SumTree.

         Args:
             data_id (int): The index of the data point in the buffer.
             prob (float, optional): The new priority value for the data point. Defaults to 10, which is considered the maximum priority.

         Description:
             This method updates the priority of a single data point in the SumTree. It adjusts the values in the tree
             to maintain the sum property after the update. This is used in PER to adjust the sampling probability of experiences.
         """
        tree_id = data_id + self.buf_len - 1

        delta = prob - self.tree[tree_id]
        self.tree[tree_id] = prob

        for depth in range(self.depth - 2):  # propagate the change through tree
            tree_id = (tree_id - 1) // 2  # faster than the recursive loop
            self.tree[tree_id] += delta

    def update_ids(self, data_ids: Tensor, prob: Tensor = 10.):  # 10 is max_prob
        """
        Updates the priorities of multiple data points in the SumTree.

        Args:
            data_ids (Tensor): A tensor of indices of the data points in the buffer.
            prob (Tensor, optional): A tensor of new priority values for the data points. Defaults to 10 for each, which is considered the maximum priority.

        Description:
            This method updates the priorities of multiple data points in the SumTree simultaneously. It ensures that
            the sum property of the tree is maintained after the updates. This method is typically used in batch updates
            in PER.
        """
        l_ids = data_ids + self.buf_len - 1

        self.tree[l_ids] = prob
        for depth in range(self.depth - 2):  # propagate the change through tree
            p_ids = torch.div(l_ids - 1, 2, rounding_mode='floor').unique()  # parent indices
            l_ids = p_ids * 2 + 1  # left children indices
            r_ids = l_ids + 1  # right children indices
            self.tree[p_ids] = self.tree[l_ids] + self.tree[r_ids]

            l_ids = p_ids

    def get_leaf_id_and_value(self, v) -> Tuple[int, float]:
        """
        Retrieves the leaf node ID and its value based on a given value.

        Args:
            v (float): The value to search for in the tree.

        Returns:
            Tuple[int, float]: A tuple containing the ID of the leaf node and its value.

        Description:
            This method searches the SumTree to find the leaf node whose value corresponds to the given value 'v'.
            It is used during the sampling process in PER to select experiences based on their priority values.
            Tree structure and array storage:
        Tree index:
              0       -> storing priority sum
            |  |
          1     2
         | |   | |
        3  4  5  6    -> storing priority for transitions
        Array type for storing: [0, 1, 2, 3, 4, 5, 6]
        """

        p_id = 0  # the leaf's parent node

        for depth in range(self.depth - 2):  # propagate the change through tree
            l_id = min(2 * p_id + 1, self.max_len - 1)  # the leaf's left node
            r_id = l_id + 1  # the leaf's right node
            if v <= self.tree[l_id]:
                p_id = l_id
            else:
                v -= self.tree[l_id]
                p_id = r_id
        return p_id, self.tree[p_id]  # leaf_id and leaf_value

    def important_sampling(self, batch_size: int, beg: int, end: int, per_beta: float) -> Tuple[Tensor, Tensor]:
        """
        Performs important sampling to select indices and compute weights for experiences.

        Args:
            batch_size (int): The number of samples to draw.
            beg (int): The beginning index for sampling.
            end (int): The ending index for sampling.
            per_beta (float): The beta parameter for PER, controlling the degree of importance sampling.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing tensors of indices and corresponding weights for the sampled experiences.

        Description:
            This method performs important sampling based on the priorities in the SumTree. It is used in PER to
            select experiences non-uniformly, giving more priority to experiences with higher expected learning value.
        """
        # get random values for searching indices with proportional prioritization
        values = (torch.arange(batch_size) + torch.rand(batch_size)) * (self.tree[0] / batch_size)

        # get proportional prioritization
        leaf_ids, leaf_values = list(zip(*[self.get_leaf_id_and_value(v) for v in values]))
        leaf_ids = torch.tensor(leaf_ids, dtype=torch.long)
        leaf_values = torch.tensor(leaf_values, dtype=torch.float32)

        indices = leaf_ids - (self.buf_len - 1)
        if indices.min()<0:
            print(f'the wrong indice is{indices.min()}')
            print(f'the whole indices is {indices}')
        # assert 0 <= indices.min()
        assert indices.max() < self.buf_len

        prob_ary = leaf_values / self.tree[beg:end].min()
        weights = torch.pow(prob_ary, -per_beta)
        return indices, weights
def get_episode_return(env, act, device):
    """
    Calculates the return of an episode.

    Args:
        env: The environment to interact with.
        act: The action function to use.
        device: The device to perform computations on.

    Returns:
        Tuple containing episode return, violation time, violation value, rewards for power, good actions, and penalties, and the list of states.
    """
    env.train=False
    episode_return = 0.0
    violation_time = 0
    reward_for_power = 0
    reward_for_good_action = 0
    reward_for_penalty = 0
    state_list=[]
    state = env.reset()
    print(f'the year:{env.year},month:{env.month},day:{env.day} is used for testing this episode')

    # Initialize arrays to store voltage values
    v_before_control_array = np.zeros((96, len(env.battery_list)))
    v_after_control_array = np.zeros((96, len(env.battery_list)))

    violation_value = 0.0

    for i in range(96):
        s_tensor = torch.as_tensor((state,), device=device, dtype=torch.float)
        a_tensor = act(s_tensor)


        action = a_tensor.detach().cpu().numpy()[0]
        next_state, reward, done,_ = env.step(action)
        state_list.append(state)


        for i in range(len(env.battery_list)):
            violation = min(0, 0.05 - abs(1.0 - env.after_control[env.battery_list[i]]))
            if violation < 0:
                violation_time += 1
        violation_value += violation


        reward_for_power += env.reward_for_power
        reward_for_good_action += 0
        reward_for_penalty += env.reward_for_penalty
        episode_return += reward
        state = next_state
        if done:
            break
    return episode_return, violation_time, violation_value, reward_for_power, reward_for_good_action, reward_for_penalty, state_list