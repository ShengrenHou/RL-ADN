import os
import random

import numpy as np
import pandas as pd
import gym
from gym import spaces
import copy as cp
import pandapower as pp

from rl_adn.utility.utils import create_pandapower_net
from rl_adn.data_manager.data_manager import GeneralPowerDataManager
from rl_adn.utility.grid import GridTensor
from rl_adn.environments.battery import Battery,battery_parameters

env_config={
  "voltage_limits": [0.95, 1.05],
  "algorithm": "Laurent",
  "battery_list": [11,15,26,29,33],
  "year": 2020,
  "month": 1,
  "day": 1,
  "train": True,
  "state_pattern": "default",
  "network_info": {'vm_pu':1.0,'s_base':1000,
                'bus_info_file': '../rl_adn/data_sources/network_data/node_34/Nodes_34.csv',
                'branch_info_file': '../rl_adn/data_sources/network_data/node_34/Lines_34.csv'},
  "time_series_data_path": "../rl_adn/data_sources/time_series_data/34_node_time_series.csv"
}

class PowerNetEnv(gym.Env):
    """
        Custom Environment for Power Network Management.

        The environment simulates a power network, and the agent's task is to
        manage this network by controlling the batteries attached to various nodes.

        Attributes:
            voltage_limits (tuple): Limits for the voltage.
            algorithm (str): Algorithm choice. Can be 'Laurent' or 'PandaPower'.
            battery_list (list): List of nodes where batteries are attached.
            year (int): Current year in simulation.
            month (int): Current month in simulation.
            day (int): Current day in simulation.
            train (bool): Whether the environment is in training mode.
            state_pattern (str): Pattern for the state representation.
            network_info (dict): Information about the network.
            node_num (int): Number of nodes in the network.
            action_space (gym.spaces.Box): Action space of the environment.
            data_manager (GeneralPowerDataManager): Manager for the time-series data.
            episode_length (int): Length of an episode.
            state_length (int): Length of the state representation.
            state_min (np.ndarray): Minimum values for each state element.
            state_max (np.ndarray): Maximum values for each state element.
            state_space (gym.spaces.Box): State space of the environment.
            current_time (int): Current timestep in the episode.
            after_control (np.ndarray): Voltages after control is applied.

        Args:
            env_config_path (str): Path to the environment configuration file.

        """
    def __init__(self, env_config: dict = env_config) -> None:
        """
         Initialize the PowerNetEnv environment.
         :param env_config_path: Path to the environment configuration file. Defaults to 'env_config.py'.
         :type env_config_path: str
         """
        config = env_config

        self.voltage_limits = config['voltage_limits']
        self.algorithm = config['algorithm']
        self.battery_list = config['battery_list']
        self.year = config['year']
        self.month = config['month']
        self.day = config['day']
        self.train = config['train']
        self.state_pattern = config['state_pattern']
        self.network_info=config['network_info']
        # network_info for building the network
        if self.network_info == 'None':
            print('create basic 34 node IEEE network, when initial data is not identified')
            self.network_info={'vm_pu':1.0,'s_base':1000,
                'bus_info_file': '../rl_adn/data_sources/network_data/node_34/Nodes_34.csv',
                'branch_info_file': '../rl_adn/data_sources/network_data/node_34/Lines_34.csv'}
            self.s_base=1000
            self.node_num=34
        else:
            self.s_base=self.network_info['s_base']
            network_bus_info=pd.read_csv(self.network_info['bus_info_file'])
            self.node_num=len((network_bus_info.NODES))
        # Conditional initialization of the distribution network based on the chosen algorithm
        if self.algorithm == "Laurent":
            # Logic for initializing with GridTensor
            self.net=GridTensor(self.network_info['bus_info_file'],
                       self.network_info['branch_info_file'])
            self.net.Q_file = np.zeros(33)
            self.dense_Ybus = self.net._make_y_bus().toarray()


        elif self.algorithm == "PandaPower":
            # Logic for initializing with PandaPower
            self.net=create_pandapower_net(self.network_info)
        else:
            raise ValueError("Invalid algorithm choice. Please choose 'Laurent' or 'PandaPower'.")

        if not self.battery_list:
            raise ValueError("No batteries specified!")
        for node_index in self.battery_list:
            battery = Battery(battery_parameters)
            setattr(self, f"battery_{node_index}", battery)
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.battery_list), 1), dtype=np.float32)
        self.data_manager = GeneralPowerDataManager(config['time_series_data_path'])
        self.episode_length:int = 24*60/self.data_manager.time_interval

        if self.state_pattern=='default':
            self.state_length=len(self.battery_list)*2+self.node_num+2
            print(self.data_manager.active_power_min)
            print(self.data_manager.price_min)
            self.state_min = np.array([self.data_manager.active_power_min, 0.2, self.data_manager.price_min, 0.0, 0.5])
            self.state_max = np.array([self.data_manager.active_power_max, 0.8, self.data_manager.price_max, self.episode_length-1, 1.5])
        else:
            raise ValueError("Invalid value for 'state_pattern'. Expected 'default' or define by yourself.")

        self.state_space=spaces.Box(low=-2, high=2,shape=(self.state_length,), dtype=np.float32)

    def reset(self) -> np.ndarray:
        """
        Reset the environment to its initial state and return the initial state.

        :return: The normalized initial state of the environment.
        :rtype: np.ndarray
        """
        self._reset_date()
        self._reset_time()
        self._reset_batteries()
        return self._build_state()
    def _reset_date(self) -> None:
        """
        Resets the date for the next episode.
        """
        if self.train:
            self.year, self.month, self.day = random.choice(self.data_manager.train_dates)
        else:
            self.year, self.month, self.day=random.choice(self.data_manager.test_dates)

    def _reset_time(self) -> None:
        """
        Resets the time for the next episode.
        """
        self.current_time=0
    def _reset_batteries(self) -> None:
        """
        Resets the batteries for the next episode.
        """
        for node_index in self.battery_list:
            getattr(self, f"battery_{node_index}").reset()

    def _build_state(self) -> np.ndarray:
        """
        Builds the current state of the environment based on the current time and data from PowerDataManager.

        Returns:
            normalized_state (np.ndarray): The current state of the environment, normalized between 0 and 1.
                The state includes the following variables:
                - Netload power
                - SOC (State of Charge) of the last battery in the battery list
                - Price of the energy
                - Time state of the day
                - Voltage from estimation
        """
        # TODO: modify get state observation to fit new resources and data
        obs = self._get_obs()
        if self.state_pattern == 'default':
            active_power = np.array(list(obs['node_data']['active_power'].values()))
            price = obs['price']
            soc_list = np.array([obs['battery_data']['soc'][f'battery_{node_index}'] for node_index in self.battery_list])
            vm_pu_battery = np.array([obs['node_data']['voltage'][f'node_{node_index}'] for node_index in self.battery_list])
            state=np.concatenate((active_power,soc_list,[price],[self.current_time],vm_pu_battery))
            self.state = state
            normalized_state = self._normalize_state(state)
            self.normalized_state = normalized_state
        return normalized_state
    def _split_state(self, state):
        net_load_length = self.node_num
        num_batteries = len(self.battery_list)

        soc_all_length = num_batteries
        vm_pu_battery_nodes_length = num_batteries

        soc_all_start = net_load_length
        price_start = soc_all_start + soc_all_length
        current_time_start = price_start + 1
        vm_pu_battery_nodes_start = current_time_start + 1

        net_load = state[:net_load_length]
        soc_all = state[soc_all_start:soc_all_start + soc_all_length]
        price = np.array([state[price_start]])
        current_time = np.array([state[current_time_start]])
        vm_pu_battery_nodes = state[vm_pu_battery_nodes_start:]

        return net_load, soc_all, price, current_time, vm_pu_battery_nodes
    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        Normalizes the state variables.

        Parameters:
            state (np.ndarray): The current state of the environment.

        Returns:
            np.ndarray: The normalized state of the environment.
        """
        state[:self.node_num]=(state[:self.node_num]-self.state_min[0])/(self.state_max[0] - self.state_min[0])
        state[self.node_num:self.node_num+len(self.battery_list)]=(state[self.node_num:self.node_num+len(self.battery_list)]-self.state_min[1])/(self.state_max[1] - self.state_min[1])
        state[self.node_num+len(self.battery_list):self.node_num+len(self.battery_list)+1]=(state[self.node_num+len(self.battery_list):self.node_num+len(self.battery_list)+1]-self.state_min[2])/(self.state_max[2] - self.state_min[2])
        state[self.node_num+len(self.battery_list)+1:self.node_num+len(self.battery_list)+2]=(state[self.node_num+len(self.battery_list)+1:self.node_num+len(self.battery_list)+2]-self.state_min[3])/(self.state_max[3] - self.state_min[3])
        normalized_state=state
        return normalized_state
    def _denormalize_state(self, normalized_state: np.ndarray) -> np.ndarray:
        """
        Denormalizes the state variables.

        Parameters:
            normalized_state (np.ndarray): The normalized state of the environment.

        Returns:
            np.ndarray: The denormalized state of the environment.
        """
        normalized_state[:self.node_num]=normalized_state[:self.node_num]*(self.state_max[0] - self.state_min[0]) + self.state_min[0]

        normalized_state[self.node_num:self.node_num+len(self.battery_list)]=normalized_state[self.node_num:self.node_num+len(self.battery_list)]*(self.state_max[1] - self.state_min[1]) + self.state_min[1]
        normalized_state[self.node_num+len(self.battery_list):self.node_num+len(self.battery_list)+1]=normalized_state[self.node_num+len(self.battery_list):self.node_num+len(self.battery_list)+1]*(self.state_max[2] - self.state_min[2]) + self.state_min[2]
        normalized_state[self.node_num+len(self.battery_list)+1:self.node_num+len(self.battery_list)+2]=normalized_state[self.node_num+len(self.battery_list)+1:self.node_num+len(self.battery_list)+2]*(self.state_max[3] - self.state_min[3]) + self.state_min[3]
        denormalized_state = normalized_state
        return denormalized_state
    def _get_obs(self):
        """
        Executes the power flow based on the chosen algorithm and returns the observations.

        Returns:
            dict: The observation dictionary containing various state elements.
        """
        if self.state_pattern=='default':
            one_slot_data = self.data_manager.select_timeslot_data(self.year, self.month, self.day, self.current_time)


            if self.algorithm == "Laurent":
                # This is where bugs comes from, if we don't use copy, this slice is actually creating a view of originally data.
                active_power = cp.copy(one_slot_data[0:34])
                renewable_active_power=one_slot_data[34:68]
                self.active_power=(active_power-renewable_active_power)[1:34]
                reactive_power=np.zeros(33)
                price=one_slot_data[-1]
                self.solution = self.net.run_pf(active_power=self.active_power)

                obs = {'node_data': {'voltage': {}, 'active_power': {}, 'reactive_power': {}, 'renewable_active_power': {}},
                       'battery_data': {'soc': {}}, 'price': {}, 'aux': {}}

                for node_index in range(len(self.net.bus_info.NODES)):#NODES[1-34], node_index[0-33]
                    if node_index==0:
                        obs['node_data']['voltage'][f'node_{node_index}']=1.0
                        obs['node_data']['active_power'][f'node_{node_index}']=0.0
                        obs['node_data']['renewable_active_power'][f'node_{node_index}'] = 0.0
                    else:
                        obs['node_data']['voltage'][f'node_{node_index}'] = abs(self.solution['v'].T[node_index - 1]).squeeze()
                        obs['node_data']['active_power'][f'node_{node_index}'] = active_power[node_index - 1]
                        obs['node_data']['renewable_active_power'][f'node_{node_index}'] = renewable_active_power[node_index - 1]
                for node_index in self.battery_list:
                    obs['battery_data']['soc'][f'battery_{node_index}'] = getattr(self, f'battery_{node_index}').SOC()
                obs['price'] = price
            else:
                active_power = one_slot_data[0:34]
                active_power[0] = 0
                renewable_active_power = one_slot_data[34:68]
                renewable_active_power[0]=0
                price = one_slot_data[-1]
                for bus_index in self.net.load.bus.index:
                    self.net.load.p_mw[bus_index] = (active_power[bus_index]-renewable_active_power[bus_index]) / self.s_base
                    self.net.load.q_mvar [bus_index]=0
                pp.runpp(self.net, algorithm='nr')
                v_real = self.net.res_bus["vm_pu"].values * np.cos(np.deg2rad(self.net.res_bus["va_degree"].values))
                v_img = self.net.res_bus["vm_pu"].values * np.sin(np.deg2rad(self.net.res_bus["va_degree"].values))
                v_result = v_real + 1j * v_img

                obs = {'node_data': {'voltage': {}, 'active_power': {}, 'reactive_power': {}, 'renewable_active_power': {}},
                       'battery_data': {'soc': {}}, 'price': {}, 'aux': {}}

                for node_index in self.net.load.bus.index:
                    bus_idx = self.net.load.at[node_index, 'bus']
                    obs['node_data']['voltage'][f'node_{node_index}'] = self.net.res_bus.vm_pu.at[bus_idx]
                    obs['node_data']['active_power'][f'node_{node_index}'] = active_power[node_index]
                    obs['node_data']['reactive_power'][f'node_{node_index}'] = self.net.res_load.q_mvar[node_index]
                    obs['node_data']['renewable_active_power'][f'node_{node_index}'] = renewable_active_power [node_index]
                for node_index in self.battery_list:
                    obs['battery_data']['soc'][f'battery_{node_index}'] = getattr(self, f'battery_{node_index}').SOC()
                obs['price'] = price
        else:
            raise ValueError('please redesign the get obs function to fit the pattern you want')
        return obs

    def _apply_battery_actions(self, action):
        '''apply action to battery charge/discharge, update the battery condition, excute power flow, update the network condition'''
        if self.state_pattern == 'default':
            if self.algorithm == "Laurent":
                v = self.solution["v"]
                v_totall = np.insert(v, 0, 1)
                current_each_node = np.matmul(self.dense_Ybus, v_totall)
                power_imported_from_ex_grid_before = current_each_node[0].real

                for i, node_index in enumerate(self.battery_list):
                    getattr(self, f"battery_{node_index}").step(action[i])
                    self.active_power[node_index-1] += getattr(self, f"battery_{node_index}").energy_change
                self.solution = self.net.run_pf(active_power=self.active_power)


                v = self.solution["v"]
                v_totall = np.insert(v, 0, 1)
                vm_pu_after_control =cp.deepcopy(abs(v_totall))
                vm_pu_after_control_bat = np.squeeze(vm_pu_after_control)[self.battery_list]
                self.after_control = vm_pu_after_control
                current_each_node = np.matmul(self.dense_Ybus, v_totall)
                power_imported_from_ex_grid_after = current_each_node[0].real
                saved_energy = power_imported_from_ex_grid_before - power_imported_from_ex_grid_after
            else:
                power_imported_from_ex_grid_before = cp.deepcopy(self.net.res_ext_grid['p_mw'])

                for i, node_index in enumerate(self.battery_list):
                    getattr(self, f"battery_{node_index}").step(action[i])
                    self.net.load.p_mw[node_index] += getattr(self, f"battery_{node_index}").energy_change / 1000
                pp.runpp(self.net, algorithm='nr')
                vm_pu_after_control = cp.deepcopy(self.net.res_bus.vm_pu).to_numpy(dtype=float)
                vm_pu_after_control_bat = vm_pu_after_control[self.battery_list]


                self.after_control = vm_pu_after_control
                power_imported_from_ex_grid_after = self.net.res_ext_grid['p_mw']
                saved_energy=power_imported_from_ex_grid_before-power_imported_from_ex_grid_after
        else:
            raise ValueError('Expected default or define yourself based on the goal')
        return saved_energy, vm_pu_after_control_bat

    def step(self, action: np.ndarray) -> tuple:
        """
        Advance the environment by one timestep based on the provided action.

        :param action: Action to execute.
        :type action: np.ndarray
        :return: Tuple containing the next normalized observation, the reward, a boolean indicating if the episode has ended, and additional info.
        :rtype: tuple
        """

        current_normalized_obs = self.normalized_state
        info = current_normalized_obs


        # Apply battery actions and get updated observations
        saved_energy, vm_pu_after_control_bat=self._apply_battery_actions(action)

        reward = self._calculate_reward(current_normalized_obs, vm_pu_after_control_bat, saved_energy)

        finish = (self.current_time == self.episode_length - 1)
        self.current_time += 1
        if finish:
            self.current_time = 0
            next_normalized_obs = self.reset()
        else:
            next_normalized_obs = self._build_state()
        return next_normalized_obs, float(reward), finish, info

    def _calculate_reward(self, current_normalized_obs: np.ndarray, vm_pu_after_control_bat: np.ndarray, saved_power: float) -> float:
        """
        Calculate the reward based on the current observation and saved power. the default version is to calculate the battey saved energy
        based on the current price

        Parameters:
            current_normalized_obs (np.ndarray): The current normalized observations.
            vm_pu_after_control_bat (np.ndarray): The voltage after control at battery locations.
            saved_power (float): The amount of power saved.

        Returns:
            float: Calculated reward.
        """
        if self.state_pattern=='default':
            reward_for_power = 1 * current_normalized_obs[self.node_num+len(self.battery_list)] * float(saved_power)
            reward_for_penalty = 0.0

            for vm_pu_bat in vm_pu_after_control_bat:

                reward_for_penalty += min(0, 100 * (0.05 - abs(1.0 - vm_pu_bat)))

            self.reward_for_power = reward_for_power
            self.reward_for_penalty = reward_for_penalty
            self.saved_money=-1*self._denormalize_state(current_normalized_obs)[self.node_num+len(self.battery_list)]*float(saved_power)

            reward = reward_for_power + reward_for_penalty
        else:
            raise  ValueError("Invalid value for 'state_pattern'. Expected 'default, or define by yourself based on different goal")

        return reward
    def render(self, current_obs, next_obs, reward, finish):
        """
        Render the environment's current state.

        :param current_obs: Current observation.
        :type current_obs: np.array
        :param next_obs: Next observation.
        :type next_obs: np.array
        :param reward: Reward obtained from the last action.
        :type reward: float
        :param finish: Whether the episode has ended.
        :type finish: bool
        """
        print('state={}, next_state={}, reward={:.4f}, terminal={}\n'.format(current_obs, next_obs, reward, finish))





