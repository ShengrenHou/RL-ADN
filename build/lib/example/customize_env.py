'''This script is used to show how to create customize environment based on the template environment
here, we modified the reward function based on the optimal goal: that is maximizing the profit of batteries, and ignore the
penalty of voltage violations'''
import numpy as np
from rl_adn.power_network_rl.environments.Environment_Integrated import PowerNetEnv
env_config={
  "voltage_limits": [0.95, 1.05],
  "algorithm": "Laurent",
  "battery_list": [11,15,26,29,33],
  "year": 2020,
  "month": 1,
  "day": 1,
  "train": True,
  "state_pattern": "default",
  "network_info": "None",
  "time_series_data_path": "../data_sources/time_series_data/34_node_time_series.csv"
}
class ProfitBatteryEnv(PowerNetEnv):
    def __init__(self, env_config:env_config):
        super().__init__(env_config)  # Call the constructor of the parent class

    def _calculate_reward(self, current_normalized_obs: np.ndarray, vm_pu_after_control_bat: np.ndarray, saved_power: float) -> float:
        """
        Your new reward calculation logic goes here.

        Parameters:
            current_normalized_obs (np.ndarray): The current normalized observations.
            vm_pu_after_control_bat (np.ndarray): The voltage after control at battery locations.
            saved_power (float): The amount of power saved.

        Returns:
            float: Calculated reward.
        """

        # Your new logic to calculate the reward
        # For examples, let's say the reward is now twice the saved power
        new_reward = 2 * saved_power

        return new_reward

if __name__ == '__main__':
    profit_battery_env = ProfitBatteryEnv(env_config)
    profit_battery_env.reset()

    for j in range(1):
        episode_reward = 0
        for i in range(1000):
            tem_action = np.ones(len(profit_battery_env.battery_list))
            next_obs, reward, finish, info = profit_battery_env.step(tem_action)
            episode_reward += reward
        print(episode_reward)
