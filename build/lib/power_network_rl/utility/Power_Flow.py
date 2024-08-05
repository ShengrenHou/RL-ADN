# This file is used for Comparison of Results from Laurent Power Flow and Pandapower Flow
from rl_adn.power_network_rl.utility.grid import GridTensor
from rl_adn.power_network_rl.utility.utils import create_pandapower_net
import numpy as np
import os
import pandas as pd
import pandapower as pp
#%% Solve base case (34 node bus)
import time

# what to do:
# 1- There should be a data input in the environment of our algorithm.
# Data can be of two types
        # a. Netowrk data
        # b. Time Series load data
# 2- It should also take the option of using pandapower or laurent power flow.
# 3- It should also take the option of which reinforcement algorithm to use.
# 4- It should use the corresponding algorithm to the training.
# 5- Then it should be tested on the available data to see the performance.
branch_info_file = bus_data_csv_file_path = os.path.join('data','Lines_34.csv')
bus_info_file = bus_data_csv_file_path = os.path.join('data', 'Nodes_34.csv')
branch_info_ = pd.read_csv(branch_info_file, encoding='utf-8')
bus_info_ = pd.read_csv(bus_info_file, encoding='utf-8')
print(bus_info_)
network = GridTensor(node_file_path = os.path.join('data', 'Nodes_34.csv'), lines_file_path=os.path.join('data','Lines_34.csv') )
# print('here is the GridTensor network ')
# print(network)
# testing = np.array([163.089,129.3191,110.8417,58.21588,84.54137,134.9425,133.3686,
#   97.21313,152.6234,107.3716,94.17626,80.66978,92.46544,118.6163,
#  107.5124,105.9869,102.6775,132.5689,127.7359,138.0429,104.8411,
#   94.3515,67.27496,118.2685,92.27614,104.5529,93.57138,109.5921,
#   81.38974,101.1428,100.1258,108.4659,104.575  ])
P_file = np.array([387.09, 0., 387.09, 387.09, 0., 0., 387.09,387.09,
   0., 387.09, 230.571, 121.176, 121.176, 121.176, 22.7205, 387.09,
 387.09, 387.09, 387.09, 387.09, 387.09, 387.09, 387.09, 387.09,
 387.09, 230.571, 126.225, 126.225, 126.225, 95.931, 95.931, 95.931,
  95.931 ])



# chainging =np.array([100.3653,0, 121.8146, 73.76572, 97.30762, 144.9831, 148.392,
#  98.80626, 158.1285, 122.9858, 99.22552, 91.22051, 113.681,
#  133.2627, 119.9355, 131.0784, 136.8616, 144.2292, 134.3264, 153.176,
#  118.446, 112.526, 83.63235, 130.329, 113.4724, 119.3644, 103.4514,
#  123.838, 85.03031, 112.4225, 118.1784, 124.5031, 113.8273 ])
start_time_laurent = time.time()
#FIX IT:see the Q file,
solution = network.run_pf(active_power= P_file)
time_laurent = time.time() - start_time_laurent
print('time_laurent', time_laurent)
Ybus = network._make_y_bus()
dense_array = Ybus.toarray()

print('Here is the voltage result from Laurent Power Flow')
print(solution["v"])
v = solution["v"]
v_34 = np.insert(v,0,1)
v_laurent_magnitude =  abs(v_34)
print('v_laurent',v_laurent_magnitude)
network_info = {'vm_pu': 1.0, 's_base': 1000, 'bus_info_file': bus_info_file, 'branch_info_file': branch_info_file}

net = create_pandapower_net(network_info)
print(net.load)
start_time = time.time()
pp.runpp(net)
time_panda = time.time() - start_time
print('time_panda', time_panda)
v_real = net.res_bus["vm_pu"].values * np.cos(np.deg2rad(net.res_bus["va_degree"].values))
v_img = net.res_bus["vm_pu"].values * np.sin(np.deg2rad(net.res_bus["va_degree"].values))
v_result = v_real + 1j * v_img
# print('Here is the voltage result from Pandapoewr')
# print(v_result)
v_panda_magnitude = abs(v_result)
print('Here is the voltage magnitude from PandaPower',v_panda_magnitude)
# print('here is the external power imported from the grid in case of PandaPower')
# print(net.res_ext_grid['p_mw'] )
# print(type(net.res_ext_grid['p_mw']))

v_diff =   v_laurent_magnitude - v_panda_magnitude
print('Here is the voltage magnitude difference between the Laurent Power Flow and PandaPowerFlow')
print(v_diff)

#
# power_each_node = np.matmul(dense_array, v_34)
# print('here is the node power')
# print(power_each_node)
# print('here is the power imported from the external grid in case of laurent power flow')
# print(power_each_node[0].real)


# plotting the bus from FastPowerFlow and PandaPower





