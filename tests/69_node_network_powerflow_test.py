import os
import time
import numpy as np
import pandas as pd
import pandapower as pp

from rl_adn.utility.grid import GridTensor
from rl_adn.utility.utils import create_pandapower_net


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../rl_adn', 'data_sources'))
config={
            'branch_info_file': os.path.join(root_dir, 'network_data/node_69', 'Lines_69.csv'),
            'bus_info_file': os.path.join(root_dir, 'network_data/node_69', 'Nodes_69.csv'),
            'vm_pu': 1.0,
            's_base': 1000
        }
branch_info = pd.read_csv(config['branch_info_file'], encoding='utf-8')
bus_info = pd.read_csv(config['bus_info_file'], encoding='utf-8')

network = GridTensor(node_file_path=config['bus_info_file'],
                     lines_file_path=config['branch_info_file'])
Ybus = network._make_y_bus().toarray()
P_file = np.array([387.09, 0., 387.09, 387.09, 0., 0., 387.09, 387.09,
                   0., 387.09, 230.571, 121.176, 121.176, 121.176, 22.7205, 387.09,
                   387.09, 387.09, 387.09, 387.09, 387.09, 387.09, 387.09, 387.09,
                   387.09, 0., 387.09, 387.09, 0., 0., 387.09, 387.09,
                   0., 387.09, 230.571, 121.176, 121.176, 121.176, 22.7205, 387.09,
                   387.09, 387.09, 387.09, 387.09, 387.09, 387.09, 387.09, 387.09,
                    387.09, 387.09, 387.09, 387.09, 387.09, 387.09, 387.09, 387.09,
                    387.09, 387.09, 387.09, 387.09, 387.09, 387.09, 387.09, 387.09,
                   387.09, 387.09, 387.09, 387.09,
                   ])
network.Q_file = np.zeros(68)
num_runs = 100




start_time_laurent = time.time()
for i in range(num_runs):
    solution_laurent = network.run_pf(active_power=P_file)
time_laurent = time.time() - start_time_laurent

net = create_pandapower_net(config)

for bus_index in net.load.bus.index:
    if bus_index == 0:
        net.load.p_mw[bus_index] = 0
        net.load.q_mvar[bus_index] = 0
    else:
        net.load.p_mw[bus_index] = P_file[bus_index - 1] / 1000
        net.load.q_mvar[bus_index - 1] = 0

start_time_panda = time.time()
for i in range(num_runs):
    pp.runpp(net)
time_panda = time.time() - start_time_panda

v_laurent = solution_laurent["v"]
v_34_laurent = np.insert(v_laurent, 0, 1)
v_laurent_magnitude = abs(v_34_laurent)

v_real_panda = net.res_bus["vm_pu"].values * np.cos(np.deg2rad(net.res_bus["va_degree"].values))
v_img_panda = net.res_bus["vm_pu"].values * np.sin(np.deg2rad(net.res_bus["va_degree"].values))
v_panda = v_real_panda + 1j * v_img_panda
v_panda_magnitude = abs(v_panda)
time_panda_ms = time_panda * 1000
time_laurent_ms = time_laurent * 1000

time_panda_ns = time_panda * 1e9
time_laurent_ns = time_laurent * 1e9
print('error percentage',np.mean((v_panda_magnitude-v_laurent_magnitude)))
print('time used by pandapower: {:.6f} seconds'.format(time_panda/num_runs))
print('time used by laurent power: {:.6f} seconds'.format(time_laurent/num_runs))
# Print the times
print('time used by pandapower: {:.2f} ms'.format(time_panda_ms/num_runs))
print('time used by laurent power: {:.2f} ms'.format(time_laurent_ms/num_runs))

print('time used by pandapower: {:.2f} ns'.format(time_panda_ns/num_runs))
print('time used by laurent power: {:.2f} ns'.format(time_laurent_ns/num_runs))