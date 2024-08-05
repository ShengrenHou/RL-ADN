# RL-ADN: A Benchmark Framework for DRL-based Battery Energy Arbitrage in Distribution Networks



**RL-ADN** is **the first open-source framework** designed to solve the optimal ESSs dispatch in active distribution networks. RL-ADN offers unparalleled flexibility in modeling distribution networks, and ESSs, accommodating a wide range of research goals. A standout feature of RL-ADN is its data augmentation module, based on Gaussian Mixture Model and Copula (GMC) functions, which elevates the performance ceiling of DRL agents. Additionally, RL-ADN incorporates the Laurent power flow solver, significantly reducing the computational burden of power flow calculations during training without sacrificing accuracy. The effectiveness of RL-ADN is demonstrated in different sizes of distribution network environments, showing marked performance improvements in the adaptability of DRL algorithms for ESS dispatch tasks. This enhancement is particularly beneficial from the increased diversity of training scenarios. Furthermore, RL-ADN achieves a tenfold increase in computational efficiency during training, making it highly suitable for large-scale network applications. The library sets a new benchmark in DRL-based ESSs dispatch in distribution networks and it is poised to advance DRL applications in distribution network operations significantly. 

## Features

- **Versatile Benchmarking**: Model diverse energy arbitrage tasks with full flexibility.
- **Laurent Power Flow**: Over 10 times faster computational speed compared to traditional methods.
- **Seamless Transition**: Designed for both simulated environments and real-world applications.
- **Open-source**: Easily accessible for modifications, customizations, and further research.


## Outline

  - [Overview](#overview)
  - [File Structure](#file-structure)
  - [Installation](#installation)
  - [Tutorials](#tutorials)
  - [Publications](#publications)
  - [Citing RL-ADN](#citing-RL-ADN)
  - [LICENSE](#license)


## File Structure

The main folder **RL-ADN** is shown below

```
└─power_network_rl
    │  README.md
    │  requirements.txt
    │  setup.py
    │  __init__.py
    │
    ├─benckmark_algorithms
    │      Optimality_pyomo.py
    │      __init__.py
    │
    ├─data_manager
    │  │  data_manager.py
    │  │  __init__.py
    │
    ├─data_sources
    │  ├─network_data
    │  │  │  __init__.py
    │  │  │
    │  │  ├─node_123
    │  │  │      Lines_123.csv
    │  │  │      Nodes_123.csv
    │  │  │
    │  │  ├─node_25
    │  │  │      Lines_25.csv
    │  │  │      Nodes_25.csv
    │  │  │
    │  │  ├─node_34
    │  │  │      Lines_34.csv
    │  │  │      Nodes_34.csv
    │  │  │
    │  │  └─node_69
    │  │          Lines_69.csv
    │  │          Nodes_69.csv
    │  │
    │  └─time_series_data
    │          123_node_time_series.csv
    │          25_node_time_series.csv
    │          34_node_time_series.csv
    │          69_node_time_series.csv
    │
    ├─docs
    ├─DRL_algorithms
    │  │  Agent.py
    │  │  DDPG.py
    │  │  PPO.py
    │  │  SAC.py
    │  │  TD3.py
    │  │  utility.py
    │  │  __init__.py
    │
    │
    ├─environments
    │  │  battery.py
    │  │  env.py
    │  │  __init__.py
    │
    ├─example
    │      customize_env.py
    │      training_DDPG.py
    │
    ├─tests
    │      123_node_network_powerflow_test.py
    │      25_node_network_powerflow_test.py
    │      69_node_network_powerflow_test.py
    │      test_comparison_power_flow.py
    │
    ├─utility
    │  │  gpu_interface.py
    │  │  grid.py
    │  │  Not_converge_Power_Flow.py
    │  │  numbarize.py
    │  │  Power_Flow.py
    │  │  utils.py
    │  │  __init__.py
    │  │



```

## Installation
To install RL-ADN, simply run:

```
pip install RL-ADN
```
Or install from git
```
git clone https://github.com/shengrenhou/RL-ADN.git
cd your repository
pip install -e .

```
Or install from local, if you have download the source code

```
cd to the path contains setup.py
python setup.py install

```


## Status Update
<details><summary><b>Version History</b> <i>[click to expand]</i></summary>
<div>
 *2023-10-05
  	0.1.3: add examples and tutorials
 *2023-09-27
  	0.1: Beta version
</div>
</details>

## Tutorials
In the example folder:
- `Tutorial_DDPG_training_using_RL_ADN.ipynb` shows a tutorial for training DDPG agents using RL-ADN step by step. 
- `Customize_env.ipynb` shows a simple tutorial for users to customize their environment by using RL-ADN



## Citing RL-ADN
Preparing the manuscript

## Contributing
Not decided yet 



## LICENSE

MIT License

**Disclaimer: We are sharing codes for academic purposes under the MIT education license. Nothing herein is financial advice, and NOT a recommendation to trade real money. Please use common sense and always first consult a professional before trading or investing.**
