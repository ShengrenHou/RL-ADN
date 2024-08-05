from rl_adn.environments.env import PowerNetEnv,env_config
import time
import torch
# change to your own configuration path
from rl_adn.DRL_algorithms.Agent import AgentDDPG
from rl_adn.DRL_algorithms.utility import Config, ReplayBuffer, get_episode_return



env = PowerNetEnv(env_config)
env_args = {
    'env_name': 'PowerNetEnv',  # Apply torque on the free end to swing a pendulum into an upright position
    'state_dim': env.state_space.shape[0],  # the x-y coordinates of the pendulum's free end and its angular velocity.
    'action_dim': env.action_space.shape[0],  # the torque applied to free end of the pendulum
    'if_discrete': False  # continuous action space, symbols → direction, value → force
}  # env_args = get_gym_env_args(env=gym.make('CartPole-v0'), if_print=True
args = Config(agent_class=AgentDDPG, env_class=None, env_args=env_args)  # see `Config` for explanation
args.run_name='DDPG_test'
'''init buffer configuration'''
args.gamma = 0.99  # discount factor of future rewards
args.target_step=1000
args.warm_up=2000
args.if_use_per = False
args.per_alpha = 0.6
args.per_beta = 0.4
args.buffer_size = int(4e5)
args.repeat_times = 1
args.batch_size=512
'''init device'''
GPU_ID=0
args.gpu_id = GPU_ID
args.num_workers = 4
args.random_seed=521
'''init agent configration'''
args.net_dims=(256,256,256)
args.learning_rate=6e-5
args.num_episode=10
'''init before training'''
args.init_before_training()
'''print configuration'''
args.print()

'''init agent'''
agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args)
'''init buffer '''
if args.if_off_policy:
    buffer = ReplayBuffer(
        gpu_id=args.gpu_id,
        num_seqs=args.num_envs,
        max_size=args.buffer_size,
        state_dim=args.state_dim,
        action_dim=1 if args.if_discrete else args.action_dim,
        if_use_per=args.if_use_per,
        args=args,
    )
    buffer_items = agent.explore_env(env, args.target_step, if_random=True)
    buffer.update(buffer_items)  # warm up for ReplayBuffer

'''train loop'''
start_time_laurent = time.time()

if args.train:
    collect_data = True
    while collect_data:
        print(f'buffer:{buffer.cur_size}')
        with torch.no_grad():
            buffer_items = agent.explore_env(env, args.target_step, if_random=True)
            buffer.update(buffer_items)
        if buffer.cur_size >= args.warm_up:
            collect_data = False
    torch.set_grad_enabled(False)
    for i_episode in range(args.num_episode):

        torch.set_grad_enabled(True)
        critic_loss, actor_loss, = agent.update_net(buffer)
        torch.set_grad_enabled(False)
        episode_reward, violation_time, violation_value, reward_for_power, reward_for_good_action, reward_for_penalty, state_list = get_episode_return(
            env, agent.act,
            agent.device)
        print(
            f'curren epsiode is {i_episode}, reward:{episode_reward},violation time of one day for all nodes:{violation_time},violation value is {violation_value},buffer_length: {buffer.cur_size}')
        if i_episode % 1 == 0:
            buffer_items = agent.explore_env(env, args.target_step, if_random=False)
            buffer.update(buffer_items)
time_laurent = time.time() - start_time_laurent
print('time laurent for DDPG is ', time_laurent)
agent.save_or_load_agent(args.cwd, if_save=True)
buffer.save_or_load_history(args.cwd, if_save=True)
print('actor and critic parameters have been saved')
print('training finished')
