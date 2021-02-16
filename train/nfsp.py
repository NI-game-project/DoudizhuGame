import os
import timeit
from datetime import datetime
import torch

from agents.random_agent import RandomAgent
from env.doudizhu import DoudizhuEnv
from utils.logger import Logger
from utils_global import tournament
from agents.nfsp_agent import NFSPAgent

eval_every = 500
eval_num = 1000
episode_num = 500000
save_every = 10000

# config dictionary
config = {
    'seed': None,
    'allow_step_back': True,
    'allow_raw_data': True,
    'record_action': True,
    'single_agent_mode': False,
    'active_player': None,
}

# Make environment
env = DoudizhuEnv(config)
eval_env = DoudizhuEnv(config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize NFSP agents
nfsp_agents = []
for i in range(env.player_num):
    nfsp_agents.append(NFSPAgent(
        scope='nfsp' + str(i),
        num_actions=env.action_num,
        state_shape=env.state_shape,
        rl_hidden_layers=[1024, 512, 1024, 512],
        sl_hidden_layers=[1024, 512, 1024, 512],
        sl_lr=0.005,
        rl_lr=0.1))

# initialize random agent to evaluate against
random_agent = RandomAgent(action_num=eval_env.action_num)
env.set_agents(nfsp_agents)
eval_env.set_agents([nfsp_agents[0], random_agent, random_agent])

log_dir = './experiments/nfsp/'
logger = Logger(log_dir)

# directory for saving the agents
save_dir = './models/nfsp'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

start_time = datetime.now().strftime("%H:%M:%S")
print("Start Time =", start_time)
start = timeit.default_timer()

for episode in range(episode_num):

    # set policy regime for NFSP agents
    for agent in nfsp_agents:
        agent.set_policy()
    # get transitions by playing an episode in env
    trajectories, _ = env.run(is_training=True)

    for i in range(env.player_num):
        for trajectory in trajectories[i]:
            nfsp_agents[i].add_transition(trajectory)

    # evaluate against random agent with average policy
    if episode % eval_every == 0:
        nfsp_agents[0].set_policy('average policy')
        result_landlord = tournament(eval_env, eval_num)[0]
        print(f'\nepisode: {episode}, result for landlord: {result_landlord}')
        logger.log_performance(env.timestep, result_landlord)
       
        if episode % save_every == 0:
            nfsp_agents[0].save_state_dict(os.path.join(save_dir, 'nfsp_agent_landlord.pt'))
            nfsp_agents[1].save_state_dict(os.path.join(save_dir, 'nfsp_agent_downpeasant.pt'))
            nfsp_agents[2].save_state_dict(os.path.join(save_dir, 'nfsp_agent_uppeasant.pt'))

end_time = datetime.now().strftime("%H:%M:%S")
print("End Time =", end_time)

stop = timeit.default_timer()
print(f'Training time: {(stop - start)/3600} hrs {episode_num} eps')

# Close files in the logger
logger.close_files()
# Plot the learning curve
logger.plot('nfsp.vs.random')

# Save model
nfsp_agents[0].save_state_dict(os.path.join(save_dir, 'nfsp_agent_landlord.pt'))
nfsp_agents[1].save_state_dict(os.path.join(save_dir, 'nfsp_agent_downpeasant.pt'))
nfsp_agents[2].save_state_dict(os.path.join(save_dir, 'nfsp_agent_uppeasant.pt'))
