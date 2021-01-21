import os
import timeit
from datetime import datetime
import torch

from agents.random_agent import RandomAgent
from agents.DQN_agent_v5 import DQNAgent
from env.doudizhu import DoudizhuEnv
from utils.logger import Logger
from utils_global import tournament

# Make environments to train and evaluate models
config = {
    'seed': 0,
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

# initialize random agent for evaluation
random_agent = RandomAgent(action_num=eval_env.action_num)

# rule_agent = DouDizhuRuleAgentV1()

# initialize DQN agents
dqn_agents = []
for i in range(env.player_num):
    dqn_agents.append(DQNAgent(
        num_actions=env.action_num,
        state_shape=env.state_shape,
        hidden_layers=[512, 1024, 512],
        lr=.0001,))
env.set_agents(dqn_agents)
eval_env.set_agents([dqn_agents[0], random_agent, random_agent])

eval_every = 1000
eval_num = 1000
episode_num = 200_000

log_dir = './experiments/dqn'
logger = Logger(log_dir)

start_time = datetime.now().strftime("%H:%M:%S")
print("Start Time =", start_time)
start = timeit.default_timer()

for episode in range(episode_num):
    # get transitions by playing an episode in env
    trajectories, _ = env.run(is_training=True)

    for i in range(env.player_num):
        for trajectory in trajectories[i]:
            dqn_agents[i].add_transition(trajectory)
    
    
    # evaluate against random agent
    if episode % eval_every == 0:
        result = tournament(eval_env, eval_num)[0]
        print(f'\nepisode: {episode}, result: {result}')
        logger.log_performance(env.timestep, result)


end_time = datetime.now().strftime("%H:%M:%S")
print("End Time =", end_time)
stop = timeit.default_timer()
print(f'Training time: {(stop - start)/3600} hrs for {episode_num} episodes')

# Close files in the logger and plot the learning curve
logger.close_files()
logger.plot('dqn.vs.random')

# Save model
save_dir = './models/dqn'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
dqn_agents[0].save_state_dict(os.path.join(save_dir, 'dqn_agent_landlord.pt'))
dqn_agents[1].save_state_dict(os.path.join(save_dir, 'dqn_agent_downpeasant.pt'))
dqn_agents[2].save_state_dict(os.path.join(save_dir, 'dqn_agent_uppeasant.pt'))
