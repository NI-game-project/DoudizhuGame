import os
import timeit
from datetime import datetime
import torch

from agents.random_agent import RandomAgent
from agents.drqn_agent import DRQNAgent
from env import doudizhu
from utils.logger import Logger
from utils_global import tournament

# Make environments to train and evaluate models
config = {
    'seed': None,
    'allow_step_back': True,
    'allow_raw_data': True,
    'record_action': True,
    'single_agent_mode': False,
    'active_player': None,
}

# Make environment
env = doudizhu.DoudizhuEnv(config)
eval_env = doudizhu.DoudizhuEnv(config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize random agent for evaluation
random_agent = RandomAgent(action_num=eval_env.action_num)

# rule_agent = DouDizhuRuleAgentV1()

# initialize DQN agents
drqn_agents = []
for i in range(env.player_num):
    drqn_agents.append(DRQNAgent(
        num_actions=env.action_num,
        state_shape=env.state_shape,
        mlp_layers=[512, 1024, 1024, 512],
        lr=.00005,
        update_every=200,
        ))
env.set_agents(drqn_agents)
eval_env.set_agents([drqn_agents[0], random_agent, random_agent])

eval_every = 500
eval_num = 1000
episode_num = 10_000

log_dir = './experiments/drqn/'
logger = Logger(log_dir)

start_time = datetime.now().strftime("%H:%M:%S")
print("Start Time =", start_time)
start = timeit.default_timer()

for episode in range(episode_num):
    # reset hidden state of recurrent agents
    for i in range(env.player_num):
        drqn_agents[i].reset_hidden()
    # get transitions by playing one episode in the env
    trajectories, _ = env.run(is_training=True)

    for i in range(env.player_num):
        drqn_agents[i].add_transition(trajectories[i])

    # evaluate against random agent
    if episode > 10 and episode % eval_every == 0:
        drqn_agents[0].reset_hidden()
        result = tournament(eval_env, eval_num)[0]
        print(f'\nepisode: {episode}, result: {result}')
        logger.log_performance(env.timestep, result)

end_time = datetime.now().strftime("%H:%M:%S")
print("End Time =", end_time)
stop = timeit.default_timer()
print(f'Training time: {(stop - start)/3600} hrs for {episode_num} episodes')

# Close files in the logger and plot the learning curve
logger.close_files()
logger.plot('drqn.vs.random')

# Save model
save_dir = './models/drqn'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
drqn_agents[0].save_state_dict(os.path.join(save_dir, 'drqn_agent_landlord.pt'))
drqn_agents[1].save_state_dict(os.path.join(save_dir, 'drqn_agent_downpeasant.pt'))
drqn_agents[2].save_state_dict(os.path.join(save_dir, 'drqn_agent_uppeasant.pt'))
