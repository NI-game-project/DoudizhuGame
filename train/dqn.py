import os
import timeit
from datetime import datetime
import torch

from agents.random_agent import RandomAgent
from agents.ddqn_agent import DQNAgent
from agents.rule_based_agent import DouDizhuRuleAgentV1
from envs.doudizhu import DoudizhuEnv
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

rule_agent = DouDizhuRuleAgentV1()

# initialize DQN agents
dqn_agents = []

for i in range(env.player_num):
    dqn_agents.append(DQNAgent(num_actions=env.action_num,
                               state_shape=env.state_shape,
                               lr=.000001,
                               use_conv=False,
                               dueling=False,
                               soft_update=True))

env.set_agents(dqn_agents)
eval_env.set_agents([dqn_agents[0], rule_agent, rule_agent])
print(dqn_agents[0].q_net)

eval_every = 500
eval_num = 1000
episode_num = 100_000

log_dir = './experiments/dqn/'
logger = Logger(log_dir)

save_dir = './experiments/dqn/models'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_dir_best = './experiments/dqn/best_models'
if not os.path.exists(save_dir_best):
    os.makedirs(save_dir_best)

start_time = datetime.now().strftime("%H:%M:%S")
print("Start Time =", start_time)
start = timeit.default_timer()
best_result = 0

for episode in range(episode_num):
    # get transitions by playing an episode in envs
    trajectories, _ = env.run(is_training=True)

    for i in range(env.player_num):
        for trajectory in trajectories[i]:
            dqn_agents[i].add_transition(trajectory)

    # evaluate against random agent
    if episode % eval_every == 0:
        result, states = tournament(eval_env, eval_num, dqn_agents[0])
        logger.log_performance(episode, result[0], dqn_agents[0].loss, states[0][-1][0]['raw_obs'],
                               dqn_agents[0].actions, dqn_agents[0].predictions, dqn_agents[0].q_values,
                               dqn_agents[0].current_q_values, dqn_agents[0].expected_q_values)
        print(f'\nepisode: {episode}, result: {result}')
        if result[0] > best_result:
            best_result = result[0]
            dqn_agents[0].save_state_dict(os.path.join(save_dir_best, 'dqn_agent_landlord_best.pt'))
            dqn_agents[1].save_state_dict(os.path.join(save_dir_best, 'dqn_agent_downpeasant_best.pt'))
            dqn_agents[2].save_state_dict(os.path.join(save_dir_best, 'dqn_agent_uppeasant_best.pt'))

end_time = datetime.now().strftime("%H:%M:%S")
print("End Time =", end_time)
stop = timeit.default_timer()
print(f'Training time: {(stop - start) / 3600} hrs for {episode_num} episodes')
print(f'best_result: {best_result}')

# Close files in the logger and plot the learning curve
logger.close_files()
logger.plot('dqn.vs.rule')

# Save model
dqn_agents[0].save_state_dict(os.path.join(save_dir, 'dqn_agent_landlord.pt'))
dqn_agents[1].save_state_dict(os.path.join(save_dir, 'dqn_agent_downpeasant.pt'))
dqn_agents[2].save_state_dict(os.path.join(save_dir, 'dqn_agent_uppeasant.pt'))

