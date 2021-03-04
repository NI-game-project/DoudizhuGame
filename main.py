import os
import timeit
from datetime import datetime
import torch

# from envs.doudizhu_rlcard import DoudizhuEnv as Env
from envs.mydoudizhu import DoudizhuEnv as Env
from utils.logger import Logger
from utils_global import tournament

### random or rule_based agents for evaluating ###
# from agents.non_rl.random_agent import RandomAgent
# from agents.non_rl.dummyrule_agent import DummyRuleAgent as RandomAgent
from agents.non_rl.rule_based_agent import DouDizhuRuleAgentV1 as RuleAgent
# from agents.non_rl.rhcp_agent import RHCPAgent as RuleAgent

### rl_agents for training ###
### uncomment these lines to import different Agent for training
# from agents.value_based.per_dqn_agent import PERDQNAgent as RLAgent
# from agents.value_based.duel_dqn_agent import DQNAgent as RLAgent
# from agents.value_based.C51_dqn_agent import C51DQNAgent as RLAgent
# from agents.value_based.n_step_dqn_agent import NStepDQNAgent as RLAgent
# from agents.value_based.noisy_dqn_agent import NoisyDQNAgent as RLAgent
from agents.value_based.rainbow_agent import RainbowAgent as RLAgent

test_name = 'ddqn'
eval_every = 400
eval_num = 1000
episode_num = 40_000

save_dir = f'./experiments/{test_name}/'
logger = Logger(save_dir)

best_result = 0

# Make environments to train and evaluate models
config = {
    'seed': 0,
    # add key 'use_conv' to config dict, if using mydoudizhu.py as env
    # to indicate whether using state_encoding for fc_net or conv_net.
    'use_conv': False,
    'allow_step_back': True,
    'allow_raw_data': True,
    'record_action': True,
    'single_agent_mode': False,
    'active_player': None,
}

env = Env(config)
eval_env = Env(config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize agent for evaluation

rule_agent = RuleAgent(action_num=eval_env.action_num)

# initialize rl_agents
agents = []
for i in range(env.player_num):
    agents.append(RLAgent(num_actions=env.action_num,
                          state_shape=env.state_shape,
                          lr=.00005,
                          soft_update=False,
                          use_conv=True
                          ))

env.set_agents(agents)
eval_env.set_agents([agents[0], rule_agent, rule_agent])
print(agents[0].local_net)

start_time = datetime.now().strftime("%H:%M:%S")
print("Start Time =", start_time)
start = timeit.default_timer()

for episode in range(episode_num):
    # get transitions by playing an episode in envs
    trajectories, _ = env.run(is_training=True)
    # train the agent in self_play mode
    for i in range(env.player_num):
        for trajectory in trajectories[i]:
            agents[i].add_transition(trajectory)

    # evaluate against random agent
    if episode % eval_every == 0:
        result, states = tournament(eval_env, eval_num, agents[0])
        logger.log_performance(episode, result[0], agents[0].loss, states[0][-1][0]['raw_obs'],
                               agents[0].actions, agents[0].predictions, agents[0].q_values,
                               agents[0].current_q_values, agents[0].expected_q_values)
        print(f'\nepisode: {episode}, result: {result}, epsilon: {agents[0].epsilon}')
        if result[0] > best_result:
            best_result = result[0]
            agents[0].save_state_dict(os.path.join(save_dir, f'{test_name}_agent_landlord_best.pt'))
            agents[1].save_state_dict(os.path.join(save_dir, f'{test_name}_agent_downpeasant_best.pt'))
            agents[2].save_state_dict(os.path.join(save_dir, f'{test_name}_agent_uppeasant_best.pt'))

end_time = datetime.now().strftime("%H:%M:%S")
print("End Time =", end_time)
stop = timeit.default_timer()
print(f'Training time: {(stop - start) / 3600} hrs for {episode_num} episodes')
print(f'best_result: {best_result}')

# Close files in the logger and plot the learning curve
logger.close_files()
logger.plot('dqn.vs.rule')

# Save model
agents[0].save_state_dict(os.path.join(save_dir, f'{test_name}_agent_landlord.pt'))
agents[1].save_state_dict(os.path.join(save_dir, f'{test_name}_agent_downpeasant.pt'))
agents[2].save_state_dict(os.path.join(save_dir, f'{test_name}_agent_uppeasant.pt'))
