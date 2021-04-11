import os
import timeit
from datetime import datetime
import torch

from envs.mydoudizhu import DoudizhuEnv as Env
from utils.logger import Logger
from utils_global import tournament

### random or rule_based agents for evaluating ###
from agents.non_rl.dummyrule_agent import DummyRuleAgent as RandomAgent
from agents.non_rl.rule_based_agent import DouDizhuRuleAgentV1 as RuleAgent
# from agents.non_rl.rhcp_agent import RHCPAgent as RuleAgent

### rl_agents for training ###
### uncomment these lines to import different Agent for training
# from agents.value_based.nstep_agent import NStepDQNAgent as RLAgent
# from agents.value_based.c51_agent import C51DQNAgent as RLAgent
# from agents.value_based.per_agent import PERDQNAgent as RLAgent
from agents.value_based.rainbow_c51 import RainbowAgent as RLAgent

which_run = '1'
which_agent = 'rainbow_c21'
eval_every = 500
eval_num = 1000
episode_num = 100_000

save_dir_landlord = f'./experiments/{which_run}/{which_agent}/landlord/'
save_dir_peasant = f'./experiments/{which_run}/{which_agent}/peasant/'
landlord_logger = Logger(save_dir_landlord)
peasant_logger = Logger(save_dir_peasant)

best_result_ll, best_result_p = 0., 0.

# Make environments to train and evaluate models
config = {
    'seed': None,
    # add key 'use_conv' to config dict, if using mydoudizhu.py as env
    # to indicate whether using state_encoding for fc_net or conv_net.
    'use_conv': False,
    'allow_step_back': True,
    'allow_raw_data': True,
    'record_action': True,
    'single_agent_mode': False,
    'active_player': None,
}
state_shape = [7, 4, 15]
env_type = 'cooperation'
# add param state_shape to the environment
# to indicate which state_shape(simple/complicated, w/o cooperation) is used for training
train_env = Env(config, state_shape=state_shape, type=env_type)

eval_env_landlord = Env(config, state_shape=state_shape, type=env_type)
eval_env_peasant = Env(config, state_shape=state_shape, type=env_type)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize agent for evaluation

rule_agent = RuleAgent(action_num=train_env.action_num)

# initialize rl_agents
agents = []
for i in range(train_env.player_num):
    agents.append(RLAgent(num_actions=train_env.action_num,
                          state_shape=train_env.state_shape,
                          lr=.000005,
                          gamma=0.99,
                          batch_size=32,
                          epsilon_start=1.0,
                          epsilon_end=0.05,
                          epsilon_decay_steps=80000,
                          train_every=1,
                          hard_update_target_every=1000,
                          replay_memory_init_size=1000,
                          replay_memory_size=int(2e5),
                          clip=True,
                          ))

train_env.set_agents(agents)
eval_env_landlord.set_agents([agents[0], rule_agent, rule_agent])
eval_env_peasant.set_agents([rule_agent, agents[1], agents[2]])

print(agents[0].online_net)

start_time = datetime.now().strftime("%H:%M:%S")
print("Start Training: ", start_time)
start = timeit.default_timer()

for episode in range(episode_num + 1):
    # get transitions by playing an episode in envs
    trajectories, _ = train_env.run(is_training=True)
    # train the agent against rule_based agent
    # set the agent's online network to training mode

    for i in range(train_env.player_num):
        agents[i].train_mode()
        for trajectory in trajectories[i]:
            agents[i].add_transition(trajectory)

        # evaluate against random agent
    if episode % eval_every == 0:
        print(datetime.now().strftime("\n%H:%M:%S"))
        # evaluating landlord against rule_based agent
        # Set agents' online network to evaluation mode
        for agent in agents:
            agent.eval_mode()
        result_ll, states_ll = tournament(eval_env_landlord, eval_num, agents[0])
        landlord_logger.log_performance(episode, result_ll[0], agents[0].loss, states_ll[0][-1][0]['raw_obs'],
                                        agents[0].actions, agents[0].predictions, agents[0].q_values,
                                        agents[0].current_q_values, agents[0].expected_q_values)

        print(datetime.now().strftime("\n%H:%M:%S"))
        print(f'step {eval_env_landlord.timestep}\n')
        print(f'train step {agents[0].train_step}\n')
        print(f'\nepisode: {episode}, landlord result: {result_ll}, epsilon: {agents[0].epsilon}, ')

        if result_ll[0] > best_result_ll:
            best_result_ll = result_ll[0]
            agents[0].save_state_dict(os.path.join(save_dir_landlord, f'{which_agent}_landlord_best.pt'))

        # evaluating peasants against rule_based agent
        result_p, states_p = tournament(eval_env_peasant, eval_num, agents[1])
        peasant_logger.log_performance(episode, result_p[1], agents[1].loss, states_p[1][-1][0]['raw_obs'],
                                       agents[1].actions, agents[1].predictions, agents[1].q_values,
                                       agents[1].current_q_values, agents[1].expected_q_values)

        print(datetime.now().strftime("\n%H:%M:%S"))
        print(f'step {eval_env_peasant.timestep}\n')
        print(f'train step {agents[1].train_step}\n')
        print(f'\nepisode: {episode}, peasant result: {result_p}, epsilon: {agents[1].epsilon}, ')

        if result_p[1] > best_result_p:
            best_result_p = result_p[1]
            agents[1].save_state_dict(os.path.join(save_dir_peasant, f'{which_agent}_downpeasant_best.pt'))
            agents[2].save_state_dict(os.path.join(save_dir_peasant, f'{which_agent}_uppeasant_best.pt'))

end_time = datetime.now().strftime("%H:%M:%S")
print("End Training: ", end_time)
stop = timeit.default_timer()
print(f'Training time: {((stop - start) / 3600):.2f} hrs '
      f'for landlord: {agents[0].train_step} steps'
      f'for peasant: {agents[1].train_step} steps')

print(f'landlord_best_result: {best_result_ll}')
print(f'peasant_best_result: {best_result_p}')

# Close files in the logger and plot the learning curve
landlord_logger.close_files()
landlord_logger.plot('dqn.vs.rule')
peasant_logger.close_files()
peasant_logger.plot('rule.vs.dqn')

# Save model
agents[0].save_state_dict(os.path.join(save_dir_landlord, f'{which_agent}_landlord.pt'))
agents[1].save_state_dict(os.path.join(save_dir_peasant, f'{which_agent}_downpeasant.pt'))
agents[2].save_state_dict(os.path.join(save_dir_peasant, f'{which_agent}_uppeasant.pt'))
