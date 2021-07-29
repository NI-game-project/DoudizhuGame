import os
import timeit
from datetime import datetime
import torch

from envs.mydoudizhu import DoudizhuEnv as Env
from utils.logger import Logger
from utils_global import tournament

from agents.non_rl.rule_based_agent import DouDizhuRuleAgentV1 as RuleAgent
from agents.nfsp_agent import NFSPAgent

which_agent = 'nfsp'
eval_every = 1000
eval_num = 500
episode_num = 50_000

save_dir_landlord = f'./experiments/{which_agent}/landlord/'
save_dir_peasant = f'./experiments/{which_agent}/peasant/'
landlord_logger = Logger(save_dir_landlord)
peasant_logger = Logger(save_dir_peasant)

save_every = 1000

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
state_shape = [9, 4, 15]
env_type = 'cooperation'
train_env = Env(config, state_shape=state_shape, type=env_type)
eval_env_landlord = Env(config, state_shape=state_shape, type=env_type)
eval_env_peasant = Env(config, state_shape=state_shape, type=env_type)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize NFSP agents
agents = []
for i in range(train_env.player_num):
    agents.append(NFSPAgent(
        num_actions=train_env.action_num,
        state_shape=train_env.state_shape,
    ))

# initialize random agent to evaluate against
rule_agent = RuleAgent(action_num=train_env.action_num)
train_env.set_agents(agents)
eval_env_landlord.set_agents([agents[0], rule_agent, rule_agent])
eval_env_peasant.set_agents([rule_agent, agents[1], agents[2]])

start_time = datetime.now().strftime("%H:%M:%S")
print("Start Time =", start_time)
start = timeit.default_timer()

for episode in range(episode_num):

    # set policy regime for NFSP agents
    for agent in agents:
        agent.set_policy()
    # get transitions by playing an episode in env
    trajectories, _ = train_env.run(is_training=True)

    for i in range(train_env.player_num):
        for trajectory in trajectories[i]:
            agents[i].add_transition(trajectory)

    # evaluate against rule based agent with average policy
    if episode % eval_every == 0:
        # set agents policy to average policy
        for i in range(train_env.player_num):
            agents[i].set_policy(policy='average_policy')

        # set rl agents' online network to evaluation mode
        for agent in agents:
            agent.rl_agent.eval_mode()

        print(datetime.now().strftime("\n%H:%M:%S"))
        # evaluating landlord against rule_based agent
        result_ll, states_ll = tournament(eval_env_landlord, eval_num, agents[0])
        landlord_logger.log_performance(episode, result_ll[0], agents[0].loss, states_ll[0][-1][0]['raw_obs'],
                                        agents[0].actions, agents[0].predictions, agents[0].rl_agent.q_values,
                                        agents[0].rl_agent.current_q_values, agents[0].rl_agent.expected_q_values)

        print(datetime.now().strftime("\n%H:%M:%S"))

        # evaluating peasants against rule_based agent
        result_p, states_p = tournament(eval_env_peasant, eval_num, agents[1])
        peasant_logger.log_performance(episode, result_p[1], agents[1].loss, states_p[1][-1][0]['raw_obs'],
                                       agents[1].actions, agents[1].predictions, agents[1].rl_agent.q_values,
                                       agents[1].rl_agent.current_q_values, agents[1].rl_agent.expected_q_values)

        print(datetime.now().strftime("\n%H:%M:%S"))
        print(f'\nepisode: {episode}, peasant result: {result_p}')

        if episode % eval_every == 0:
            agents[0].save_state_dict(os.path.join(save_dir_peasant, f'{which_agent}_landlord_ckpt.pt'))
            agents[1].save_state_dict(os.path.join(save_dir_peasant, f'{which_agent}_downpeasant_ckpt.pt'))
            agents[2].save_state_dict(os.path.join(save_dir_peasant, f'{which_agent}_uppeasant_ckpt.pt'))
            now = datetime.now().strftime("%H:%M:%S")
            print(f'\nckpt save for ep:{episode} at {now}')

end_time = datetime.now().strftime("%H:%M:%S")
print("End Time =", end_time)
stop = timeit.default_timer()
print(f'Training time: {((stop - start) / 3600):.2f} hrs '
      f'for landlord: {agents[0].train_step} steps'
      f'for peasant: {agents[1].train_step} steps')

# Close files in the logger and plot the learning curve
landlord_logger.close_files()
landlord_logger.plot('nfsp.vs.rule')
peasant_logger.close_files()
peasant_logger.plot('rule.vs.nfsp')

# Save model
agents[0].save_state_dict(os.path.join(save_dir_landlord, f'{which_agent}_landlord.pt'))
agents[1].save_state_dict(os.path.join(save_dir_peasant, f'{which_agent}_downpeasant.pt'))
agents[2].save_state_dict(os.path.join(save_dir_peasant, f'{which_agent}_uppeasant.pt'))
