import os
import timeit
from datetime import datetime
import torch

from envs.mydoudizhu import DoudizhuEnv as Env
from utils.logger import Logger
from utils_global import tournament

from agents.non_rl.rule_based_agent import DouDizhuRuleAgentV1 as RuleAgent
from agents.nfsp_agent import NFSPAgent

which_env = '7c'
which_agent = 'nfsp'
eval_every = 1000
eval_num = 1000
episode_num = 50_000

save_dir_landlord = f'./experiments/{which_env}/landlord/{which_agent}/'
save_dir_peasant = f'./experiments/{which_env}/peasant/{which_agent}/'
landlort_logger = Logger(save_dir_landlord)
peasant_logger = Logger(save_dir_peasant)

best_result_ll, best_result_p = 0, 0

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
type = 'cooperation'
env = Env(config, state_shape=state_shape, type=type)
eval_env_landlord = Env(config, state_shape=state_shape, type=type)
eval_env_peasant = Env(config, state_shape=state_shape, type=type)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize NFSP agents
agents = []
for i in range(env.player_num):
    agents.append(NFSPAgent(
        num_actions=env.action_num,
        state_shape=env.state_shape,
        ))

# initialize random agent to evaluate against
rule_agent = RuleAgent(action_num=env.action_num)
env.set_agents(agents)
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
    trajectories, _ = env.run(is_training=True)

    for i in range(env.player_num):
        for trajectory in trajectories[i]:
            agents[i].add_transition(trajectory)

    # evaluate against random agent with average policy
    if episode % eval_every == 0:
        agents[0].set_policy('average policy')
        agents[1].set_policy('average policy')
        agents[2].set_policy('average policy')
        print(datetime.now().strftime("\n%H:%M:%S"))
        # Set agent's online network to evaluation mode
        result_ll, states_ll = tournament(eval_env_landlord, eval_num, agents[0])
        landlort_logger.log_performance(episode, result_ll[0], agents[0].loss, states_ll[0][-1][0]['raw_obs'],
                                        agents[0].actions, agents[0].predictions)

        print(datetime.now().strftime("\n%H:%M:%S"))
        print(f'\nepisode: {episode}, landlord result: {result_ll}, ')

        if result_ll[0] > best_result_ll:
            best_result = result_ll[0]
            agents[0].save_state_dict(os.path.join(save_dir_landlord, f'{which_agent}_landlord_best.pt'))

        result_p, states_p = tournament(eval_env_peasant, eval_num, agents[1])
        landlort_logger.log_performance(episode, result_p[1], agents[1].loss, states_ll[1][-1][0]['raw_obs'],
                                        agents[1].actions, agents[1].predictions)

        print(datetime.now().strftime("\n%H:%M:%S"))
        print(f'\nepisode: {episode}, landlord result: {result_p}')

        if result_p[1] > best_result_ll:
            best_result = result_p[1]
            agents[1].save_state_dict(os.path.join(save_dir_landlord, f'{which_agent}_downpeasant_best.pt'))
            agents[2].save_state_dict(os.path.join(save_dir_landlord, f'{which_agent}_uppeasant_best.pt'))

end_time = datetime.now().strftime("%H:%M:%S")
print("End Time =", end_time)
stop = timeit.default_timer()
print(f'Training time: {(stop - start) / 3600} hrs for {episode_num} episodes')
print(f'peasant_best_result: {best_result_p}')
print(f'landlord_best_result: {best_result_ll}')

# Close files in the logger and plot the learning curve
landlort_logger.close_files()
landlort_logger.plot('nfsp.vs.rule')
peasant_logger.close_files()
peasant_logger.plot('rule.vs.nfsp')

# Save model
agents[0].save_state_dict(os.path.join(save_dir_landlord, f'{which_agent}_landlord.pt'))
agents[1].save_state_dict(os.path.join(save_dir_peasant, f'{which_agent}_downpeasant.pt'))
agents[2].save_state_dict(os.path.join(save_dir_peasant, f'{which_agent}_uppeasant.pt'))