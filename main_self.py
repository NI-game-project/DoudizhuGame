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
#from agents.value_based.n_step_dqn import NStepDQNAgent as RLAgent
#from agents.value_based.per_agent import PERDQNAgent as RLAgent
#from agents.value_based.rainbow_c51 import RainbowAgent as RLAgent
#from agents.value_based.rainbow_qr import RainbowAgent as RLAgent
from agents.value_based.rainbow_mog import RainbowAgent as RLAgent

which_agent = 'rainbow_wo_dist'
eval_every = 500
eval_num = 1000
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
# add param state_shape to the environment
# to indicate which state_shape(simple/complicated, w/o cooperation) is used for training
train_env = Env(config, state_shape=state_shape, type=env_type)
eval_env_landlord = Env(config, state_shape=state_shape, type=env_type)
eval_env_peasant = Env(config, state_shape=state_shape, type=env_type)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize agent for evaluation

rule_agent = RuleAgent(action_num=train_env.action_num)

# initialize rl_agents
agent = RLAgent(num_actions=train_env.action_num,
                state_shape=train_env.state_shape,
                )
train_env.set_agents([agent, agent, agent])
eval_env_landlord.set_agents([agent, rule_agent, rule_agent])
eval_env_peasant.set_agents([rule_agent, agent, agent])

print(agent.online_net)

start_time = datetime.now().strftime("%H:%M:%S")
print("Start Training: ", start_time)
start = timeit.default_timer()

for episode in range(episode_num + 1):
    # get transitions by playing an episode in envs
    trajectories, _ = train_env.run(is_training=True)
    # train the agent against rule_based agent
    # set the agent's online network to training mode
    agent.train_mode()
    for i in range(3):
        for trajectory in trajectories[i]:
            agent.add_transition(trajectory)

        # evaluate against random agent
    if episode % eval_every == 0:
        print(datetime.now().strftime("\n%H:%M:%S"))
        # evaluating landlord against rule_based agent
        # Set agents' online network to evaluation mode
        agent.eval_mode()
        result_ll, states_ll = tournament(eval_env_landlord, eval_num, agent)
        landlord_logger.log_performance(episode, result_ll[0], agent.loss, states_ll[0][-1][0]['raw_obs'])

        print(datetime.now().strftime("\n%H:%M:%S"))
        print(f'step {eval_env_landlord.timestep}\n')
        print(f'train step {agent.train_step}\n')
        print(f'\nepisode: {episode}, landlord result: {result_ll}')

        # evaluating peasants against rule_based agent
        result_p, states_p = tournament(eval_env_peasant, eval_num, agent)
        peasant_logger.log_performance(episode, result_p[1], agent.loss, states_p[1][-1][0]['raw_obs'])

        print(datetime.now().strftime("\n%H:%M:%S"))
        print(f'step {eval_env_peasant.timestep}\n')
        print(f'train step {agent.train_step}\n')
        print(f'\nepisode: {episode}, peasant result: {result_p}')

    if episode % save_every == 0:
        agent.save_state_dict(os.path.join(save_dir_landlord, f'ckpt.pt'))
        now = datetime.now().strftime("%H:%M:%S")
        print(f'\nckpt save for ep:{episode} at {now}')

end_time = datetime.now().strftime("%H:%M:%S")
print("End Training: ", end_time)
stop = timeit.default_timer()
print(f'Training time: {((stop - start) / 3600):.2f} hrs '
      f'for landlord: {agent.train_step} steps'
      f'for peasant: {agent.train_step} steps')

# Close files in the logger and plot the learning curve
landlord_logger.close_files()
landlord_logger.plot('dqn.vs.rule')
peasant_logger.close_files()
peasant_logger.plot('rule.vs.dqn')

# Save final model
agent.save_state_dict(os.path.join(save_dir_landlord, f'{which_agent}.pt'))
