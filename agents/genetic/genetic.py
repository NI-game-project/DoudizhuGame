import numpy as np
import random
import os

import agents.non_rl.doudizhu_rule_models as rule_based_agent
import envs.mydoudizhu
import utils.logger_hypernetwork
from utils_global import set_global_seed, tournament
import agents.policy_based.a2c


class Genetic_Algorithm():

    def __init__(self, population_size=32, elite_workers_num = 8, evaluation_num = 50, generations = 300, epsilon = 0.001):
        
        self.config = {  'allow_step_back':True,
        'allow_raw_data': True, 
        'record_action': True,
        'seed': 42,
        'single_agent_mode': False,
        'active_player': True}

        self.model_load_path = 'DouDizhu/models/genetic/d2_0.489.h5'
        self.model_save_path = 'DouDizhu/models/genetic/d3_'
        self.model_save_path_dir = 'DouDizhu/models/genetic'
        self.log_dir = 'DouDizhu/experiments/genetic/d3'

        if not os.path.exists(self.model_save_path_dir):
            os.makedirs(self.model_save_path_dir)

        self.training = True
        self.random = True
        self.type = 'a2c'
        self.train_episode_num = 5
        self.num_threads = 4
        self.workers = []
        self.load_data = False

        self.population_size = population_size
        self.elite_workers_num = elite_workers_num
        self.evaluate_num = evaluation_num
        self.generations = generations
        self.epsilon = epsilon
        
        self.logger = envs.logger.Logger(self.log_dir)      
        self.env = envs.doudizhu.DoudizhuEnv(self.config)
        
        #self.agent = agents.ddqn.DQNAgent(action_num=self.env.action_num)
        self.agent = agents.a2c.Actor_Critic(action_num=self.env.action_num)
        self.random_agent = agents.random_agent.RandomAgent(action_num=self.env.action_num)
        self.rule_based_agent = agents.doudizhu_rule_models.DouDizhuRuleAgentV1()

        #self.weight_space = 652985 # This is for DQN
        self.weight_space = 494_081 + 652_085 #225_073 + 226_358 # This is for the A2C

    def set_parameters_dqn(self, worker):
        
        for k, x in enumerate(worker):
            np.random.seed(x)
            if k == 0:
                weights = self.core + np.random.normal(0,1,size=self.weight_space)
            else:
                weights += self.epsilon * np.random.normal(0,1,size=self.weight_space)

        last_used = 0

        for i in range(len(self.agent.q_estimator.layers)):

            if 'dense' in self.agent.q_estimator.layers[i].name:
                weights_shape = self.agent.q_estimator.layers[i].kernel.shape
                no_of_weights = tf.reduce_prod(weights_shape)
                new_weights = tf.reshape(weights[last_used:last_used+no_of_weights], weights_shape) 
                last_used += no_of_weights
                
                weights_shape_bias = self.agent.q_estimator.layers[i].bias.shape
                no_of_weights_bias = tf.reduce_prod(weights_shape_bias)
                new_weights_bias = tf.reshape(weights[last_used:last_used+no_of_weights_bias], weights_shape_bias) 
                
                self.agent.q_estimator.layer[i].set_weigths([new_weights, new_weights_bias])
                last_used += no_of_weights_bias

        self.agent.target_estimator.set_weights(self.agent.q_estimator.get_weights())

    def set_parameters_a2c(self, worker):

        for k, x in enumerate(worker):
            np.random.seed(x)
            if k == 0:
                weights = self.core + np.random.normal(0,1,size=self.weight_space)
            else:
                weights += self.epsilon * np.random.normal(0,1,size=self.weight_space)
    
        last_used = 0
                
        for i in range(len(self.agent.actor.layers)):

            if 'dense' in self.agent.actor.layers[i].name:
                weights_shape = self.agent.actor.layers[i].kernel.shape
                no_of_weights = tf.reduce_prod(weights_shape)
                new_weights = tf.reshape(weights[last_used:last_used+no_of_weights], weights_shape) 
                last_used += no_of_weights
                
                weights_shape_bias = self.agent.actor.layers[i].bias.shape
                no_of_weights_bias = tf.reduce_prod(weights_shape_bias)
                new_weights_bias = tf.reshape(weights[last_used:last_used+no_of_weights_bias], weights_shape_bias) 
                
                self.agent.actor.layers[i].set_weights([new_weights,new_weights_bias])
                last_used += no_of_weights_bias

        for i in range(len(self.agent.critic.layers)):

            if 'dense' in self.agent.critic.layers[i].name:
                weights_shape = self.agent.critic.layers[i].kernel.shape
                no_of_weights = tf.reduce_prod(weights_shape)
                new_weights = tf.reshape(weights[last_used:last_used+no_of_weights], weights_shape) 
                last_used += no_of_weights
                
                weights_shape_bias = self.agent.critic.layers[i].bias.shape
                no_of_weights_bias = tf.reduce_prod(weights_shape_bias)
                new_weights_bias = tf.reshape(weights[last_used:last_used+no_of_weights_bias], weights_shape_bias) 
                
                self.agent.critic.layers[i].set_weights([new_weights,new_weights_bias])
                last_used += no_of_weights_bias
        
    def initial_population(self):

        # These are hyperparameters and can be changed
        fan_in = 6400*4
        np.random.seed(42)
        self.core = np.random.uniform(low=-np.sqrt(6/fan_in), high=np.sqrt(6/fan_in), size=self.weight_space)

        for _ in range(self.population_size):

            # These are hyperparameters and can be changed
            z = random.randint(0,1_000_000)
            self.workers.append([z])
        
    def evaluate_population(self):

        elite_workers = []
        rewards = []

        for worker in self.workers:
            
            if self.type == 'dqn':
                self.set_parameters_dqn(worker)
            else: 
                self.set_parameters_a2c(worker)

            if self.random == True:
                self.env.set_agents([self.agent, self.random_agent, self.random_agent])
            else:
                self.env.set_agents([self.agent, self.rule_based_agent, self.rule_based_agent])

            if self.training == True:
                for _ in range(self.train_episode_num):
                    trajectories, _ = self.env.run(is_training=True)

                for ts in trajectories[0]:
                    self.agent.feed(ts)

            payoff = tournament(self.env, self.evaluate_num)[0]
            rewards.append(payoff)
        
        rewards = np.array(rewards)

        elite_idx = np.argsort(rewards)[self.population_size-self.elite_workers_num:]

        for idx in elite_idx:
            elite_workers.append(self.workers[idx])

        return elite_workers, rewards

    def mutate_population(self, elite_workers):

        self.workers = []
        self.workers.append(elite_workers[-1])
        self.elite = elite_workers[-1]

        for _ in range(self.population_size - 1):
            
            idx = random.randint(0,self.elite_workers_num-1)

            # This is also a hyperparameter and can be changed
            seed = random.randint(0,1_000_000)
            new_worker = elite_workers[idx].copy()
            new_worker.append(seed)
            self.workers.append(new_worker) 
        
    def run(self):

        random.seed(42)
        self.initial_population()
        max_reward = 0
        
        for i in range(self.generations):
            
            elite_workers, rewards = self.evaluate_population()

            self.mutate_population(elite_workers)

            print('these are the scores', rewards, 'and this is the generation', i)
            
            self.logger.log_performance(i, rewards.mean(), self.agent.history_actor, self.agent.history_critic, self.agent.optimizer._decayed_lr(tf.float32).numpy(), self.agent.actions, self.agent.predictions)

            
            if i % 1 == 0:

                max_reward = rewards.mean()
                path = '{}{:.3f}.txt'.format(self.model_save_path,max_reward)
                with open(path, 'w') as data:
                    for x in elite_workers:
                        data.write('{}\n'.format(x))
                data.close()

                
                path_a = '{}_actor_{:.3f}.h5'.format(self.model_save_path,max_reward)
                path_c = '{}_critic_{:.3f}.h5'.format(self.model_save_path,max_reward)


                if self.type == 'a2c':
                    self.agent.critic.save(path_a)
                    self.agent.actor.save(path_c)
                else:
                    self.agent.q_estimator.save(path)
                
            if i == 30:
                self.random = False
            
            elite_workers, rewards = [],[]

