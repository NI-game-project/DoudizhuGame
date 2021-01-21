import numpy as np
import torch

from agents.DQN_agent import Memory
from agents.networks import CategoricalDQN
from utils_global import remove_illegal

# this might have some bugs, need to check it again 


class CategoricalDQNAgent():
    def __init__(self,
                 state_shape=None,
                 num_actions=309,
                 num_atoms=51,
                 Vmin=-10,
                 Vmax=10,
                 replay_memory_init_size=100,
                 replay_memory_size=int(1e5),
                 discount_factor=0.99,
                 epsilons=None,
                 epsilon_decay_steps=20000,
                 learning_rate=0.0001,
                 update_every=1000,
                 train_every=1,
                 batch_size=64,
                 mlp_layers=None,
                 device=None):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.discount_factor = discount_factor
        self.update_every = update_every
        self.train_every = train_every
        self.replay_memory_init_size = replay_memory_init_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.mlp_layers = mlp_layers
        self.epsilon_decay_steps = epsilon_decay_steps
        self.use_raw = False

        self.delta_z = (self.Vmax - self.Vmin) / (self.num_atoms - 1)
        self.epsilons = np.linspace(1.0, 0.1, epsilon_decay_steps)

        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.memory = Memory(replay_memory_size, batch_size)

        self.q_net = CategoricalDQN(state_shape=self.state_shape,
                                    num_actions=self.num_actions,
                                    mlp_layers=self.mlp_layers,
                                    num_atoms=self.num_atoms,
                                    Vmin=self.Vmin, Vmax=self.Vmax).to(device)
        self.target_net = CategoricalDQN(state_shape=state_shape,
                                         num_actions=num_actions,
                                         mlp_layers=self.mlp_layers,
                                         num_atoms=self.num_atoms,
                                         Vmin=self.Vmin, Vmax=self.Vmax).to(device)

        self.q_net.eval()
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
        self.timestep = 0
        self.softmax = torch.nn.Softmax(dim=1)

    def projection_distribution(self, next_states, rewards, dones):
        batch_size = next_states.size()[0]
        rewards = rewards.data.cpu()
        dones = dones.data.cpu()

        delta_z = float(self.Vmax - self.Vmin) / (self.num_atoms - 1)  # 0.4
        support = torch.linspace(self.Vmin, self.Vmax, self.num_atoms)

        next_dist = self.target_net(next_states).data.cpu() * support  # [309,51]
        next_action = next_dist.sum(2).max(1)[1]
        # [batch_size, 1, num_atoms]
        next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(2))

        # [batch_size, num_atoms]
        next_dist = next_dist.gather(1, next_action).squeeze(1)

        # [batch_size, num_atoms]
        rewards = rewards.unsqueeze(1).expand_as(next_dist)
        dones = dones.unsqueeze(1).expand_as(next_dist)
        support = support.unsqueeze(0).expand_as(next_dist)

        Tz = rewards + (1 - dones) * self.discount_factor * support
        Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)
        b = (Tz - self.Vmin) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        offset = torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size).long() \
            .unsqueeze(1).expand(batch_size, self.num_atoms)

        proj_dist = torch.zeros(next_dist.size())
        proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))
        # batch_size * num_atoms
        # print(f'proj_dist: {proj_dist}')
        return proj_dist

    def step(self, state):
        with torch.no_grad():
            epsilon = self.epsilons[min(self.timestep, self.epsilon_decay_steps - 1)]
            state_obs = torch.FloatTensor(state['obs']).view(1, -1).to(self.device)
            # [num_actions, num_atoms]
            q_values_dist = self.q_net(state_obs)[0]
            q_values = torch.sum(q_values_dist * torch.FloatTensor(self.num_atoms).view(1, 1, -1), dim=2).view(
                -1).cpu().detach().numpy()
            probs = remove_illegal(q_values, state['legal_actions'])
            max_action = np.argmax(probs)

            if np.random.uniform() < epsilon:
                probs = remove_illegal(np.ones(self.num_actions), state['legal_actions'])
                action = np.random.choice(self.num_actions, p=probs)
            else:
                action = max_action

        return action

    def eval_step(self, state, use_max=True):
        with torch.no_grad():
            state_obs = torch.FloatTensor(state['obs']).view(1, -1).to(self.device)
            # [num_actions, num_atoms]
            # q_values_dist = self.softmax(self.q_net(state_obs)[0])
            q_values_dist = self.q_net(state_obs)[0]
            q_values = torch.sum(q_values_dist * torch.FloatTensor(self.num_atoms).view(1, 1, -1), dim=2).view(
                -1).cpu().detach().numpy()
            #print(q_values)
            probs = remove_illegal(q_values, state['legal_actions'])

            if use_max:
                action = np.argmax(probs)
            else:
                action = np.random.choice(self.num_actions, p=probs)
 
        return action, probs

    def add_transition(self, transition):
        state, action, reward, next_state, done = transition
        self.memory.save(state['obs'], action, reward, next_state['obs'], done)
        self.timestep += 1
        if self.timestep >= self.replay_memory_init_size and self.timestep % self.train_every == 0:
            batch_loss = self.batch_update()
            print(f'\rstep: {self.timestep}, loss on batch: {batch_loss}', end='')
        if self.timestep % self.update_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.target_net.eval()
            print(f'target parameters updated at step {self.timestep}')

    def batch_update(self):
        states, actions, rewards, next_states, dones = self.memory.sample()

        states = torch.FloatTensor(states).view(self.batch_size, -1).to(self.device)
        next_states = torch.FloatTensor(next_states).view(self.batch_size, -1).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = (1 - torch.LongTensor(dones)).to(self.device)

        proj_dists = self.projection_distribution(next_states, rewards, dones)
        proj_dists.float().to(self.device)
        #print(proj_dists)
        dists = self.q_net(states)
        actions = actions.unsqueeze(1).unsqueeze(1).expand(self.batch_size, 1, self.num_atoms)
        dists = dists.gather(1, actions).squeeze(1)
        dists.data.clamp_(0.01, 0.99)
        loss = -(proj_dists * dists.log()).sum(-1).mean()

        self.q_net.train()
        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()
        self.q_net.eval()

        return loss.item()

    def save_state_dict(self, file_path):
        """
        save state dict for the networks of DQN agent
        Input:
            file_path (str): string filepath to save the agent at
        """

        state_dict = dict()
        state_dict['q_net'] = self.q_net.state_dict()
        state_dict['target_net'] = self.target_net.state_dict()

        torch.save(state_dict, file_path)

    def load_from_state_dict(self, filepath):
        """
        Load agent parameters from filepath
        Input:
            file_path (str) : string filepath to load parameters from
        """
        state_dict = torch.load(filepath, map_location=self.device)
        self.q_net.load_state_dict(state_dict['q_net'])
        self.target_net.load_state_dict(state_dict['target_net'])

    @staticmethod
    def KL_divergence_two_dist(dist_p, dist_q):
        kld = torch.sum(dist_p * (torch.log(dist_p) - torch.log(dist_q)))
        return kld

