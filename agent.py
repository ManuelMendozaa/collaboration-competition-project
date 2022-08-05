import copy
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from buffer import ReplayMemory, PrioritizedMemory
from models import Actor, Critic

class Agent():
    def __init__(self, state_size, action_size, params, seed=14):
        # Extract args
        self._lr = params['lr']                                     #  learning rate (α)
        self._gamma = params['gamma']                               #  discount factor (γ)
        self._tau = params['tau']                                   #  soft update factor (τ)
        self._epsilon = params['epsilon']                           #  noise factor (ε)
        self._epsilon_decay = params['epsilon_decay']               #  ε decay per training
        self._epsilon_decay_type = params['epsilon_decay_type']     #  ε decay per training
        self._batch_size = params['batch_size']                     #  size of every sample batch
        self._buffer_size = params['buffer_size']                   #  max number of experiences to store
        self._n_steps = params['n_steps']                           #  bootstrapping from experiences for the loss
        self._learning_iters = params['learning_iters']             #  number of loops at every learning step
        self._min_experience = params['min_experience']             #  min number of experiences to start learning
        self._device = params['device']                             #  available processing device
        self._extensions = params['extensions']                     #  reinforcement learning extensions
        self._ou_theta = params['ou_theta']                         #  noise parameter θ
        self._ou_sigma = params['ou_sigma']                         #  noise parameter σ
        self._device = params['device']

        layer_units = params['layer_units']

        # Extensions
        self._prioritized_replay = 'Prioritized Replay' in self._extensions
        self._dueling = 'Dueling Network' in self._extensions

        # Initialize seeds
        self._random_seed = random.seed(seed)
        self._numpy_seed = np.random.seed(seed)

        # Actor Networks
        self._actor_local = Actor(state_size, action_size, seed=seed, layer_units=layer_units).to(self._device)
        self._actor_target = Actor(state_size, action_size, seed=seed, layer_units=layer_units).to(self._device)
        self._actor_optimizer = optim.Adam(self._actor_local.parameters(), lr=self._lr)

        # Critic Networks
        self._critic_local = Critic(state_size, action_size, seed=seed, layer_units=layer_units, dueling=self._dueling).to(self._device)
        self._critic_target = Critic(state_size, action_size, seed=seed, layer_units=layer_units, dueling=self._dueling).to(self._device)
        self._critic_optimizer = optim.Adam(self._critic_local.parameters(), lr=self._lr)

        # Noise process for two agent
        self._noise = OUNoise(action_size, seed=seed, theta=self._ou_theta, sigma=self._ou_sigma)

        # Replay buffer
        BufferStructure = PrioritizedMemory if self._prioritized_replay else ReplayMemory
        self._buffer = BufferStructure(self._buffer_size, self._batch_size, self._device)

        # T step counter
        self._waiting_steps = 0

    def step(self, state, action, reward, next_state, done):
        self._buffer.save(state, action, reward, next_state, done)
        self._waiting_steps = (self._waiting_steps + 1) % self._n_steps

        if len(self._buffer) >= self._min_experience and self._waiting_steps == self._n_steps - 1:
            for _ in range(self._learning_iters):
                self.learn()



    def act(self, state, add_noise=True):
        """ Agent takes the action based on the state and its actor local model """
        # Get state into available device
        state = torch.from_numpy(state).float().to(self._device)

        # Activate evaluation mode
        self._actor_local.eval()

        # Compute actions
        with torch.no_grad():
            actions = self._actor_local(state).cpu().data.numpy()

        if add_noise:
            actions += self._epsilon * self._noise.sample()

        # Return actor to train mode
        self._actor_local.train()

        # Keep continuous action between -1 and 1
        return np.clip(actions, -1, 1)

    def learn(self):
        # Decompose buffer sample
        if self._prioritized_replay:
            (states, actions, rewards, next_states, dones), weights, indices = self._buffer.sample()
        else:
            states, actions, rewards, next_states, dones = self._buffer.sample()

        # Critic forward
        next_actions = self._actor_target(next_states)
        Q_values_next = self._critic_target(next_states, next_actions)
        Q_values = rewards + (self._gamma * Q_values_next * (1 - dones))
        Q_values_expected = self._critic_local(states, actions)
        # Critic loss
        critic_loss = F.mse_loss(Q_values_expected, Q_values, reduce=(not self._prioritized_replay))
        # Actor forward
        local_actions = self._actor_local(states)
        actor_loss = -self._critic_local(states, local_actions)


        if self._prioritized_replay:
            actor_loss = torch.mean(weights * actor_loss)
            critic_loss = torch.mean(weights * critic_loss)
            errors = torch.abs(Q_values_expected - Q_values).mean(dim=1).cpu().data.numpy()
        else:
            actor_loss = actor_loss.mean()


        # Backward steps
        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        self._actor_optimizer.step()

        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._critic_local.parameters(), 1)
        self._critic_optimizer.step()


        # Update target networks
        self.soft_update()

        # Update priorities in buffer
        if self._prioritized_replay:
            self._buffer.update_priorities(indices, errors)

        # Update noise factor
        self._epsilon = max(self._epsilon - self._epsilon_decay, 0.01)
        self.reset()

    def soft_update(self):
        """ Update the target networks with their own parameters and the local networks' """
        # Critic update
        for critic_target_param, critic_local_param in zip(self._critic_target.parameters(), self._critic_local.parameters()):
            critic_target_param.data.copy_(self._tau * critic_local_param.data + (1.0-self._tau) * critic_target_param.data)

        # Actor update
        for actor_target_param, actor_local_param in zip(self._actor_target.parameters(), self._actor_local.parameters()):
            actor_target_param.data.copy_(self._tau * actor_local_param.data + (1.0-self._tau) * actor_target_param.data)

    def reset(self):
        self._noise.reset()

class OUNoise():
    """ Ornstein-Unlenbeck process """
    def __init__(self, size, seed, mu=0, theta=0.2, sigma=0.15):
        self._mu = mu * np.ones(size)
        self._theta = theta
        self._sigma = sigma
        self._seed = random.seed(seed)
        self._size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self._state = copy.copy(self._mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self._state
        dx = self._theta * (self._mu - x) + self._sigma * np.random.standard_normal(self._size)
        self._state = x + dx
        return self._state
