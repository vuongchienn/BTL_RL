import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt
import time
from parking_env import create_parking_env

# ================= Q-Network =================
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# ================= Replay Buffer =================
from collections import deque
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# ================= DQN Agent =================
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.95, epsilon_start=1.0, epsilon_final=0.05, epsilon_decay=500):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start

        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        self.update_target_network()
        self.memory = ReplayBuffer(10000)
        self.steps_done = 0

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, state):
        self.epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                       np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                return self.q_network(state).argmax().item()

    def train(self, batch_size):
        if len(self.memory) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_network(next_states).max(1)[0]
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, expected_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# ================= Huấn luyện DQN =================
def train_dqn(env, agent, episodes=5000, batch_size=64, target_update=10):
    rewards_history = []
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            agent.train(batch_size)

        rewards_history.append(total_reward)

        if ep % target_update == 0:
            agent.update_target_network()

        if ep % 100 == 0:
            print(f"Episode {ep}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    return rewards_history

# ================= Show policy chạy thử =================
def demo_run(env, agent):
    state = env.reset()
    done = False
    print("\n--- Demo chạy thử với policy DQN ---")
    while not done:
        env.render()
        time.sleep(0.05)
        action = agent.q_network(torch.FloatTensor(state).unsqueeze(0)).argmax().item()
        state, reward, done, _ = env.step(action)
    print("Kết thúc episode với reward:", reward)

# ================= Show Q-table dạng đơn giản =================
def show_q_table(agent, env, n_states=20):
    print("\n--- Một phần Q-table ---")
    for _ in range(n_states):
        state = env.reset()
        with torch.no_grad():
            q_vals = agent.q_network(torch.FloatTensor(state)).numpy()
        print(f"State: {state} -> {q_vals}")

if __name__ == "__main__":
    env = create_parking_env()
    env.random_occupancy = True
    state_dim = len(env.reset())
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    train_dqn(env, agent, episodes=2000, batch_size=64, target_update=10)

    show_q_table(agent, env, n_states=10)
    demo_run(env, agent)
    plt.show(block=True)
    plt.close('all')
