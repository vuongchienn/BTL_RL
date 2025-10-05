import numpy as np
from parking_env import ParkingEnv

env = ParkingEnv(N=10, parking_spots=[(8,6),(4,3),(6,9)], obstacles=[(2,2),(5,6)], occupancy_init=[0,0,1])
n_actions = env.action_space.n
state_size = len(env._get_obs())

# Q-table
Q = {}

def get_Q(state):
    return Q.setdefault(tuple(state), np.zeros(n_actions))

episodes = 200
alpha = 0.1
gamma = 0.9
epsilon = 0.1

for ep in range(episodes):
    state = env.reset()
    done = False

    while not done:
        env.render()
        if np.random.rand() < epsilon:
            action = np.random.choice(n_actions)
        else:
            action = np.argmax(get_Q(state))

        next_state, reward, done, _ = env.step(action)

        Qs = get_Q(state)
        Q_next = get_Q(next_state)

        Qs[action] += alpha * (reward + gamma * np.max(Q_next) - Qs[action])
        state = next_state

print("Q-learning xong!")
