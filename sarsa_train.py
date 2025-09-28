import numpy as np
from parking_env import ParkingEnv

env = ParkingEnv(N=5, parking_spots=[(4,4)], obstacles=[(2,2)])
n_actions = env.action_space.n
state_size = len(env._get_obs())

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

    # Chọn hành động đầu tiên
    if np.random.rand() < epsilon:
        action = np.random.choice(n_actions)
    else:
        action = np.argmax(get_Q(state))

    while not done:
        env.render()
        next_state, reward, done, _ = env.step(action)

        if np.random.rand() < epsilon:
            next_action = np.random.choice(n_actions)
        else:
            next_action = np.argmax(get_Q(next_state))

        Qs = get_Q(state)
        Q_next = get_Q(next_state)

        Qs[action] += alpha * (reward + gamma * Q_next[next_action] - Qs[action])

        state = next_state
        action = next_action

print("SARSA xong!")
