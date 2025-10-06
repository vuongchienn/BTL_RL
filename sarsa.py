import numpy as np
from parking_env import ParkingEnv
from parking_env import create_parking_env

env = ParkingEnv()
env = create_parking_env()
env.random_occupancy = True
n_actions = env.action_space.n

Q = {}
def get_Q(state):
    return Q.setdefault(tuple(state), np.zeros(n_actions))

episodes = 5000
alpha = 0.1
gamma = 0.9
epsilon = 0.1

# Train
for ep in range(episodes):
    state = env.reset()
    if np.random.rand() < epsilon:
        action = np.random.choice(n_actions)
    else:
        action = np.argmax(get_Q(state))

    done = False
    while not done:
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

print("✅ SARSA training done!")

# In Q-table (10 state đầu tiên)
print("\n--- Một phần Q-table ---")
for i, (s, q_vals) in enumerate(Q.items()):
    print(f"State {s} -> {q_vals}")
    if i >= 10:
        break

# Demo chạy thử
state = env.reset()
done = False
print("\n--- Demo chạy thử với Q-table ---")
while not done:
    action = np.argmax(get_Q(state))  # greedy
    state, reward, done, _ = env.step(action)
    env.render()
print("Kết thúc episode với reward:", reward)
env.close() 

#hien thi ket qua
import matplotlib.pyplot as plt

def show_q_table(Q):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")

    rows = []
    for i, (state, q_vals) in enumerate(Q.items()):
        state_id = env._state_to_id(state)  # Ánh xạ state → state_id
        rows.append([str(state_id)] + [f"{q:.2f}" for q in q_vals])
        if i >= 15:
            break

    col_labels = ["StateID", "Up", "Down", "Left", "Right", "Park"]
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.2)

    plt.show()

show_q_table(Q)
