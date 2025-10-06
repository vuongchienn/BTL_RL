import numpy as np
from parking_env import ParkingEnv
import matplotlib.pyplot as plt
from parking_env import create_parking_env

env = ParkingEnv()
env = create_parking_env()
env.random_occupancy =True
n_actions = env.action_space.n

# Q-table dạng dict với key = state_id (int)
Q = {}
def get_Q(state):
    sid = env._state_to_id(state)   # ánh xạ state -> số nguyên
    return Q.setdefault(sid, np.zeros(n_actions))

# Tham số
episodes = 5000  # train lâu nhưng không render
alpha = 0.1
gamma = 0.9
epsilon = 0.1

# Train
for ep in range(episodes):
    state = env.reset()
    done = False
    while not done:
        # Epsilon-greedy
        if np.random.rand() < epsilon:
            action = np.random.choice(n_actions)
        else:
            action = np.argmax(get_Q(state))

        next_state, reward, done, _ = env.step(action)

        Qs = get_Q(state)
        Q_next = get_Q(next_state)

        Qs[action] += alpha * (reward + gamma * np.max(Q_next) - Qs[action])
        state = next_state

print("✅ Q-learning training done!")

# In Q-table (chỉ vài state đầu, đã ánh xạ sang số nguyên)
print("\n--- Một phần Q-table ---")
for i, (sid, q_vals) in enumerate(Q.items()):
    print(f"StateID {sid} -> {q_vals}")
    if i >= 10:  # in 10 dòng đầu thôi
        break

# Test chạy 1 episode với policy greedy
state = env.reset()
done = False
print("\n--- Demo chạy thử với Q-table ---")
while not done:
    action = np.argmax(get_Q(state))  # greedy
    state, reward, done, _ = env.step(action)
    env.render()
print("Kết thúc episode với reward:", reward)
env.close()

# Hiển thị Q-table trên giao diện
def show_q_table(Q):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")

    # Lấy một số state để minh họa (vd: 15 state đầu)
    rows = []
    for i, (sid, q_vals) in enumerate(Q.items()):
        rows.append([str(sid)] + [f"{q:.2f}" for q in q_vals])
        if i >= 15:
            break

    col_labels = ["StateID", "Up", "Down", "Left", "Right", "Park"]
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.2)

    plt.show()

show_q_table(Q)