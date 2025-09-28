import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ParkingEnv(gym.Env):
    def __init__(self, N=5, parking_spots=None, obstacles=None,occupancy_init=None):
        super(ParkingEnv, self).__init__()
        self.N = N

        # Tọa độ xe (x,y)
        self.pos_agent = (0, 0)

        # Chỗ đỗ
        self.parking_spots = parking_spots if parking_spots else [(N-1, N-1)]
        # Nếu không truyền occupancy thì mặc định tất cả trống (0)
        if occupancy_init is None:
            self.occupancy_init = [0 for _ in self.parking_spots]
        else:
            assert len(occupancy_init) == len(self.parking_spots), "occupancy_init phải cùng độ dài với parking_spots"
            self.occupancy_init = occupancy_init.copy()
        # Bản occupancy hiện tại (sẽ thay đổi trong quá trình chạy)
        self.occupancy = self.occupancy_init.copy()
        # Vật cản
        self.obstacles = obstacles if obstacles else [(2, 2)]

        # Action: 0=Up,1=Down,2=Left,3=Right,4=Park
        self.action_space = spaces.Discrete(5)

        # Quan sát: (x, y, trạng thái chỗ đỗ)
        low = np.array([0, 0] + [0] * len(self.parking_spots))
        high = np.array([N-1, N-1] + [1] * len(self.parking_spots))
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        # Vẽ giao diện
        self.fig, self.ax = plt.subplots()

    def _state_to_id(self, state):
        """Ánh xạ (x,y,occupancy) -> số nguyên"""
        x, y, *occ = state
        occ_num = int("".join(map(str, occ)), 2) if occ else 0
        return y * self.N + x + occ_num * (self.N * self.N)
    
    def reset(self):
        self.pos_agent = (0, 0)
        self.occupancy = self.occupancy_init.copy()
        return self._get_obs()

    def _get_obs(self):
        return np.array(list(self.pos_agent) + self.occupancy, dtype=np.int32)

    def step(self, action):
        x, y = self.pos_agent
        reward = -5   # mặc định -5 mỗi bước
        done = False

        # Tạo danh sách ô bị cấm đi vào (vật cản + chỗ đỗ đã đầy)
        blocked_cells = set(self.obstacles)
        for i, spot in enumerate(self.parking_spots):
            if self.occupancy[i] == 1:  # chỗ này đã đầy
                blocked_cells.add(spot)

        if action == 0:  # Up
            if y > 0 and (x, y - 1) not in blocked_cells:
                y -= 1
        elif action == 1:  # Down
            if y < self.N - 1 and (x, y + 1) not in blocked_cells:
                y += 1
        elif action == 2:  # Left
            if x > 0 and (x - 1, y) not in blocked_cells:
                x -= 1
        elif action == 3:  # Right
            if x < self.N - 1 and (x + 1, y) not in blocked_cells:
                x += 1
        elif action == 4:  # Park
            if self.pos_agent in self.parking_spots:
                idx = self.parking_spots.index(self.pos_agent)
                if self.occupancy[idx] == 0:
                    reward = 100   # đỗ thành công
                    done = True
                else:
                    reward = -100  # cố gắng đỗ vào chỗ đã đầy
                    done = True
            else:
                reward = -50       # đỗ sai vị trí
                done = True

        # Check va chạm vật cản
        if (x, y) in self.obstacles:
            reward = -100
            done = True

        self.pos_agent = (x, y)
        return self._get_obs(), reward, done, {}


    def render(self, mode="human"):
        self.ax.clear()
        self.ax.set_xlim(-0.5, self.N - 0.5)
        self.ax.set_ylim(-0.5, self.N - 0.5)
        self.ax.set_xticks(range(self.N))
        self.ax.set_yticks(range(self.N))
        self.ax.grid(True)

        # Vẽ vật cản
        for ox, oy in self.obstacles:
            self.ax.add_patch(patches.Rectangle((ox-0.5, oy-0.5), 1, 1, color="black"))

        # Vẽ chỗ đỗ
        for i, (px, py) in enumerate(self.parking_spots):
            color = "green" if self.occupancy[i] == 0 else "red"
            self.ax.add_patch(patches.Rectangle((px-0.5, py-0.5), 1, 1, edgecolor="black", facecolor=color, alpha=0.5))

        # Vẽ xe
        ax, ay = self.pos_agent
        self.ax.add_patch(patches.Circle((ax, ay), 0.3, color="blue"))

        plt.pause(0.5)
    def close(self):
        plt.show(block=True)   # giữ cửa sổ mở cho đến khi người dùng đóng

