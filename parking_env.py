import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ParkingEnv(gym.Env):
    def __init__(self, N=50, parking_spots=None, obstacles=None,occupancy_init=None,random_occupancy=False):
        super(ParkingEnv, self).__init__()
        self.N = N
        self.pos_agent = (4,8)
        self.random_occupancy = random_occupancy
        self.parking_spots = parking_spots if parking_spots else [(N-1, N-1)]
        if occupancy_init is None:
            self.occupancy_init = [0 for _ in self.parking_spots]
        else:
            assert len(occupancy_init) == len(self.parking_spots), "occupancy_init phải cùng độ dài với parking_spots"
            self.occupancy_init = occupancy_init.copy()
        self.occupancy = self.occupancy_init.copy()
        self.obstacles = obstacles if obstacles else [(5, 5)]

        #0 = up,1 = down,2 = left,3 = right,4 = park
        self.action_space = spaces.Discrete(5)

        low = np.array([0, 0] + [0] * len(self.parking_spots))
        high = np.array([N-1, N-1] + [1] * len(self.parking_spots))
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        self.fig, self.ax = plt.subplots(figsize=(15, 15))  # hoặc to hơn nếu muốn


    def _state_to_id(self, state):
        return tuple(state)

    
    def reset(self):
        self.pos_agent = (4,8)

        # Chỉ random occupancy 1 lần khi bắt đầu
        if not hasattr(self, "_occupancy_initialized"):
            if self.random_occupancy:
                self.occupancy = np.random.choice([0, 1], size=len(self.parking_spots)).tolist()
            else:
                self.occupancy = self.occupancy_init.copy()
            self._occupancy_initialized = True
        else:
            self.occupancy = self.occupancy.copy()  # Giữ nguyên occupancy đã random trước đó

        return self._get_obs()


    def _get_obs(self):
        return np.array(list(self.pos_agent) + self.occupancy, dtype=np.int32)

    def step(self, action):
        x, y = self.pos_agent
        reward = -5 
        done = False

        blocked_cells = set(self.obstacles)
        for i, spot in enumerate(self.parking_spots):
            if self.occupancy[i] == 1:  
                blocked_cells.add(spot)

        if action == 0:  # up
            if y > 0:
                y -= 1
        elif action == 1:  # down
            if y < self.N - 1:
                y += 1
        elif action == 2:  # left
            if x > 0:
                x -= 1
        elif action == 3:  # right
            if x < self.N - 1:
                x += 1
        elif action == 4:  # park
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

        # check va chạm vật cản
        if (x, y) in blocked_cells:
            reward = -100
            done = True

        self.pos_agent = (x, y)
        return self._get_obs(), reward, done, {}


    def render(self, mode="human"):
        cell_size = 2 # 👈 chỉnh số này (2,3,5...) để ô vuông to lên
        self.ax.clear()
        self.ax.set_xlim(-0.5 * cell_size, (self.N - 0.5) * cell_size)
        self.ax.set_ylim(-0.5 * cell_size, (self.N - 0.5) * cell_size)
        self.ax.set_aspect('equal')

        # Đặt ticks (số hàng/cột) theo đúng lưới
        self.ax.set_xticks([i * cell_size for i in range(self.N)])
        self.ax.set_yticks([j * cell_size for j in range(self.N)])

        # Đặt nhãn (labels) hiển thị 0..N-1
        self.ax.set_xticklabels([str(i) for i in range(self.N)])
        self.ax.set_yticklabels([str(j) for j in range(self.N)])


        # Vẽ nền ô vuông dạng bàn cờ
        for i in range(self.N):
            for j in range(self.N):
                color = "#FFFFFF"
                self.ax.add_patch(
                    patches.Rectangle(
                        (i * cell_size - 0.5 * cell_size, j * cell_size - 0.5 * cell_size),
                        cell_size, cell_size,
                        edgecolor='gray',
                        facecolor=color
                    )
                )

        # Vẽ vật cản
        for ox, oy in self.obstacles:
            self.ax.add_patch(
                patches.Rectangle(
                    (ox * cell_size - 0.5 * cell_size, oy * cell_size - 0.5 * cell_size),
                    cell_size, cell_size,
                    color="black"
                )
            )

        # Vẽ chỗ đỗ
        for i, (px, py) in enumerate(self.parking_spots):
            color = "green" if self.occupancy[i] == 0 else "red"
            self.ax.add_patch(
                patches.Rectangle(
                    (px * cell_size - 0.5 * cell_size, py * cell_size - 0.5 * cell_size),
                    cell_size, cell_size,
                    edgecolor="black",
                    facecolor=color,
                    alpha=0.6
                )
            )

        # Vẽ xe
        ax, ay = self.pos_agent
        self.ax.add_patch(
            patches.Circle((ax * cell_size, ay * cell_size), 0.3 * cell_size, color="blue")
        )

        plt.pause(0.05)

def create_parking_env():
    # ———————————— VẬT CẢN ————————————

    # Tường biên
    wall_obstacles = [(x, 0) for x in range(0, 51)]
    column50_obstacles = [(49, y) for y in range(0, 50)]
    row50_obstacles = [(x, 49) for x in range(0, 50)]

    # Vật cản dạng cột/hàng
    column_obstacles_35 = [(35, y) for y in range(0, 5)]
    row_obstacles_4 = [(x, 4) for x in range(4, 36)]
    extra_obstacles = [(4, 5), (4, 6)]
    toll_station_obstacles = [(x, y) for x in range(4, 8) for y in range(9, 12)]
    horizontal_obstacles_6 = [(x, 6) for x in range(4, 8)]
    horizontal_obstacles_14 = [(x, 14) for x in range(4, 36) if x not in (11, 12)]

    # Vật cản dạng hàng/cột đặc biệt
    obstacles_row_26 = [(x, 26) for x in range(3, 36)]
    obstacles_column_3 = [(3, y) for y in range(14, 27) if y not in (18, 19, 22, 23)]
    obstacles_column_40 = [(40, y) for y in range(36, 45)]
    obstacles_row_30 = [(x, 30) for x in range(32, 41)]
    obstacles_column_35 = [(35, y) for y in range(27, 30)]
    obstacles_column_40_31_35 = [(40, y) for y in range(31, 33)]
    obstacles_row_38 = [(x, 38) for x in range(32, 40)]
    obstacles_column_32 = [(32, y) for y in range(31, 38)]
    obstacles_row_44 = [(x, 44) for x in range(0, 40)]

    # Ô vuông / Hình chữ nhật vật cản
    square_obstacles_4_20_6_21 = [(x, y) for x in range(4, 7) for y in range(20, 22)]
    square_obstacles_4_24_6_25 = [(x, y) for x in range(4, 7) for y in range(24, 26)]
    square_obstacles_0_24_2_26 = [(x, y) for x in range(0, 3) for y in range(24, 27)]
    square_obstacles_0_20_2_21 = [(x, y) for x in range(0, 3) for y in range(20, 22)]
    square_obstacles_0_14_2_17 = [(x, y) for x in range(0, 3) for y in range(14, 18)]
    square_obstacles_0_9_3_11 = [(x, y) for x in range(0, 4) for y in range(9, 12)]
    square_obstacles_0_4_3_6 = [(x, y) for x in range(0, 4) for y in range(4, 7)]
    rectangle_obstacles_45_48 = [(x, y) for x in range(45, 49) for y in range(11, 49)]
    rectangle_obstacles_2_46_5_47 = [(x, y) for x in range(2, 6) for y in range(46, 48)]

    # ———————————— CHỖ ĐỖ ————————————

    parking_spots_rectangle = [(x, y) for x in range(13, 41) for y in range(16, 18)]
    blocked_spots = [(17, 17), (22, 17), (28, 17), (33, 17)]
    parking_spots_rectangle = [spot for spot in parking_spots_rectangle if spot not in blocked_spots]

    parking_spots_rectangle_2 = [(x, y) for x in range(13, 41) for y in range(20, 22)]
    parking_spots_rectangle_3 = [(x, y) for x in range(13, 36) for y in range(24, 26)]
    parking_spots_row_1 = [(x, 1) for x in range(36, 49)]
    parking_spots_row_2 = [(x, 15) for x in range(4, 11)]
    parking_spots_row_3 = [(x, 13) for x in range(13, 41)]
    parking_spots_rectangle_4 = [(x, y) for x in range(38, 47) for y in range(3, 5)]
    parking_spots_rectangle_38_46_7_8 = [(x, y) for x in range(38, 47) for y in range(7, 9)]
    parking_spots_square_37_39_27_28 = [(x, y) for x in range(37, 40) for y in range(27, 29)]
    parking_spots_column_41 = [(41, y) for y in range(24, 45) if y not in (35, 33, 34)]
    parking_spots_column_44 = [(44, y) for y in range(24, 49)]
    parking_rectangle_34_38 = [(x, y) for x in range(34, 39) for y in range(32, 34)]
    parking_rectangle_34_38_35_36 = [(x, y) for x in range(34, 39) for y in range(35, 37)]
    parking_row_10_43_48 = [(x, 48) for x in range(10, 44)]
    parking_spots_row_10_41_45 = [(x, 45) for x in range(10, 42)]

    # ———————————— GỘP VẬT CẢN ————————————
    all_obstacles = (
        wall_obstacles + column50_obstacles + row50_obstacles +
        row_obstacles_4 + column_obstacles_35 + extra_obstacles +
        toll_station_obstacles + horizontal_obstacles_6 +
        horizontal_obstacles_14 + obstacles_row_26 +
        obstacles_column_3 + obstacles_column_40 +
        obstacles_row_30 + obstacles_column_35 +
        obstacles_column_40_31_35 + obstacles_row_38 +
        obstacles_column_32 + obstacles_row_44 +
        square_obstacles_4_20_6_21 + square_obstacles_4_24_6_25 +
        square_obstacles_0_24_2_26 + square_obstacles_0_20_2_21 +
        square_obstacles_0_14_2_17 + square_obstacles_0_9_3_11 +
        square_obstacles_0_4_3_6 + rectangle_obstacles_45_48 +
        rectangle_obstacles_2_46_5_47
    )

    # ———————————— GỘP CHỖ ĐỖ ————————————
    parking_spots_rectangle += (
        parking_spots_rectangle_2 + parking_spots_rectangle_3 +
        parking_spots_row_1 + parking_spots_row_2 +
        parking_spots_row_3 + parking_spots_rectangle_4 +
        parking_spots_rectangle_38_46_7_8 + parking_spots_square_37_39_27_28 +
        parking_spots_column_41 + parking_spots_column_44 +
        parking_rectangle_34_38 + parking_rectangle_34_38_35_36 +
        parking_row_10_43_48 + parking_spots_row_10_41_45
    )

    env = ParkingEnv(
        N=50,
        parking_spots=parking_spots_rectangle,
        obstacles=all_obstacles,
        occupancy_init=[0] * len(parking_spots_rectangle)
    )

    return env




if __name__ == "__main__":
   # Tạo hàng tường (trừ đoạn 7-12)
    wall_obstacles = [(x, 0) for x in range(0, 51)]

    # Full cột số 50
    column50_obstacles = [(49, y) for y in range(0, 50)]
    # Full hàng số 50
    row50_obstacles = [(x, 49) for x in range(0, 50)]
    # Cột tường từ (35,0) đến (35,4)
    column_obstacles = [(35, y) for y in range(0, 5)]
     # Hàng tường từ (4,4) đến (35,4)
    row_obstacles = [(x, 4) for x in range(4, 36)]
    # Hàng vật cản mới từ (3,26) → (35,26)
    obstacles_row_26 = [(x, 26) for x in range(3, 36)]
     # Thêm hàng mới (4,15) → (10,15)
    parking_spots_row_2 = [(x, 15) for x in range(4, 11)]
      # Thêm cột vật cản riêng lẻ (4,5) và (4,6)
    extra_obstacles = [(4, 5), (4, 6)]
     # Trạm thu phí
    toll_station_obstacles = [(x, y) for x in range(4, 8) for y in range(9, 12)]
    horizontal_obstacles = [(x, 6) for x in range(4, 8)]
    horizontal_obstacles_3 = [(x, 14) for x in range(4, 36) if x not in (11, 12)]

    parking_spots_rectangle = [
        (x, y) for x in range(13, 41) for y in range(16, 18)
    ]

    # Các ô cần biến thành vật cản
    blocked_spots = [(17,17), (22,17), (28,17), (33,17)]

    # Loại bỏ các ô đó khỏi chỗ đỗ
    parking_spots_rectangle = [
        spot for spot in parking_spots_rectangle if spot not in blocked_spots
    ]
    # Tạo chỗ đỗ hình chữ nhật
    parking_spots_rectangle_2 = [
        (x, y) for x in range(13, 41) for y in range(20, 22)
    ]
    # Tạo hình chữ nhật chỗ đỗ mới
    parking_spots_rectangle_3 = [
        (x, y) for x in range(13, 36) for y in range(24, 26)
    ]
    parking_spots_row = [(x, 1) for x in range(36, 49)]
        # Hàng chỗ đỗ mới (13,13) → (35,13)
    parking_spots_row_3 = [(x, 13) for x in range(13, 41)]
    # Cột vật cản từ (3,14) → (3,26)
    # Cột vật cản từ (3,14) → (3,26), trừ một số ô
    obstacles_column_3 = [(3, y) for y in range(14, 27) if y not in (18, 19, 22, 23)]
    # Ô vuông vật cản từ (4,20) đến (6,21)
    square_obstacles = [(x, y) for x in range(4, 7) for y in range(20, 22)]
    square_obstacles1 = [(x, y) for x in range(4, 7) for y in range(24, 26)]
    horizontal_row_obstacles = [(x, 17) for x in range(4, 7)]
    parking_spots_rectangle_4 = [
    (x, y) for x in range(38, 47) for y in range(3, 5)
    ]
    rectangle_obstacles = [
    (x, y) for x in range(45, 49) for y in range(11, 49)
    ]
    column_obstacles_40 = [(40, y) for y in range(36, 45)]
    row_obstacles_35_40_30 = [(x, 30) for x in range(32, 41)]
    column_obstacles_35_27_29 = [(35, y) for y in range(27, 30)]    
    column_obstacles_40_31_35 = [(40, y) for y in range(31, 33)]
    obstacle_row_38 = [
    (x, 38) for x in range(32, 40)
    ]
    obstacle_column_32 = [
    (32, y) for y in range(31, 38)
    ]
    square_obstacles_0_24_2_26 = [(x, y) for x in range(0, 3) for y in range(24, 27)]


    parking_spots_square_37_39_27_28 = [
    (x, y) for x in range(37, 40) for y in range(27, 29)
    ]
    # Gộp vật cản
    
    parking_spots_rectangle_38_46_7_8 = [
    (x, y) for x in range(38, 47) for y in range(7, 9)
    ]
    parking_spots_column_41 = [
    (41, y) for y in range(24, 45) if y not in (35,33,34)
    ]
    parking_spots_column_44 = [
    (44, y) for y in range(24, 49)
    ]
    parking_rectangle_34_38 = [
    (x, y) for x in range(34, 39) for y in range(32, 34)
    ]
    parking_rectangle_34_38_35_36 = [
        (x, y) for x in range(34, 39) for y in range(35, 37)
    ]
    parking_row_10_43_48 = [(x, 48) for x in range(10, 44)]

    obstacle_row_0_39_44 = [(x, 44) for x in range(0, 40)]
    parking_spots_row_10_41_45 = [(x, 45) for x in range(10, 42)]
    rectangle_obstacles_2_46_5_47 = [(x, y) for x in range(2, 6) for y in range(46, 48)]
    square_obstacles_0_20_2_21 = [(x, y) for x in range(0, 3) for y in range(20, 22)]
    square_obstacles_0_14_2_17 = [(x, y) for x in range(0, 3) for y in range(14, 18)]
    square_obstacles_0_9_3_11 = [(x, y) for x in range(0, 4) for y in range(9, 12)]
    square_obstacles_0_4_3_6 = [(x, y) for x in range(0, 4) for y in range(4, 7)]



    all_obstacles =square_obstacles_0_4_3_6+square_obstacles_0_9_3_11+square_obstacles_0_14_2_17+square_obstacles_0_20_2_21+square_obstacles_0_24_2_26+ rectangle_obstacles_2_46_5_47+obstacle_row_0_39_44+obstacle_column_32+obstacle_row_38+column_obstacles_40_31_35+column_obstacles_35_27_29+row_obstacles_35_40_30+ column_obstacles_40+rectangle_obstacles+horizontal_row_obstacles+square_obstacles1+square_obstacles+obstacles_column_3+ obstacles_row_26+wall_obstacles + column50_obstacles+row50_obstacles +row_obstacles+column_obstacles + extra_obstacles + toll_station_obstacles+horizontal_obstacles +horizontal_obstacles_3 + blocked_spots
    # Thêm vào parking_spots hiện tại
    parking_spots_rectangle += parking_spots_row_10_41_45+ parking_row_10_43_48+ parking_rectangle_34_38_35_36+ parking_rectangle_34_38+parking_spots_column_44+ parking_spots_column_41+ parking_spots_rectangle_38_46_7_8+ parking_spots_square_37_39_27_28+ parking_spots_rectangle_4+ parking_spots_row+ parking_spots_rectangle_2 + parking_spots_rectangle_3 + parking_spots_row_2 + parking_spots_row_3
    env = ParkingEnv(
        N=50,
        parking_spots=parking_spots_rectangle,
        obstacles=all_obstacles,
         occupancy_init=[0] * len(parking_spots_rectangle)
    )

    obs = env.reset()
    env.render() 
    plt.show(block=True)
    env.close()    