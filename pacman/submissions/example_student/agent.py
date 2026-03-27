
"""
Example student submission showing the required interface.

Students should implement their own PacmanAgent and/or GhostAgent
following this template.
"""
from collections import deque
from inspect import currentframe
from re import X
import sys
from pathlib import Path
import time
import pickle 

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np
import random

import pacmanAlgorithm

class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Predictive ML Pacman"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        
        # Đọc luật capture_distance từ Command Line
        self.capture_threshold = 1 
        if '--capture-distance' in sys.argv:
            try:
                idx = sys.argv.index('--capture-distance')
                self.capture_threshold = int(sys.argv[idx + 1])
            except (ValueError, IndexError):
                pass
                
        self.algorithm = "ASTAR" 
        
        # --- NẠP MÔ HÌNH MACHINE LEARNING ---
        self.ml_model = None
        try:
            # Dùng Path để tìm chính xác file .pkl nằm cùng thư mục với agent.py
            model_path = Path(__file__).parent / 'ghost_predictor.pkl'
            with open(model_path, 'rb') as f:
                self.ml_model = pickle.load(f)
            print(f"[THÔNG BÁO] Nạp thành công Não bộ AI từ {model_path}!")
        except Exception as e:
            print(f"[CẢNH BÁO] Không thể tải mô hình ML, tự động dùng A* gốc. Lỗi: {e}")
        
        self.locked_target = None
        self.lock_counter = 0

    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        
        try:
            path = []
            
            # 1. KIỂM TRA KHOẢNG CÁCH THỰC TẾ
            dist_to_ghost = abs(my_position[0] - enemy_position[0]) + abs(my_position[1] - enemy_position[1])
            
            # 2. KHỞI ĐỘNG CHIẾN THUẬT HYBRID
            target_pos = enemy_position # Mặc định là đuổi theo sau đuôi
            
            # Kích hoạt Thiên Nhãn (Dự đoán) nếu Ghost ở xa và Model đã nạp
            # Kích hoạt Thiên Nhãn (Dự đoán) nếu Ghost ở xa và Model đã nạp
            if dist_to_ghost > 5 and self.ml_model is not None:
                # NẾU ĐANG BỊ KHÓA MỤC TIÊU: Giữ nguyên quyết định cũ để tránh nhảy qua nhảy lại
                if self.lock_counter > 0 and self.locked_target is not None:
                    target_pos = self.locked_target
                    self.lock_counter -= 1
                else:
                    # NẾU HẾT KHÓA: Bắt đầu dự đoán mục tiêu mới
                    input_data = np.array([[my_position[0], my_position[1], enemy_position[0], enemy_position[1]]])
                    prediction = self.ml_model.predict(input_data)[0] 
                    pred_r, pred_c = map(int, prediction.split('_'))
                    predicted_pos = (pred_r, pred_c)
                    
                    if self._is_valid_position(predicted_pos, map_state):
                        target_pos = predicted_pos 
                        # KHÓA MỤC TIÊU LẠI: Bắt buộc Pacman phải theo hướng này trong 3 bước tới
                        self.locked_target = target_pos
                        self.lock_counter = 3
            else:
                # Nếu đã áp sát Ghost (khoảng cách <= 5), mở khóa ngay lập tức để cắn xé
                self.lock_counter = 0
                self.locked_target = None

            # 3. GỌI THUẬT TOÁN TÌM ĐƯỜNG (A*)
            if self.algorithm == "ASTAR":
                path = pacmanAlgorithm.astar(my_position, target_pos, map_state)
                
                # HỆ THỐNG AN TOÀN (FALLBACK)
                # Nếu model ML dự đoán vào một góc chết không thể tới, lập tức quay lại đuổi Ghost
                if not path and target_pos != enemy_position:
                    path = pacmanAlgorithm.astar(my_position, enemy_position, map_state)
            
            elif self.algorithm == "BFS":
                path = pacmanAlgorithm.bfs(my_position, enemy_position, map_state)
            elif self.algorithm == "GREEDY":
                path = pacmanAlgorithm.greedy_search(my_position, enemy_position, map_state)
            else:
                path = []
                
            # 4. LOGIC DI CHUYỂN TỐI ƯU
            if path:
                best_move = path[0]
                
                # CẬP NHẬT: Dồn góc khi áp sát
                if len(path) == 1:
                    desired_steps = 1
                else:
                    desired_steps = 1
                    for i in range(1, len(path)):
                        if path[i] == best_move:
                            desired_steps += 1
                        else:
                            break
                            
                    # Tăng tốc độ tối đa để nhanh chóng thu hẹp khoảng cách
                    if desired_steps == len(path) and desired_steps < self.pacman_speed:
                        desired_steps = self.pacman_speed
                        
                steps = self._max_valid_steps(my_position, best_move, map_state, desired_steps)
                if steps > 0:
                    return (best_move, steps)
                    
            return (Move.STAY, 1)

        # TÚI KHÍ BẢO VỆ CUỐI CÙNG
        except Exception as e:
            print(f"[CẢNH BÁO TỚI MẠNG] Pacman lỗi tại bước {step_number}: {e}")
            fallback_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
            for move in fallback_moves:
                delta_row, delta_col = move.value
                next_pos = (my_position[0] + delta_row, my_position[1] + delta_col)
                if self._is_valid_position(next_pos, map_state):
                    return (move, 1)
            return (Move.STAY, 1)
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0

    def _max_valid_steps(self, pos: tuple, move: Move, map_state: np.ndarray, desired_steps: int) -> int:
        steps = 0
        max_steps = min(self.pacman_speed, max(1, desired_steps))
        current = pos
        for _ in range(max_steps):
            delta_row, delta_col = move.value
            next_pos = (current[0] + delta_row, current[1] + delta_col)
            if not self._is_valid_position(next_pos, map_state):
                break
            steps += 1
            current = next_pos
        return steps

    def _desired_steps(self, move: Move, row_diff: int, col_diff: int) -> int:
        if move in (Move.UP, Move.DOWN):
            return abs(row_diff)
        if move in (Move.LEFT, Move.RIGHT):
            return abs(col_diff)
        return 1


class GhostAgent(BaseGhostAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Smart Minimax Ghost"
        self.last_move = None # lưu vị trí đã đi để phạt nếu quay đầu


    def _random_move(self, pos: tuple, map_state: np.ndarray) -> tuple:
        moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        valid_moves = []

        for move in moves:
            dx, dy = move.value
            new_pos = (pos[0] + dx, pos[1] + dy)
            if self._is_valid_position(new_pos, map_state):
                valid_moves.append(new_pos)

    # nếu không có đường đi thì đứng yên
        if not valid_moves:
           return pos

        return random.choice(valid_moves)


    def monte_carlo(self, ghost_pos, pacman_pos, map_state, simulations=100):
        """thực hiện một số mô phỏng đoán trước step của pacman"""
        if simulations<=0:
            return 0

        total_distance =0
        for _ in range(simulations):
            sim_ghost_pos = ghost_pos
            sim_pacman_pos=pacman_pos
             
            for _ in range(10):# mô phỏng 10 bước đi của pacman
                sim_ghost_pos = self._random_move(sim_ghost_pos, map_state)
                sim_pacman_pos = self._random_move(sim_pacman_pos, map_state)

            dist_map = self._compute_distance_map(sim_pacman_pos, map_state)
            dist = dist_map.get(sim_ghost_pos, 0)
            total_distance += dist


        return total_distance/simulations

       

    def minimax(self, ghost_pos, pacman_pos, map_state, depth, ghost_turn):
        # Base case: Hết độ sâu dự đoán
        if depth == 0:
            # Thay vì dùng Manhattan, ta gọi BFS nhanh từ pacman_pos hiện tại
            # để biết khoảng cách THỰC TẾ đến ghost_pos
            dist_map = self._compute_distance_map(pacman_pos, map_state)
            return dist_map.get(ghost_pos, -1) # Trả về số bước đi thực tế

        if ghost_turn:
            best = -float("inf") # Lượt Ghost: Muốn khoảng cách max, khởi tạo vs gtri nhỏ nhất
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dx, dy = move.value
                new_pos = (ghost_pos[0] + dx, ghost_pos[1] + dy)
                if self._is_valid_position(new_pos, map_state):
                    score = self.minimax(new_pos, pacman_pos, map_state, depth-1, False) #True là lượt ghost,False là pacman
                    best = max(best, score)
            # Nếu bị kẹt không có đường đi hợp lệ
            return best if best != -float("inf") else -1

        else:
            best = float("inf") # Lượt Pacman: Muốn khoảng cách min
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dx, dy = move.value
                new_pos = (pacman_pos[0] + dx, pacman_pos[1] + dy)
                if self._is_valid_position(new_pos, map_state):
                    score = self.minimax(ghost_pos, new_pos, map_state, depth-1, True)
                    best = min(best, score)
            return best if best != float("inf") else -1

        # tóm lại thuật toán minimax dự đoán lượt đi của ghost và pacman,sau đó sẽ đánh giá đưa ra nước đi tốt nhất cho ghost
        # dựa trên 2 lượt đi của pacman và ghost




    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:

        best_move = Move.STAY
        best_score = -float("inf")

        dist_now = self._compute_distance_map(enemy_position, map_state).get(my_position, 0)

        # Duyệt 4 hướng di chuyển hiện tại của Ghost
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dx, dy = move.value
            new_pos = (my_position[0] + dx, my_position[1] + dy)# tính vị trí mới nếu ghost đi htai

            if self._is_valid_position(new_pos, map_state):
                # Gọi Minimax với độ sâu 2 (1 lượt Ghost đi, 1 lượt Pacman đi)
                score = self.minimax(new_pos, enemy_position, map_state, 2, False)
                #monte carlo dự đoán 10 step của pacman sau đó tính kcach tb
                mc_score = self.monte_carlo(new_pos, enemy_position, map_state, simulations =50)
                score += 0.3 * mc_score # kết hợp điểm minimax và điểm monte carlo để xem xét các bước đi

                # nếu đi xa hơn
                dist_new= self._compute_distance_map(enemy_position, map_state). get(new_pos, 0)
                if dist_new > dist_now:
                    score +=1


                # nếu quay đầu
                if self.last_move:
                    opposite = {
                        Move.UP: Move.DOWN,
                        Move.DOWN: Move.UP,
                        Move.LEFT:Move.RIGHT,
                        Move.RIGHT:Move.LEFT
                        }
                    if move == opposite.get(self.last_move):
                        score -= 2 # phạt nếu quay đầu
                if score > best_score:
                    best_score = score
                    best_move = move


        self.last_move = best_move

        return best_move
                 
    def _compute_distance_map(self, start_pos: tuple, map_state: np.ndarray) -> dict:
        """Dùng BFS để tính khoảng cách từ một điểm đến toàn bộ các ô khác."""
        queue = deque([(start_pos, 0)]) 
        visited = {start_pos: 0}#lưu khoảng cách những ô đã đi qua và khoảng cách tới ô đó
        height, width = map_state.shape # kích thước bản đồ (lun mặc định rồi)
        
        while queue:
            (curr_row, curr_col), dist = queue.popleft()
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                nr, nc = curr_row + dr, curr_col + dc
                
                if (0 <= nr < height and 0 <= nc < width and # lấy dki: ko chạm tường,ko vượt map,ko đi vào ô đã đi
                    map_state[nr, nc] == 0 and (nr, nc) not in visited):
                    
                    visited[(nr, nc)] = dist + 1
                    queue.append(((nr, nc), dist + 1))#lưu kcach mới đã đi
        return visited
      # như cũ kiểm tra xem vị trí mới có hợp lệ vượt map hay chạm tường ko  
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        row, col = pos
        h, w = map_state.shape
        return 0 <= row < h and 0 <= col < w and map_state[row, col] == 0

       
