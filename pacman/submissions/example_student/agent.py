
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
        """
        Initialize the Pacman agent.
        Students can set up any data structures they need here.
        """
        super().__init__(**kwargs)
        self.name = "Choose Search Pacman"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))

        #Chọn thuật toán sẽ sử dụng:
        self.algorithm = "ASTAR"  # Options: "ASTAR", "BFS", "DFS", "GREEDY", "RANDOM"
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        
        try:
            # Chọn thuật toán tương ứng
            if self.algorithm == "ASTAR":
                path = pacmanAlgorithm.astar(my_position, enemy_position, map_state)
            elif self.algorithm == "BFS":
                path = pacmanAlgorithm.bfs(my_position, enemy_position, map_state)
            elif self.algorithm == "DFS":
                path = pacmanAlgorithm.dfs(my_position, enemy_position, map_state)
            elif self.algorithm == "GREEDY":
                path = pacmanAlgorithm.greedy_search(my_position, enemy_position, map_state)
            elif self.algorithm == "RANDOM":
                path = pacmanAlgorithm.random_search(my_position, map_state)
            else:
                path = []
                
            if path:
                best_move = path[0]
                if len(path) == 1:
                    # Lấy số micro-giây hiện tại chia lấy dư cho 100 để tạo tỷ lệ phần trăm
                    rand_percent = int(time.time() * 1000000) % 100
                    
                    # 70% xác suất Pacman phóng lố (đề phòng Ghost chạy thẳng)
                    if rand_percent < 70:
                        steps = self._max_valid_steps(my_position, best_move, map_state, self.pacman_speed)
                        if steps > 0:
                            return (best_move, steps)
                    # 30% xác suất đứng yên, chờ Ghost "tự hủy" chui đầu vào rọ
                    else:
                        return (Move.STAY, 1)
                    
                desired_steps = 1
                
                # Đếm số bước liên tiếp trên cùng một hướng
                for i in range(1, len(path)):
                    if path[i] == best_move:
                        desired_steps += 1
                    else:
                        break
                        
                # Tuyệt chiêu phóng lố đón đầu (Lunge)
                if desired_steps == len(path) and desired_steps < self.pacman_speed:
                    desired_steps = self.pacman_speed
                    
                steps = self._max_valid_steps(my_position, best_move, map_state, desired_steps)
                if steps > 0:
                    return (best_move, steps)
                    
            return (Move.STAY, 1)

        
        except Exception as e:
            # Việc in này không làm sập framework Arena
            #print(f"[CẢNH BÁO TỚI MẠNG] Pacman lỗi tại bước {step_number}: {e}")
            
            # Cố gắng tìm một hướng bất kỳ không bị vướng tường để lách qua
            fallback_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
            
            for move in fallback_moves:
                delta_row, delta_col = move.value
                next_pos = (my_position[0] + delta_row, my_position[1] + delta_col)
                
                if self._is_valid_position(next_pos, map_state):
                    # Nếu thấy đường trống, đi đại 1 bước
                    return (move, 1)
                    
            # 3. Nếu cả 4 bề đều là tường (hoặc lỗi map), đành đứng im chịu trận
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

    def minimax(self, ghost_pos, pacman_pos, map_state, depth, ghost_turn):
        # Base case: Hết độ sâu dự đoán
        if depth == 0:
            # Thay vì dùng Manhattan, ta gọi BFS nhanh từ pacman_pos hiện tại
            # để biết khoảng cách THỰC TẾ đến ghost_pos
            dist_map = self._compute_distance_map(pacman_pos, map_state)
            return dist_map.get(ghost_pos, -1) # Trả về số bước đi thực tế

        if ghost_turn:
            best = -float("inf") # Lượt Ghost: Muốn khoảng cách max
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dx, dy = move.value
                new_pos = (ghost_pos[0] + dx, ghost_pos[1] + dy)
                if self._is_valid_position(new_pos, map_state):
                    score = self.minimax(new_pos, pacman_pos, map_state, depth-1, False)
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

    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:

        best_move = Move.STAY
        best_score = -float("inf")

        # Duyệt 4 hướng di chuyển hiện tại của Ghost
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dx, dy = move.value
            new_pos = (my_position[0] + dx, my_position[1] + dy)

            if self._is_valid_position(new_pos, map_state):
                # Gọi Minimax với độ sâu 2 (1 lượt Ghost đi, 1 lượt Pacman đi)
                score = self.minimax(new_pos, enemy_position, map_state, 2, False)

                if score > best_score:
                    best_score = score
                    best_move = move

        return best_move
                 
    def _compute_distance_map(self, start_pos: tuple, map_state: np.ndarray) -> dict:
        """Dùng BFS để tính khoảng cách từ một điểm đến toàn bộ các ô khác."""
        queue = deque([(start_pos, 0)]) 
        visited = {start_pos: 0}
        height, width = map_state.shape 
        
        while queue:
            (curr_row, curr_col), dist = queue.popleft()
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                nr, nc = curr_row + dr, curr_col + dc
                
                if (0 <= nr < height and 0 <= nc < width and 
                    map_state[nr, nc] == 0 and (nr, nc) not in visited):
                    
                    visited[(nr, nc)] = dist + 1
                    queue.append(((nr, nc), dist + 1))
        return visited
        
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        row, col = pos
        h, w = map_state.shape
        return 0 <= row < h and 0 <= col < w and map_state[row, col] == 0


       
