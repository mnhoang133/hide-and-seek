
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
        
        # Gọi thuật toán dựa trên lựa chọn (Nộp bài chọn 1 thuật toán thôi)
        if self.algorithm == "ASTAR":
            path = pacmanAlgorithm.astar(my_position, enemy_position, map_state)
        elif self.algorithm == "BFS":
            path = pacmanAlgorithm.bfs(my_position, enemy_position, map_state)
        elif self.algorithm == "DFS":
            path = pacmanAlgorithm.dfs(my_position, enemy_position, map_state)
        elif self.algorithm == "GREEDY":
            path = pacmanAlgorithm.greedy_search(my_position, enemy_position, map_state)
        else:  # RANDOM
            path = pacmanAlgorithm.random_search(my_position, map_state)
        
        if path:
            best_move = path[0]
            desired_steps = 1
            
            # Đếm số bước liên tiếp trên cùng một hướng
            for i in range(1, len(path)):
                if path[i] == best_move:
                    desired_steps += 1
                else:
                    break
                    
            # Tuyệt chiêu phóng lố (Lunge) đón đầu Ghost
            if desired_steps == len(path) and desired_steps < self.pacman_speed:
                desired_steps = self.pacman_speed
                
            steps = self._max_valid_steps(my_position, best_move, map_state, desired_steps)
            if steps > 0:
                return (best_move, steps)
                
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

    #gọi minimax để dự đoán hướng đi của pacman, sau đó đưa ra hướng tránh xa nhất trên map
   def minimax(self, ghost_pos, pacman_pos, map_state, depth, ghost_turn):
    #base case: khi dự đoán đủ bước thì trả về khoảng ca1h giữa ghost và pacman
    if depth == 0:
        return abs(ghost_pos[0] - pacman_pos[0]) + abs(ghost_pos[1] - pacman_pos[1])

    if ghost_turn:

        best = -float("inf") # chọn kcach xa nhất
         # dự đoán text 4 hướng đi, dự đoán hướng đi của pacman
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:

            dx, dy = move.value
            new_pos = (ghost_pos[0] + dx, ghost_pos[1] + dy)
            # ko đi ngoài map và đụng tường
            if self._is_valid_position(new_pos, map_state):
                # gọi đệ quy để dự đoán hướng đi tiếp của pacman
                score = self.minimax(new_pos, pacman_pos, map_state, depth-1, False)

                best = max(best, score)

        return best

    else:
        # lược pacman 
        best = float("inf")

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:

            dx, dy = move.value
            new_pos = (pacman_pos[0] + dx, pacman_pos[1] + dy)

            if self._is_valid_position(new_pos, map_state):

                score = self.minimax(ghost_pos, new_pos, map_state, depth-1, True)

                best = min(best, score)

        return best

   def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        self.name = "Example Evasive Ghost"

    
   def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:


       best_move = None
       best_score = -float("inf")

       for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:

            dx, dy = move.value
            new_pos = (my_position[0] + dx, my_position[1] + dy)

            if self._is_valid_position(new_pos, map_state):

                score = self.minimax(new_pos, enemy_position, map_state, 2, False)

                if score > best_score:
                    best_score = score
                    best_move = move

       if best_move:
            return best_move
             
        
        # gọi bfs, xác định vị trí của th pacman, sau đó chôn hướng ngc lại xa nhất
       dist_from_enemy = self._compute_distance_map(enemy_position, map_state)



       x, y = my_position
       height, width = map_state.shape 
       best_direction = None
       max_dist = -1



        #ktra hướng có chạm tường ko     , và hướng có khả năng di chuyển cho hide
       possible_moves =[Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]

       for move in possible_moves:
            delta_row, delta_col = move.value
            new_x=x+ delta_row
            new_y=y+delta_col

            #kiểm tra nếu hướng đi hợp lệ ko chạm tường
            if self._is_valid_position((new_x, new_y), map_state):
                # Lấy khoảng cách thực tế từ BFS (nếu ô này ko đến được thì mặc định là -1)
                actual_dist = dist_from_enemy.get((new_x, new_y), -1)

                
                if actual_dist > max_dist:
                    max_dist = actual_dist
                    best_direction = move

        
       if best_direction:
            return best_direction

  



         # Calculate direction away from Pacman
       row_diff = my_position[0] - enemy_position[0]
       col_diff = my_position[1] - enemy_position[1]
        
        # List of posible moves in order of preference
       moves = []
        
        # Prioritize vertical movement away from Pacman
       if row_diff > 0:
            moves.append(Move.DOWN)
       elif row_diff < 0:
            moves.append(Move.UP)
        
        # Prioritize horizontal movement away from Pacman
       if col_diff > 0:
            moves.append(Move.RIGHT)
       elif col_diff < 0:
            moves.append(Move.LEFT)
        
        # Try each move in order
       for move in moves:
            delta_row, delta_col = move.value
            new_pos = (my_position[0] + delta_row, my_position[1] + delta_col)
            
            # Check if move is valid
            if self._is_valid_position(new_pos, map_state):
                return move
        
        # If no preferred move is valid, try any valid move
       all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
       random.shuffle(all_moves)
        
       for move in all_moves:
            delta_row, delta_col = move.value
            new_pos = (my_position[0] + delta_row, my_position[1] + delta_col)
            
            if self._is_valid_position(new_pos, map_state):
                return move
        
        # If no move is valid, stay
       return Move.STAY
                 
    # BFS tìm vị trí xa nhất cho ghost so vs pacman
   def _compute_distance_map(self, start_pos: tuple, map_state: np.ndarray) -> dict:
        """Dùng BFS để tính khoảng cách từ một điểm đến toàn bộ các ô khác."""


        queue = deque([(start_pos, 0)]) # (vị trí, khoảng cách) # dnah sách các ô cần duyệt và lưu khaong cach tính
        visited = {start_pos: 0}
        
        height, width = map_state.shape 
        # lặp bfs để check từng lớp
        while queue:
            (curr_row, curr_col), dist = queue.popleft()
            # check 4 hướng
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                # tìm vị trí mới
                dr, dc = move.value
                nr, nc = curr_row + dr, curr_col + dc
                
                if (0 <= nr < height and 0 <= nc < width and 
                    map_state[nr, nc] == 0 and (nr, nc) not in visited):
                    
                    visited[(nr, nc)] = dist + 1
                    queue.append(((nr, nc), dist + 1))
        return visited
    # thuật toán ktra vị trí hợp lệ
   def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        row, col = pos
        h, w = map_state.shape
        return 0 <= row < h and 0 <= col < w and map_state[row, col] == 0


       
