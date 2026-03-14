
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
        self.algorithm = "RANDOM"  # Options: "ASTAR", "BFS", "DFS", "GREEDY", "RANDOM"
    
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


   def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        self.name = "Example Evasive Ghost"

    
   def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
             
        
        # gọi bfs
        target = self.BFS(my_position, enemy_position, map_state)
        x, y = my_position
        height, width = map_state.shape 
        best_direction = None
        max_distance = -1



        #ktra hướng có chạm tường ko     
        move =[Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]

        for move in move:
            delta_row, delta_col = move.value
            new_x=x+ delta_row
            new_y=y+delta_col
            if 0 <= new_x< height and 0<= new_y <width and map_state[new_x][new_y]==0:
                distance = abs(new_x - enemy_position[0])+ abs(new_y - enemy_position[1])

                distance = abs(enemy_position[0] - new_x) + abs(enemy_position[1] - new_y)
                if distance>max_distance:
                    max_distance=distance
                    best_direction=move

        
        if best_direction:
         return best_direction

        return Move.STAY



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
    
   def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0               

    # BFS tìm vị trí xa nhất cho ghost so vs pacman
   def BFS(self, my_position, enemy_position, map_state):

        queue = deque([my_position])
        visited = set([my_position])

        best_pos = my_position
        best_dist = abs(my_position[0] - enemy_position[0]) + abs(my_position[1] - enemy_position[1])

        while queue:

            current = queue.popleft()

            distance = abs(current[0] - enemy_position[0]) + abs(current[1] - enemy_position[1])

            if distance > best_dist:
               best_dist = distance
               best_pos = current

    # BFS explore map
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:

               delta_row, delta_col = move.value
               new_pos = (current[0] + delta_row, current[1] + delta_col)

               if self._is_valid_position(new_pos, map_state) and new_pos not in visited:

                 visited.add(new_pos)
                 queue.append(new_pos)
               


        return best_pos



       
