"""
Example student submission showing the required interface.

Students should implement their own PacmanAgent and/or GhostAgent
following this template.
"""

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


class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        """
        Initialize the Pacman agent.
        Students can set up any data structures they need here.
        """
        super().__init__(**kwargs)
        self.name = "BFS Pacman"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        path = self.bfs(my_position, enemy_position, map_state)
        
        if path:
            best_move = path[0]
            desired_steps = 1
            
            # Kiểm tra xem có thể đi 2 bước trên cùng một đường thẳng không
            if len(path) > 1 and path[0] == path[1]:
                desired_steps = 2
                
            # Đảm bảo an toàn (không đâm vào tường)
            steps = self._max_valid_steps(my_position, best_move, map_state, desired_steps)
            
            if steps > 0:
                return (best_move, steps)
                
        # Nếu không có đường đi (bị kẹt), đứng im
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

    def _get_neighbors(self, pos: tuple, map_state: np.ndarray) -> list:
        """Lấy danh sách các ô hợp lệ xung quanh vị trí hiện tại."""
        neighbors = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            delta_row, delta_col = move.value
            next_pos = (pos[0] + delta_row, pos[1] + delta_col)
            
            if self._is_valid_position(next_pos, map_state):
                neighbors.append((next_pos, move))
        return neighbors

    def bfs(self, start: tuple, goal: tuple, map_state: np.ndarray) -> list:
        queue = [(start, [])]
        visited = {start}
        
        while queue:
            # Dùng pop(0) để lấy phần tử đầu tiên của list (hoạt động như queue)
            current_pos, path = queue.pop(0)
            
            # Đã tìm thấy Ghost
            if current_pos == goal:
                return path
                
            # Khám phá các ô lân cận
            for next_pos, move in self._get_neighbors(current_pos, map_state):
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [move]))
                    
        return []


class GhostAgent(BaseGhostAgent):


       def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        self.name = "Example Evasive Ghost"

    
   def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
             
        import random
        # gọi bfs
        self.BFS(my_position, enemy_position, map_state)
        x, y = my_position
        height, width = map_state.shape 
        best_direction = None
        max_distance = -1



        #ktra hướng có chạm tường ko     
        move =[Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        move = random.choice(move)

        #ktra xem các hướng ngẫu nhiên có hướng nào chạm tường ko, nếu  không thì tính kcah vs pacman, chọn hướng xa nhất

        if move == Move.UP:

            new_x = x - 1
            new_y = y

            if new_x >= 0 and map_state[new_x][new_y] == 0:
                distance = abs(enemy_position[0] - new_x) + abs(enemy_position[1] - new_y)

                if distance > max_distance:
                    max_distance = distance
                    best_direction = 4

        elif move == Move.DOWN:

            new_x = x+1
            new_y = y

            if new_x < height and map_state[new_x][new_y] == 0:
                distance = abs(enemy_position[0] - new_x) + abs(enemy_position[1] - new_y)

                if distance > max_distance:
                    max_distance=distance
                    best_direction = 2

        elif move == Move.LEFT:

            new_x=x 
            new_y=y-1

            if new_y >= 0 and map_state[new_x][new_y] == 0:
                distance = abs(enemy_position[0] - new_x) + abs(enemy_position[1] - new_y)
                if distance > max_distance:
                    max_distance = distance
                    best_direction = 3

        elif move== Move.RIGHT:

            new_x=x 
            new_y=y+1

            if new_y < width and map_state[new_x][new_y] == 0:
                distance =abs(enemy_position[0] - new_x) + abs(enemy_position[1] - new_y)
                if distance > max_distance:
                    max_distance = distance
                    best_direction =1

        

         #nếu ko có hướng nào hợp lệ sẽ đứng yên, còn có sẽ chọn đi hướng đó           
        #nếu ko có hướng nào hợp lệ sẽ đứng yên, còn có sẽ chọn đi hướng đó           
        if best_direction == 1:
          return Move.RIGHT
        elif best_direction == 2:
          return Move.DOWN
        elif best_direction == 3:
          return Move.LEFT
        elif best_direction == 4: 
           return Move.UP

# fallback nếu không có hướng hợp lệ
        return Move.STAY

       



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
