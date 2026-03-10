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
    def step ( self , map_state , my_pos , enemy_pos , step_number ) :


            return Move . UP | Move . DOWN | Move . LEFT | Move . RIGHT | Move . STAY
    
    def __init__(self, **kwargs):
        """
        Initialize the Pacman agent.
        Students can set up any data structures they need here.
        """
        super().__init__(**kwargs)
        self.name = "Example Greedy Pacman"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        """
        Simple greedy strategy: move towards the ghost.
        
        Students should implement better search algorithms like:
        - BFS (Breadth-First Search)
        - DFS (Depth-First Search)
        - A* Search
        - Greedy Best-First Search
        - etc.
        """
        # Calculate direction to ghost
        row_diff = enemy_position[0] - my_position[0]
        col_diff = enemy_position[1] - my_position[1]
        
        # List of possible moves in order of preference
        moves = []
        
        # Prioritize vertical movement if needed
        if row_diff > 0:
            moves.append(Move.DOWN)
        elif row_diff < 0:
            moves.append(Move.UP)
        
        # Prioritize horizontal movement if needed
        if col_diff > 0:
            moves.append(Move.RIGHT)
        elif col_diff < 0:
            moves.append(Move.LEFT)
        
        # Try each move in order
        for move in moves:
            desired_steps = self._desired_steps(move, row_diff, col_diff)
            steps = self._max_valid_steps(my_position, move, map_state, desired_steps)
            if steps > 0:
                return (move, steps)
        
        # If no preferred move is valid, try any valid move
        all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(all_moves)
        
        for move in all_moves:
            steps = self._max_valid_steps(my_position, move, map_state, self.pacman_speed)
            if steps > 0:
                return (move, steps)
        
        # If no move is valid, stay
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

        import random
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
        if best_direction == 1:
            return (Move.RIGHT, 1)
        elif best_direction == 2:
            return (Move.DOWN, 1)
        elif best_direction == 3:
            return (Move.LEFT, 1)
        elif best_direction == 4: 
            return (Move.UP, 1)

        from collections import deque



    # BFS tìm vị trí xa nhất cho ghost so vs pacman
   def BFS(self, start, pacman, map_state):

        queue = deque([start])
        visited = set([start])

        best_pos = start
        best_dist = abs(start[0] - pacman[0]) + abs(start[1] - pacman[1])

        while queue:

         current = queue.popleft()

         # calculate distance between ghost and pacman
         distance = abs(current[0] - pacman[0]) + abs(current[1] - pacman[1])

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
        
        # List of possible moves in order of preference
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
