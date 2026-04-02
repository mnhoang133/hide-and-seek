"""
Example student submission showing the required interface.

Students should implement their own PacmanAgent and/or GhostAgent
following this template.
"""

import sys
from pathlib import Path
from collections import deque

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
    """
    Pacman (Seeker) Agent - Goal: Catch the Ghost
    
    Implement your search algorithm to find and catch the ghost.
    Suggested algorithms: BFS, DFS, A*, Greedy Best-First
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        # TODO: Initialize any data structures you need
        # Examples:
        # - self.path = []  # Store planned path
        # - self.visited = set()  # Track visited positions

        self.name = "Fog Walker Pacman"

        # Memory for limited observation mode
        self.last_known_enemy_pos = None
        self.internal_map = np.full((21, 21), -1, dtype=int)  # -1 = unseen, 0 = empty, 1 = wall

        # Dồn góc
        self.locked_target = None
        self.lock_counter = 0
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        """
        Decide the next move.
        
        Args:
            map_state: 2D numpy array where 1=wall, 0=empty, -1=unseen (fog)
            my_position: Your current (row, col) in absolute coordinates
            enemy_position: Ghost's (row, col) if visible, None otherwise
            step_number: Current step number (starts at 1)
            
        Returns:
            Move or (Move, steps): Direction to move (optionally with step count)
        """
        # TODO: Implement your search algorithm here
        
        try:
            # Cập nhật bộ nhớ bản đồ với thông tin mới nhất
            self._update_memory(map_state)
            
            # Tính toán hướng đi (quán tính) của Ghost
            ghost_dir = None
            if enemy_position is not None:
                if self.last_known_enemy_pos is not None and self.last_known_enemy_pos != enemy_position:
                    ghost_dir = (enemy_position[0] - self.last_known_enemy_pos[0], 
                                 enemy_position[1] - self.last_known_enemy_pos[1])
                self.last_known_enemy_pos = enemy_position

            # Nếu đã chạy tới nơi nghi ngờ mà không thấy -> xóa dấu vết
            if my_position == self.last_known_enemy_pos:
                self.last_known_enemy_pos = None

            # Trạng thái hành động
            target_pos = None
            is_chasing = False

            if enemy_position is not None:
                # 1. Khi thấy Ghost: Đuổi theo trực tiếp
                is_chasing = True
                target_pos = enemy_position
                
                # Chiến thuật dồn góc chặn ngã tư (Chỉ kích hoạt sau step 50)
                if step_number > 50 and ghost_dir is not None:
                    if self.lock_counter > 0 and self.locked_target is not None:
                        target_pos = self.locked_target
                        self.lock_counter -= 1
                        # Nếu đến điểm chặn rồi thì mở khóa sớm
                        if my_position == self.locked_target:
                            self.lock_counter = 0
                            self.locked_target = None
                            target_pos = enemy_position
                    else:
                        choke_point = self._get_forward_choke_point(enemy_position, ghost_dir, self.internal_map)
                        target_pos = choke_point
                        if choke_point != enemy_position:
                            self.locked_target = choke_point
                            self.lock_counter = 3
                            
            elif self.last_known_enemy_pos is not None:
                # Mất dấu nhưng còn nhớ vị trí cuối -> Đi thẳng về hướng đó
                target_pos = self.last_known_enemy_pos
                self.lock_counter = 0
                self.locked_target = None
                
            else:
                # Khám phá bản đồ khi chưa thấy gì
                target_pos = self._find_closest_frontier(my_position)
                self.lock_counter = 0
                self.locked_target = None

            # Sử dụng A* để tìm đường đến target_pos
            path = []
            if target_pos is not None:
                path = pacmanAlgorithm.astar(my_position, target_pos, self.internal_map)
                
                # Fallback: Nếu đường chặn ngã tư bị kẹt, quay lại đuổi thẳng mặt
                if not path and is_chasing and target_pos != enemy_position:
                    target_pos = enemy_position
                    self.lock_counter = 0
                    self.locked_target = None
                    path = pacmanAlgorithm.astar(my_position, enemy_position, self.internal_map)

            if path:
                best_move = path[0]
                
                # Đếm số bước đi thẳng liên tiếp trên path
                consecutive_steps = 1
                for i in range(1, len(path)):
                    if path[i] == best_move:
                        consecutive_steps += 1
                    else:
                        break
                
                desired_steps = min(consecutive_steps, self.pacman_speed)
                steps = self._max_valid_steps(my_position, best_move, self.internal_map, desired_steps)
                
                if steps > 0:
                    return (best_move, steps)
                    
            # Fallback cuối cùng nếu A* không tìm được đường
            fallback_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
            for move in fallback_moves:
                if self._is_valid_move(my_position, move, self.internal_map):
                    return (move, 1)
            
            return (Move.STAY, 1)

        except Exception as e:
            print(f"LỖI PACMAN: {e}")
            return (Move.STAY, 1)
        
    # Helper methods
    
    def _update_memory(self, map_state: np.ndarray):
        """Cập nhật bộ nhớ siêu tốc bằng np.where"""
        self.internal_map = np.where(map_state != -1, map_state, self.internal_map)

    def _get_forward_choke_point(self, ghost_pos, ghost_dir, map_state):
        """Phóng tia Raycast tìm ngã tư chặn đầu Ghost"""
        curr_pos = ghost_pos
        for _ in range(7): 
            next_pos = (curr_pos[0] + ghost_dir[0], curr_pos[1] + ghost_dir[1])
            if not self._is_valid_position(next_pos, map_state):
                return curr_pos 
            
            valid_neighbors = 0
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                n = (next_pos[0] + move.value[0], next_pos[1] + move.value[1])
                if self._is_valid_position(n, map_state):
                    valid_neighbors += 1
            if valid_neighbors > 2:
                return next_pos # Tìm thấy ngã tư
            curr_pos = next_pos
        return curr_pos

    def _find_closest_frontier(self, my_pos):
        """Dùng BFS tìm ô sáng sát cạnh ô sương mù (-1) gần nhất"""
        queue = deque([(my_pos, [])])
        visited = {my_pos}
        height, width = self.internal_map.shape
        
        while queue:
            curr_pos, path = queue.popleft()
            
            # Kiểm tra xem có giáp vùng tối (-1) không
            is_frontier = False
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = curr_pos[0] + dr, curr_pos[1] + dc
                if 0 <= nr < height and 0 <= nc < width:
                    if self.internal_map[nr, nc] == -1:
                        is_frontier = True
                        break
                        
            if is_frontier:
                return curr_pos
                
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                next_pos = (curr_pos[0] + dr, curr_pos[1] + dc)
                if self._is_valid_position(next_pos, self.internal_map) and next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [move]))
        return None
    
    # Default Helper methods (you can add more)
    
    def _choose_action(self, pos: tuple, moves, map_state: np.ndarray, desired_steps: int):
        for move in moves:
            max_steps = min(self.pacman_speed, max(1, desired_steps))
            steps = self._max_valid_steps(pos, move, map_state, max_steps)
            if steps > 0:
                return (move, steps)
        return None

    def _max_valid_steps(self, pos: tuple, move: Move, map_state: np.ndarray, max_steps: int) -> int:
        steps = 0
        current = pos
        for _ in range(max_steps):
            delta_row, delta_col = move.value
            next_pos = (current[0] + delta_row, current[1] + delta_col)
            if not self._is_valid_position(next_pos, map_state):
                break
            steps += 1
            current = next_pos
        return steps
    
    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        """Check if a move from pos is valid for at least one step."""
        return self._max_valid_steps(pos, move, map_state, 1) == 1
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0


class GhostAgent(BaseGhostAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Ghost Fixed"
        self.last_move = None
        self.prev_positions = []  # lưu lịch sử vị trí để phá loop

        self.known_map = np.full((21, 21), -1)
        self.last_known_enemy_pos = None


    def _explore(self, pos):
        best_move = Move.STAY
        best_score = -float("inf")

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dx, dy = move.value
            new_pos = (pos[0] + dx, pos[1] + dy)

            # không đi vào tường / ngoài map
            if not self._is_valid_position(new_pos):
                continue

            score = 0

            # ưu tiên ô chưa khám phá
            if self.known_map[new_pos[0]][new_pos[1]] == -1:
                score += 5

            # ưu tiên nhiều đường thoát
            escape_routes = self._count_valid_moves(new_pos)
            score += 2 * escape_routes

            # phạt ngõ cụt
            if escape_routes <= 1:
                score -= 10

            # tránh quay đầu
            if self.last_move is not None:
                opposite = {
                    Move.UP: Move.DOWN,
                    Move.DOWN: Move.UP,
                    Move.LEFT: Move.RIGHT,
                    Move.RIGHT: Move.LEFT,
                }
                if move == opposite.get(self.last_move):
                    score -= 5

            # phá loop mạnh: tránh quay lại vị trí cũ
            if new_pos in self.prev_positions[-4:]:
                score -= 15

            # random nhẹ
            score += random.random()

            if score > best_score:
                best_score = score
                best_move = move

        return best_move


    def step(self, map_state, my_position, enemy_position, step_number):

        # cập nhật map đã thấy
        self._update_memory(map_state)

        # lưu vị trí đã đi để tránh loop
        self.prev_positions.append(my_position)
        if len(self.prev_positions) > 6:
            self.prev_positions.pop(0)

        # lưu enemy nếu nhìn thấy
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position

        threat = enemy_position or self.last_known_enemy_pos

        # không thấy enemy -> explore
        if threat is None:
            return self._explore(my_position)

        best_move = Move.STAY
        best_score = -float("inf")

        # BFS từ pacman để biết khoảng cách thật
        dist_map = self._compute_distance_map(threat)

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dx, dy = move.value
            new_pos = (my_position[0] + dx, my_position[1] + dy)

            # không hợp lệ thì bỏ
            if not self._is_valid_position(new_pos):
                continue

            # tránh chết ngay
            if self._is_dead(new_pos, threat):
                continue

            # nếu BFS không tới được -> bỏ (tránh đi lung tung gây đứng)
            dist = dist_map.get(new_pos, -1)
            if dist == -1:
                continue

            escape_routes = self._count_valid_moves(new_pos)

            score = 0

            # càng xa pacman càng tốt
            score += 4 * dist

            # nhiều đường thoát càng tốt
            score += 2 * escape_routes

            # bonus khám phá
            if self.known_map[new_pos[0]][new_pos[1]] == -1:
                score += 4

            # gần quá thì phạt mạnh
            if dist <= 2:
                score -= 80

            # ngõ cụt
            if escape_routes <= 1:
                score -= 30

            # tránh quay đầu
            if self.last_move is not None:
                opposite = {
                    Move.UP: Move.DOWN,
                    Move.DOWN: Move.UP,
                    Move.LEFT: Move.RIGHT,
                    Move.RIGHT: Move.LEFT,
                }
                if move == opposite.get(self.last_move):
                    score -= 20

            # phá loop mạnh: tránh quay lại vị trí gần đây
            if new_pos in self.prev_positions[-4:]:
                score -= 40

            # random nhẹ để tránh stuck
            score += random.random()

            if score > best_score:
                best_score = score
                best_move = move

        self.last_move = best_move

        # nếu vẫn bị STAY -> fallback random để tránh đứng im
        if best_move == Move.STAY:
            return self._random_move(my_position)

        return best_move


    def _update_memory(self, map_state):
        for i in range(21):
            for j in range(21):
                if map_state[i][j] != -1:
                    self.known_map[i][j] = map_state[i][j]


    def _is_valid_position(self, pos):
        row, col = pos
        return (
            0 <= row < 21 and
            0 <= col < 21 and
            self.known_map[row][col] != 1
        )


    def _compute_distance_map(self, start_pos):
        queue = deque([(start_pos, 0)])
        visited = {start_pos: 0}

        while queue:
            (row, col), dist = queue.popleft()

            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                nr, nc = row + dr, col + dc

                if (nr, nc) not in visited and self._is_valid_position((nr, nc)):
                    visited[(nr, nc)] = dist + 1
                    queue.append(((nr, nc), dist + 1))

        return visited


    def _random_move(self, my_position):
        moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(moves)

        for move in moves:
            dx, dy = move.value
            new_pos = (my_position[0] + dx, my_position[1] + dy)

            if self._is_valid_position(new_pos):
                return move

        return Move.STAY


    def _count_valid_moves(self, pos):
        count = 0
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dx, dy = move.value
            new_pos = (pos[0] + dx, pos[1] + dy)

            if self._is_valid_position(new_pos):
                count += 1

        return count


    def _is_dead(self, ghost_pos, pacman_pos):
        if pacman_pos is None:
            return False
        return abs(ghost_pos[0] - pacman_pos[0]) + abs(ghost_pos[1] - pacman_pos[1]) <= 1
