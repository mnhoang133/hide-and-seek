
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
        self.name = "Cornering Master Pacman"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        
        self.capture_threshold = 1 
        if '--capture-distance' in sys.argv:
            try:
                idx = sys.argv.index('--capture-distance')
                self.capture_threshold = int(sys.argv[idx + 1])
            except (ValueError, IndexError):
                pass
                
        self.algorithm = "ASTAR" 
        self.last_ghost_pos = None # Ghi nhớ vị trí cũ của Ghost để tính quán tính

        self.locked_target = None
        self.lock_counter = 0

    def _get_forward_choke_point(self, ghost_pos, ghost_dir, map_state):
        """Hàm quét thẳng về phía trước để tìm ngã tư hoặc góc chết để chặn đầu"""
        curr_pos = ghost_pos
        # Quét tối đa 7 ô để tránh nhìn quá xa (Ghost có thể rẽ trước khi tới)
        for _ in range(7): 
            next_pos = (curr_pos[0] + ghost_dir[0], curr_pos[1] + ghost_dir[1])
            
            # Nếu đụng tường hoặc ra khỏi bản đồ, chặn ngay tại vị trí hiện tại của Ghost
            if not self._is_valid_position(next_pos, map_state):
                return curr_pos 
            
            # Đếm số đường đi tại ô tiếp theo
            valid_neighbors = 0
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                n = (next_pos[0] + move.value[0], next_pos[1] + move.value[1])
                if self._is_valid_position(n, map_state):
                    valid_neighbors += 1
            
            # Nếu có > 2 đường đi (Ngã 3, ngã 4)
            if valid_neighbors > 2:
                return next_pos
                
            curr_pos = next_pos
            
        return curr_pos # Nếu đường thẳng quá dài, chặn ở ô thứ 7

    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        
        try:
            path = []
            # Tính hướng di chuyển của Ghost
            ghost_dir = None
            if self.last_ghost_pos:
                ghost_dir = (enemy_position[0] - self.last_ghost_pos[0], 
                             enemy_position[1] - self.last_ghost_pos[1])
            self.last_ghost_pos = enemy_position

            target_pos = enemy_position 

            # Dồn góc nếu Ghost đã đi được một khoảng nhất định và có hướng di chuyển rõ ràng
            if step_number > 10 and ghost_dir is not None:
                # Khóa mục tiêu
                if self.lock_counter > 0 and self.locked_target is not None:
                    target_pos = self.locked_target
                    self.lock_counter -= 1
                    
                    # Nếu Pacman đã chạy tới ngay điểm chặn rồi thì mở khóa sớm
                    if my_position == self.locked_target:
                        self.lock_counter = 0
                        self.locked_target = None
                        target_pos = enemy_position
                
                # Quét tìm điểm chặn mới
                else:
                    choke_point = self._get_forward_choke_point(enemy_position, ghost_dir, map_state)
                    target_pos = choke_point
                    
                    # Nếu tìm ra được một ngã tư/góc chết hợp lý (khác chỗ Ghost đang đứng)
                    if choke_point != enemy_position:
                        self.locked_target = choke_point
                        # Khóa mục tiêu trong 3 lượt 
                        self.lock_counter = 3  
                
                # Tìm đường tới điểm dồn góc
                path = pacmanAlgorithm.astar(my_position, target_pos, map_state)
                
                # Nếu không tìm được đường tới điểm chặn (hoặc rủi ro), quay lại đuổi thẳng mặt và xóa khóa
                if not path and target_pos != enemy_position:
                    target_pos = enemy_position
                    self.lock_counter = 0
                    self.locked_target = None
            else:
                # Thuần A*
                self.lock_counter = 0
                self.locked_target = None

           
            if target_pos == enemy_position:
                best_moves = []
                min_path_len = float('inf')
                paths_dict = {}

                for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                    dr, dc = move.value
                    next_pos = (my_position[0] + dr, my_position[1] + dc)
                    
                    if self._is_valid_position(next_pos, map_state):
                        if next_pos == enemy_position:
                            min_path_len = 0
                            best_moves = [move]
                            paths_dict[move] = []
                            break
                            
                        sub_path = pacmanAlgorithm.astar(next_pos, enemy_position, map_state)
                        if not sub_path: continue
                            
                        path_len = len(sub_path)
                        paths_dict[move] = sub_path
                        
                        if path_len < min_path_len:
                            min_path_len = path_len
                            best_moves = [move]
                        elif path_len == min_path_len:
                            best_moves.append(move) 
                
                # Tie-breaker: Chọn ngã rẽ trùng với hướng Ghost vừa đi
                chosen_move = None
                if len(best_moves) > 1 and ghost_dir is not None:
                    for m in best_moves:
                        if m.value == ghost_dir:
                            chosen_move = m
                            break
                            
                if chosen_move is None and best_moves:
                    chosen_move = best_moves[0]
                    
                if chosen_move:
                    path = [chosen_move] + paths_dict[chosen_move]

            # Di chuyển với tốc độ tối đa có thể
            if path:
                best_move = path[0]
                desired_steps = 1
                for i in range(1, len(path)):
                    if path[i] == best_move:
                        desired_steps += 1
                    else:
                        break
                        
                if desired_steps == len(path) and desired_steps < self.pacman_speed:
                    desired_steps = self.pacman_speed
                    
                steps = self._max_valid_steps(my_position, best_move, map_state, desired_steps)
                if steps > 0:
                    return (best_move, steps)
                    
            return (Move.STAY, 1)

        except Exception as e:
            # Fallback an toàn nếu lỗi
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
