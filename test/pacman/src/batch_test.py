import sys
import os
import argparse
import numpy as np
from pathlib import Path
from arena import Arena

# --- HÀM 1: ĐỌC MAP TỪ FILE TEXT ---
def load_maps_from_file(filepath):
    if not os.path.exists(filepath):
        print(f"[!] Không tìm thấy file: {filepath}")
        return []
        
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        
    raw_maps = content.split('\n\n')
    maps = []
    
    for raw_map in raw_maps:
        lines = [line.strip() for line in raw_map.split('\n') if line.strip()]
        if lines:
            maps.append(lines)
            
    return maps

# --- HÀM 2: DỊCH MAP SANG NUMPY ---
def parse_map_to_numpy(layout):
    map_array = np.zeros((len(layout), len(layout[0])), dtype=int)
    pacman_start = None
    ghost_start = None
    
    for i, row in enumerate(layout):
        for j, cell in enumerate(row):
            if cell == '#' or cell == '-':
                map_array[i, j] = 1
            else:
                map_array[i, j] = 0
                if cell == 'P':
                    pacman_start = (i, j)
                elif cell == 'G':
                    ghost_start = (i, j)
                    
    return map_array, pacman_start, ghost_start

# --- HÀM 3: CHẠY TEST AUTO VÀ IN BẢNG THỐNG KÊ ---
def run_batch_tests(pacman_id, ghost_id, maps, num_games=100, pacman_speed=1):
    original_stdout = sys.stdout
    total_maps = len(maps)
    
    if total_maps == 0:
        print("[!] Không có map nào để chạy!")
        return

    print(f"🚀 Bắt đầu test {total_maps} map. Mỗi map {num_games} trận...")
    
    # Biến lưu trữ thống kê cho bảng tổng kết
    all_maps_stats = []
    total_p_wins = 0
    total_g_wins = 0
    total_draws = 0

    for map_idx, current_map_layout in enumerate(maps):
        print(f" -> Đang cày Map {map_idx + 1:02d}/{total_maps}...", end="", flush=True)
        
        np_map, p_start, g_start = parse_map_to_numpy(current_map_layout)
        
        map_stats = {'p_wins': 0, 'g_wins': 0, 'draws': 0, 'errors': 0, 'p_steps': [], 'g_dists': []}
        
        for i in range(num_games):
            sys.stdout = open(os.devnull, 'w', encoding='utf-8')
            try:
                arena = Arena(
                    pacman_id = pacman_id,
                    ghost_id = ghost_id,
                    submissions_dir = "../submissions",
                    max_steps = 200,
                    visualize = False,
                    deterministic_starts = True,
                    pacman_speed = pacman_speed,
                    step_timeout = 60.0
                )
                
                arena.env.map = np_map.copy()
                arena.env.height, arena.env.width = np_map.shape
                arena.env.default_pacman_start = p_start
                arena.env.default_ghost_start = g_start
                arena.env.reset()
                
                arena.load_agents()
                result, stats = arena.run_game()
                final_dist = arena.env.get_distance(arena.env.pacman_pos, arena.env.ghost_pos)
                
                if result == 'pacman_wins':
                    map_stats['p_wins'] += 1
                    map_stats['p_steps'].append(stats['total_steps'])
                elif result == 'ghost_wins':
                    map_stats['g_wins'] += 1
                    map_stats['g_dists'].append(final_dist)
                elif result == 'draw':
                    map_stats['draws'] += 1
                    map_stats['g_dists'].append(final_dist)
                    
            except Exception as e:
                map_stats['errors'] += 1
            finally:
                if getattr(sys.stdout, 'closed', False) is False and sys.stdout != original_stdout:
                    sys.stdout.close()
                sys.stdout = original_stdout
                
        # Tính toán trung bình cho Map này
        avg_steps = sum(map_stats['p_steps']) / len(map_stats['p_steps']) if map_stats['p_steps'] else 0
        avg_dist = sum(map_stats['g_dists']) / len(map_stats['g_dists']) if map_stats['g_dists'] else 0
        
        all_maps_stats.append({
            'map': map_idx + 1,
            'p_wins': map_stats['p_wins'],
            'g_wins': map_stats['g_wins'],
            'draws': map_stats['draws'],
            'avg_steps': avg_steps,
            'avg_dist': avg_dist,
            'errors': map_stats['errors']
        })
        
        total_p_wins += map_stats['p_wins']
        total_g_wins += map_stats['g_wins']
        total_draws += map_stats['draws']
        
        print(" Xong!")

    # ================= IN BẢNG THỐNG KÊ =================
    print("\n" + "="*80)
    print(f"{'BẢNG PHONG THẦN THỐNG KÊ TỪNG MAP':^80}")
    print("="*80)
    print(f"{'Map':<6} | {'Pacman (Win)':<15} | {'Ghost (Win)':<15} | {'TB Bước (P Win)':<18} | {'TB Khoảng Cách (G Win)':<20}")
    print("-" * 80)
    
    for s in all_maps_stats:
        p_win_str = f"{s['p_wins']} ({s['p_wins']/num_games*100:.0f}%)"
        g_win_str = f"{s['g_wins']+s['draws']} ({(s['g_wins']+s['draws'])/num_games*100:.0f}%)"
        step_str = f"{s['avg_steps']:.1f} bước" if s['avg_steps'] > 0 else "N/A"
        dist_str = f"{s['avg_dist']:.1f} ô" if s['avg_dist'] > 0 else "N/A"
        
        err_warning = f" ⚠️ {s['errors']} lỗi" if s['errors'] > 0 else ""
        
        print(f"{s['map']:<6} | {p_win_str:<15} | {g_win_str:<15} | {step_str:<18} | {dist_str:<20} {err_warning}")

    print("="*80)
    
    # Tổng kết chung
    total_valid = total_p_wins + total_g_wins + total_draws
    if total_valid == 0: total_valid = 1
    
    print(f"🏆 TỔNG THẮNG PACMAN: {total_p_wins} trận ({(total_p_wins/total_valid)*100:.1f}%)")
    print(f"💀 TỔNG THẮNG GHOST: {total_g_wins + total_draws} trận ({((total_g_wins+total_draws)/total_valid)*100:.1f}%)")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy test tự động nhiều trận trên nhiều map")
    parser.add_argument('--seek', type=str, required=True, help="Tên thư mục bot Pacman")
    parser.add_argument('--hide', type=str, required=True, help="Tên thư mục bot Ghost")
    parser.add_argument('--maps-file', type=str, required=True, help="Đường dẫn đến file all_maps.txt")
    parser.add_argument('--games', type=int, default=100, help="Số trận đấu MỖI MAP (mặc định 100)")
    parser.add_argument('--pacman-speed', type=int, default=1, help="Tốc độ của Pacman (mặc định 1)")
    
    args = parser.parse_args()
    
    loaded_maps = load_maps_from_file(args.maps_file)
    
    if loaded_maps:
        run_batch_tests(
            pacman_id=args.seek, 
            ghost_id=args.hide, 
            maps=loaded_maps,
            num_games=args.games, 
            pacman_speed=args.pacman_speed
        )
