import sys
import os
import numpy as np
import argparse
from pathlib import Path

# Tích hợp vào cấu trúc thư mục của đồ án
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from environment import Environment, Move
from agent_loader import AgentLoader

def run_single_game(pacman_id, ghost_id, args):
    try:
        # ĐÃ FIX: Truyền đầy đủ Speed và Capture Distance từ command line
        env = Environment(
            max_steps=200,
            pacman_speed=args.pacman_speed,
            capture_distance_threshold=args.capture_distance
        )
        
        submissions_dir = src_path.parent / "submissions"
        loader = AgentLoader(submissions_dir=str(submissions_dir))
        
        pacman_agent = loader.load_agent(pacman_id, "pacman", init_kwargs={"pacman_speed": args.pacman_speed})
        ghost_agent = loader.load_agent(ghost_id, "ghost")
        
        step_num = 1
        while step_num <= 200:
            # 1. Lượt Pacman (ĐÃ FIX: Dùng get_observation gốc của hệ thống)
            p_obs, p_pos, p_enemy = env.get_observation('pacman', args.pacman_obs_radius, args.ghost_obs_radius)
            p_move_data = pacman_agent.step(p_obs, p_pos, p_enemy, step_num)
            p_move, p_steps = loader.validate_agent_move(p_move_data, 'pacman', pacman_id, args.pacman_speed)
            
            # 2. Lượt Ghost
            g_obs, g_pos, g_enemy = env.get_observation('ghost', args.pacman_obs_radius, args.ghost_obs_radius)
            g_move_data = ghost_agent.step(g_obs, g_pos, g_enemy, step_num)
            g_move = loader.validate_agent_move(g_move_data, 'ghost', ghost_id)
            
            # 3. Thực thi
            is_over, result, new_state = env.step(
                pacman_move=(p_move, p_steps), 
                ghost_move=g_move
            )
            
            if is_over:
                if result == 'pacman_wins':
                    return "WIN", step_num
                else:
                    return "LOSS/DRAW", 200
                    
            step_num += 1
            
        return "DRAW", 200
    except Exception as e:
        return f"ERROR: {str(e)}", 0

def main():
    parser = argparse.ArgumentParser(description='Batch Testing for Pacman AI')
    parser.add_argument('--seek', type=str, required=True, help='Student ID for Pacman')
    parser.add_argument('--hide', type=str, required=True, help='Student ID for Ghost')
    parser.add_argument('--games', type=int, default=100, help='Number of games to run')
    
    # --- ĐÃ FIX: Thêm đầy đủ tham số y hệt arena.py ---
    parser.add_argument('--pacman-speed', type=int, default=1, help='Speed of Pacman')
    parser.add_argument('--capture-distance', type=int, default=1, help='Distance to capture ghost')
    parser.add_argument('--pacman-obs-radius', type=int, default=0, help='Observation radius for Pacman')
    parser.add_argument('--ghost-obs-radius', type=int, default=0, help='Observation radius for Ghost')
    
    args = parser.parse_args()

    results = []
    win_steps = []
    
    print(f"\n🚀 Đang kiểm tra: {args.seek} vs {args.hide}")
    print(f"⚙️  Cấu hình: Speed={args.pacman_speed}, CaptureDist={args.capture_distance}, P_Vision={args.pacman_obs_radius}, G_Vision={args.ghost_obs_radius}")
    print(f"📊 Số lượng: {args.games} trận đấu")
    print("Đang giả lập, vui lòng chờ...\n")

    for i in range(args.games):
        if (i + 1) % 10 == 0 or (i + 1) == args.games:
            print(f"🔄 Đã chạy xong {i + 1}/{args.games} trận...")
            
        res, steps = run_single_game(args.seek, args.hide, args)
        
        if "ERROR" in res:
            print(f"\n🚨 LỖI CHI TIẾT TRẬN {i+1}: {res}")
            break
            
        results.append(res)
        if res == "WIN":
            win_steps.append(steps)

    # Thống kê kết quả
    total = len(results)
    if total == 0:
        print("❌ Không có trận đấu nào được hoàn thành do lỗi!")
        return

    wins = results.count("WIN")
    draws = results.count("DRAW") + results.count("LOSS/DRAW")
    errors = total - wins - draws
    
    win_rate = (wins / total) * 100
    avg_steps = np.mean(win_steps) if win_steps else 0
    min_steps = np.min(win_steps) if win_steps else 0
    max_steps_win = np.max(win_steps) if win_steps else 0

    print("\n" + "="*45)
    print(f"       BẢNG TỔNG KẾT HIỆU SUẤT ({total} GAME)")
    print("="*45)
    print(f"🏆 Tỷ lệ thắng (Win Rate):    {win_rate:>6.2f}%")
    print(f"⏱️  Số bước trung bình/thắng: {avg_steps:>6.2f} bước")
    print(f"⚡ Thắng nhanh nhất:          {min_steps:>6} bước")
    print(f"🐢 Thắng chậm nhất:           {max_steps_win:>6} bước")
    print("-" * 45)
    print(f"✅ Số trận thắng:             {wins:>6}")
    print(f"🤝 Số trận chưa bắt được:     {draws:>6}")
    print(f"❌ Số trận lỗi (Error):       {errors:>6}")
    print("="*45)

if __name__ == "__main__":
    main()
