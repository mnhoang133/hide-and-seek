import sys
import os
import argparse
from pathlib import Path
from arena import Arena

def run_batch_tests(pacman_id, ghost_id, num_games=100, pacman_speed=1):
    results = { 'pacman_wins': 0, 'ghost_wins': 0, 'errors': 0}
    original_stdout = sys.stdout
    
    for i in range (num_games):
        sys.stdout = open(os.devnull, 'w', encoding='utf-8')
        try:
            arena = Arena(
                pacman_id = pacman_id,
                ghost_id = ghost_id,
                submissions_dir = "../submissions",
                max_steps = 200,
                visualize = False,
                deterministic_starts = False,
                pacman_speed = pacman_speed,
                step_timeout = 60.0  # Ngầm định luôn là 60 giây, không cần gõ lệnh
            )
            arena.load_agents()
            result, stats = arena.run_game()
            
            if result == 'pacman_wins':
                results['pacman_wins'] += 1
            elif result == 'ghost_wins' or result == 'draw':
                results['ghost_wins'] += 1
                
        except Exception as e:
            results['errors'] += 1
            sys.stdout.close()
            sys.stdout = original_stdout
            print(f"\n[!] PHÁT HIỆN LỖI TẠI TRẬN {i+1}:")
            print(f" -> Nguyên nhân: {e}")
            
        finally:
            if not getattr(sys.stdout, 'closed', True) and sys.stdout != original_stdout:
                sys.stdout.close()
            sys.stdout = original_stdout
            
    print("\n" + "="*50)
    print(f"{'TỔNG KẾT SAU ' + str(num_games) + ' TRẬN ĐẤU':^50}")
    print(f"{'(Tốc độ: ' + str(pacman_speed) + ' | Giới hạn: 60s/bước)':^50}")
    print("="*50)
    
    total_valid_games = results['pacman_wins'] + results['ghost_wins']
    if total_valid_games == 0: total_valid_games = 1 
    
    print(f"Pacman ({pacman_id}) thắng: {results['pacman_wins']} trận ({(results['pacman_wins']/total_valid_games)*100:.1f}%)")
    print(f"Ghost  ({ghost_id}) thắng:  {results['ghost_wins']} trận ({(results['ghost_wins']/total_valid_games)*100:.1f}%)")
    
    if results['errors'] > 0:
        print("-" * 50)
        print(f"⚠️ CẢNH BÁO: BỊ CRASH/QUÁ 1 PHÚT TẠI {results['errors']} TRẬN! ⚠️")
    
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy test tự động nhiều trận")
    parser.add_argument('--seek', type=str, required=True, help="Tên thư mục bot Pacman")
    parser.add_argument('--hide', type=str, required=True, help="Tên thư mục bot Ghost")
    parser.add_argument('--games', type=int, default=100, help="Số trận đấu (mặc định 100)")
    parser.add_argument('--pacman-speed', type=int, default=1, help="Tốc độ của Pacman (mặc định 1)")
    
    args = parser.parse_args()
    
    run_batch_tests(
        pacman_id=args.seek, 
        ghost_id=args.hide, 
        num_games=args.games, 
        pacman_speed=args.pacman_speed
    )
