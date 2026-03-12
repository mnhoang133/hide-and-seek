import sys
import os
from pathlib import Path
from arena import Arena

def run_batch_tests(pacman_id, ghost_id, num_games = 100):
    results = { 'pacman_wins': 0, 'ghost_wins': 0}
    original_stdout = sys.stdout
    for i in range (num_games):
        sys.stdout = open(os.devnull, 'w', encoding = 'utf-8')
        try:
            arena = Arena(
                pacman_id = pacman_id,
                ghost_id = ghost_id,
                submissions_dir = "../submissions",
                max_steps = 200,
                visualize = False,
                deterministic_starts = False
            )
            arena.load_agents()
            result, starts = arena.run_game()
        finally:
            sys.stdout.close()
            sys.stdout = original_stdout
        if result in results:
            results[result] +=1
    print("\n" + "= "*50)
    print(f"{'TỔNG KẾT SAU ' + str(num_games) + ' TRẬN ĐẤU':^50}")
    print("="*50)
    print(f"Pacman ({pacman_id}) thắng: {results['pacman_wins']} trận ({(results['pacman_wins']/(results['pacman_wins']+results['ghost_wins']))*100}%)")
    print(f"Ghost  ({ghost_id}) thắng:  {results['ghost_wins']} trận ({(results['ghost_wins']/(results['pacman_wins']+results['ghost_wins']))*100}%)")
    print("="*50)
if __name__ == "__main__":
    
    PACMAN = "example_student"
    GHOST = "example_student"
    SO_TRAN_DAU = 100
    
    run_batch_tests(pacman_id=PACMAN, ghost_id=GHOST, num_games=SO_TRAN_DAU)