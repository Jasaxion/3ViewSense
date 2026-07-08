import random
import argparse
import os
import json
from tqdm import tqdm
from utils import get_actual_dimensions, unique_solution, save_solutions_to_file, print_case

def get_conflicts(grid, R, C):
    """
    Calculate the list of grid cells that do not satisfy the uniqueness condition.
    Return: bad_cells (list of (r, c))
    """
    bad_cells = []
    
    # 1. Pre-calculate the maximum values and counts of rows and columns
    # For performance, here we do not use object encapsulation, directly use arrays
    r_maxs = [-1] * R
    r_counts = [0] * R
    for r in range(R):
        mx = -1
        cnt = 0
        for val in grid[r]:
            if val > mx:
                mx = val
                cnt = 1
            elif val == mx:
                cnt += 1
        r_maxs[r] = mx
        r_counts[r] = cnt
        
    c_maxs = [-1] * C
    c_counts = [0] * C
    for c in range(C):
        mx = -1
        cnt = 0
        for r in range(R):
            val = grid[r][c]
            if val > mx:
                mx = val
                cnt = 1
            elif val == mx:
                cnt += 1
        c_maxs[c] = mx
        c_counts[c] = cnt
        
    # 2. Check each cell
    for r in range(R):
        for c in range(C):
            val = grid[r][c]
            if val == 0:
                continue # 0 is always valid
            
            is_valid = False
            
            if val == 1:
                # Must satisfy: row max is 1 OR column max is 1
                if r_maxs[r] == 1 or c_maxs[c] == 1:
                    is_valid = True
            else:
                # val > 1
                # Must satisfy: row unique max OR column unique max
                is_row_peak = (val == r_maxs[r] and r_counts[r] == 1)
                is_col_peak = (val == c_maxs[c] and c_counts[c] == 1)
                if is_row_peak or is_col_peak:
                    is_valid = True
            
            if not is_valid:
                bad_cells.append((r, c))
                
    return bad_cells

def generate_with_min_conflicts(R, C, H, max_steps=1000, random_walk_prob=0.05):
    """
    Generate unique solution cases using the Min-Conflicts algorithm.
    """
    # 1. Random initialization
    current_grid = [[random.randint(0, H) for _ in range(C)] for _ in range(R)]
    
    for step in range(max_steps):
        # Get all conflict points
        bad_cells = get_conflicts(current_grid, R, C)
        
        # If no conflicts, means a valid solution is found
        if not bad_cells:
            return current_grid, step
        
        # 2. Randomly select a conflict point for repair
        r, c = random.choice(bad_cells)
        
        # 3. Try to modify each value in 0..H, see which one can minimize the total number of conflicts
        original_val = current_grid[r][c]
        best_val = original_val
        min_conflict_count = float('inf')
        
        # To increase randomness, shuffle the order of attempts
        candidates = list(range(H + 1))
        random.shuffle(candidates)
        
        # Random walk mechanism: occasionally randomly select a value to prevent getting stuck in local optima
        if random.random() < random_walk_prob:
            best_val = random.choice(candidates)
        else:
            # Greedy search
            for val in candidates:
                if val == original_val:
                    continue # Skip the current value (although it can be calculated, but no change)
                
                # Modify the grid
                current_grid[r][c] = val
                
                # Calculate the new number of conflicts (here it can be optimized to incremental calculation, but for 5x5 it is also very fast to calculate directly)
                current_bad_len = len(get_conflicts(current_grid, R, C))
                
                if current_bad_len < min_conflict_count:
                    min_conflict_count = current_bad_len
                    best_val = val
                
                # Backtrack (to test the next value)
                current_grid[r][c] = original_val
        
        # Apply the best modification
        current_grid[r][c] = best_val
        
    return None, max_steps # Not found a solution within the specified number of steps

def batch_generate_unique(R, C, H, num_required, max_global_attempts=50000000):
    """
    Batch generate unique solution cases.
    """
    results = []
    # Use a set to store the generated grid fingerprints (Tuple of Tuples)
    seen_hashes = set()
    
    attempts = 0
    total_steps = 0
    
    # Create a progress bar
    pbar = tqdm(total=num_required, desc="Generating", unit="case")
    
    while len(results) < num_required:
        attempts += 1
        
        # Prevent dead loops caused by space exhaustion or tight constraints
        if attempts > max_global_attempts:
            print(f"Warning: Stopped after {max_global_attempts} attempts. Found {len(results)} unique cases.")
            break
            
        # Generate a candidate solution
        grid, steps = generate_with_min_conflicts(R, C, H, max_steps=1000)
        
        if grid:
            # Check the actual dimensions: as long as one of the actual used rows, columns, or maximum height equals the required value, keep it
            actual_R, actual_C, actual_H = get_actual_dimensions(grid)
            if not (actual_R == R or actual_C == C or actual_H == H):
                # The actual dimensions do not meet the requirements, skip this solution, continue generating
                continue
            
            # Filter conditions: only keep samples that need top view (needs_top_view returns False)
            # If needs_top_view returns True (no top view needed), skip this sample
            if not unique_solution(grid):
                # print("jump")
                continue
            
            # Key step: convert the two-dimensional list to a hashable tuple structure
            # For example: [[1,0], [2,1]] -> ((1,0), (2,1))
            grid_hash = tuple(tuple(row) for row in grid)
            
            # Check if it is duplicated
            if grid_hash not in seen_hashes:
                seen_hashes.add(grid_hash)  # Record the fingerprint
                results.append(grid)        # Save the original list format
                total_steps += steps
                # Update the progress bar
                pbar.update(1)
                pbar.set_postfix({"Generated": len(results), "Total attempts": attempts})
            else:
                # print("Duplicate found, discarding...") # For debugging
                pass
        
    pbar.close()
    return results, {
        "total_attempts": attempts, 
        "restarts": attempts - len(results), # Includes failed and duplicated cases
        "avg_steps": total_steps / len(results) if results else 0
    }

# --- Test run ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate block layouts that satisfy the three-view unique solution conditions using the Min-Conflicts algorithm')
    parser.add_argument('--R', type=int, default=5, help='Number of rows (height of the top view), default: 5')
    parser.add_argument('--C', type=int, default=5, help='Number of columns (width of the top view), default: 5')
    parser.add_argument('--H', type=int, default=5, help='Maximum allowed height, default: 5')
    parser.add_argument('--num', type=int, default=100, help='Number of data to generate, default: 100')
    parser.add_argument('--output-dir', type=str, default='OUTPUT_DIR', help='Output directory path, default: OUTPUT_DIR')
    
    args = parser.parse_args()
    
    R, C, H, TARGET = args.R, args.C, args.H, args.num
    output_base_dir = args.output_dir
    
    # Output path format: {output_base_dir}/level{R}{C}{H}/block_solutions.json
    output_dir = os.path.join(output_base_dir, f'level{R}{C}{H}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Full output path
    output_path = os.path.join(output_dir, 'block_solutions.json')
    
    print(f"Generating {TARGET} unique cases for {R}x{C} grid (MaxH={H}) using Min-Conflicts Repair...")
    
    final_grids, stats = batch_generate_unique(R, C, H, TARGET)
    
    print(f"Total unique solutions found: {len(final_grids)}")
    print(f"\nStats: Restarts={stats['restarts']}, Avg Steps to Converge={stats['avg_steps']:.1f}")
    
    # Save all solutions to file
    save_solutions_to_file(final_grids, output_path)
    
    # Show the first few examples
    if len(final_grids) > 0:
        print("\nFirst 3 examples:")
        for res in final_grids[:3]:
            print_case(res)