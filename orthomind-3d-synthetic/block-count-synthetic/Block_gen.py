import sys
import argparse
import os
from utils import get_actual_dimensions, unique_solution, save_solutions_to_file, print_case

def solve_cube_counting_unique(R, C, MaxH):
    """
    Traverse to generate all block layouts that meet the condition of "unique solution of three views."
    R: number of rows (height of the top view)
    C: number of columns (width of the top view)
    MaxH: maximum allowed height
    """
    
    solutions = []
    
    # Pre-generate all possible single row permutations to reduce repetitive calculations in recursion
    # For 3x3x3, there are 4^3 = 64 possible single rows, which is very fast
    import itertools
    possible_rows = list(itertools.product(range(MaxH + 1), repeat=C))
    
    # Pre-calculate the "row satisfied" mask for each possible row shape
    # row_sat_mask[i] is a boolean list that indicates which positions in the i-th possible row satisfy the condition of "unique within the row" or "the row is 1 and the maximum is 1"
    row_props = []
    for row in possible_rows:
        max_v = max(row)
        count_max = row.count(max_v)
        # Mask: if the row condition is satisfied, then True
        # Condition: (is the maximum value AND the maximum value is unique) OR (is 1 AND the maximum value is 1)
        mask = []
        for v in row:
            if v == 0:
                mask.append(True) # 0 is always valid (empty cell)
            elif (v == max_v and count_max == 1) or (v == 1 and max_v == 1):
                mask.append(True)
            else:
                mask.append(False)
        row_props.append((row, mask))

    def dfs(r_idx, current_grid, col_maxs, col_caps):
        """
        r_idx: the index of the current row being filled
        current_grid: the current filled grid
        col_maxs: the maximum height of each column so far [c0, c1, ...]
        col_caps: the strict upper limit for each column allowed to be placed next [limit0, limit1, ...]
                  if limit is MaxH + 1, it means no limit
        """
        # Base case: all rows are filled
        if r_idx == R:
            solutions.append([list(row) for row in current_grid])
            return

        # Traverse all possible rows
        for row, row_sat_mask in row_props:
            
            # 1. Quick check: does the current row violate the previous "column upper limit" constraint
            # Must satisfy row[c] < col_caps[c] (if the upper limit is for strictly greater values)
            # Special case: if the previous constraint is "the maximum of the column is 1", then col_caps[c] is set to 2 (i.e. < 2)
            valid_structure = True
            for c in range(C):
                if row[c] >= col_caps[c]:
                    valid_structure = False
                    break
            if not valid_structure:
                continue

            # 2. Logical check and new constraint generation
            new_col_caps = list(col_caps) # Copy the constraints
            possible = True
            
            for c in range(C):
                val = row[c]
                
                # If this position is already "self-sufficient" within the row (row satisfied), then it does not depend on the column
                if row_sat_mask[c]:
                    # Since it does not depend on the column, it will not produce new "more strict" constraints in the future
                    # But must inherit the old constraints (because the blocks above may depend on the smaller ones below)
                    pass 
                
                else:
                    # If the row is not satisfied, it must depend on "unique within the column"
                    # Condition A: must be higher than all the blocks above (become the current column boss)
                    # For val=1, the logic is: if the row condition is not satisfied (i.e. the maximum of the row is >1), then the maximum of the column must be 1.
                    # This means col_maxs[c] must <= 1 (actually if val=1, it is also valid, although not strictly greater)
                    
                    if val == 1:
                        # Depends on the column being satisfied, and val=1. This means the column must be limited to height 1.
                        # Check history: the previous heights cannot exceed 1.
                        if col_maxs[c] > 1:
                            possible = False; break
                        # Apply future constraints: in the future, it cannot exceed 1 (i.e. < 2)
                        if new_col_caps[c] > 2:
                            new_col_caps[c] = 2
                            
                    elif val > 1:
                        # Depends on the column being satisfied, and val > 1.
                        # Check history: must be strictly greater than all the previous heights
                        if val <= col_maxs[c]:
                            possible = False; break
                        
                        # Apply future constraints: in the future, it must be strictly less than me
                        # Only update if the new constraint is tighter than the old constraint
                        if new_col_caps[c] > val:
                            new_col_caps[c] = val
                    
                    else:
                        # val == 0
                        # 0 is always valid (in row_sat_mask, it is already treated as True, so this branch will not be entered)
                        pass

            if possible:
                # Update the history of the maximum height of the columns
                new_col_maxs = [max(col_maxs[c], row[c]) for c in range(C)]
                
                # Recursively fill the next row
                dfs(r_idx + 1, current_grid + [row], new_col_maxs, new_col_caps)

    # Initial state
    # col_maxs are all 0
    # col_caps are all MaxH + 1 (no limit)
    dfs(0, [], [0]*C, [MaxH + 1]*C)
    
    return solutions

def filter_by_actual_dimensions(solutions, required_R, required_C, required_H):
    """
    Filter solutions: keep only those where the actual used number of rows, columns, or maximum height equals the required value.
    
    Args:
        solutions: the list of solutions
        required_R: the required number of rows
        required_C: the required number of columns
        required_H: the required maximum height
    
    Returns:
        the list of filtered solutions
    """
    filtered = []
    for sol in solutions:
        actual_R, actual_C, actual_H = get_actual_dimensions(sol)
        # Keep only those where the actual used number of rows, columns, or maximum height equals the required value
        if actual_R == required_R or actual_C == required_C or actual_H == required_H:
            filtered.append(sol)
    
    return filtered

# Main program entry
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate block layouts that meet the condition of "unique solution of three views."')
    parser.add_argument('--R', type=int, default=3, help='number of rows (height of the top view), default: 3')
    parser.add_argument('--C', type=int, default=3, help='number of columns (width of the top view), default: 3')
    parser.add_argument('--H', type=int, default=3, help='maximum allowed height, default: 3')
    parser.add_argument('--output-dir', type=str, default='OUTPUT_DIR', help='output directory path, default: OUTPUT_DIR')
    
    args = parser.parse_args()
    
    R, C, H = args.R, args.C, args.H
    output_base_dir = args.output_dir
    
    # Output path format: {output_base_dir}/level{R}{C}{H}/block_solutions.json
    output_dir = os.path.join(output_base_dir, f'level{R}{C}{H}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Complete output path
    output_path = os.path.join(output_dir, 'block_solutions.json')
    
    print(f"Generating unique cases for {R}x{C} grid with max height {H}...")
    results = solve_cube_counting_unique(R, C, H)
    
    print(f"Total unique solutions found: {len(results)}")
    
    # Filter solutions: keep only those where the actual used number of rows, columns, or maximum height equals the required value
    filtered_results = filter_by_actual_dimensions(results, R, C, H)
    print(f"After filtering (actual R={R} or C={C} or H={H}): {len(filtered_results)} solutions")
    
    # Further filtering: keep only those that need the top view (needs_top_view returns False)
    final_results = []
    for sol in filtered_results:
        # If unique_solution returns True (no top view needed), then keep this sample
        if unique_solution(sol):
            final_results.append(sol)
    
    print(f"After filtering (needs top view): {len(final_results)} solutions")
    
    # Save the filtered solutions to a file
    save_solutions_to_file(final_results, output_path)
    
    # Show the first few examples
    if len(final_results) > 0:
        print("\nFirst 3 examples:")
        for res in final_results[:3]:
            print_case(res)