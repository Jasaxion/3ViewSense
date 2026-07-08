# Color definition and mapping

# Color list (RGB format, 0-255 range)
COLOR_LIST = [
    # (0, 0, 0),      # Black - mapped to 1
    (255, 255, 255), # White - mapped to 2
    (255, 0, 0),    # Red - mapped to 3
    (255, 255, 0),  # Yellow - mapped to 4
    (0, 0, 255),    # Blue - mapped to 5
    (0, 255, 0),    # Green - mapped to 6
]

# RGB to color ID mapping (starting from 1, 0 means no cell)
RGB_TO_ID = {
    # (0, 0, 0): 1,           # Black
    (255, 255, 255): 2,     # White
    (255, 0, 0): 3,         # Red
    (255, 255, 0): 4,       # Yellow
    (0, 0, 255): 5,         # Blue
    (0, 255, 0): 6,         # Green
}

import json

def get_actual_dimensions(grid):
    """
    Calculate the actual used number of rows, columns and maximum height of the grid.
    Actual number of rows = the last row index with non-zero value - the first row index with non-zero value + 1
    Actual number of columns = the last column index with non-zero value - the first column index with non-zero value + 1
    Actual maximum height = the maximum value of all positions in the grid
    
    Args:
        grid: R x C grid, each position is the height value (0 means empty)
    
    Returns:
        (actual_R, actual_C, actual_H): The actual used number of rows, columns and maximum height
    """
    R = len(grid)
    C = len(grid[0]) if R > 0 else 0
    
    # Find the first and last row and column indices with non-zero values
    first_row = None
    last_row = None
    first_col = None
    last_col = None
    max_height = 0
    
    for r in range(R):
        for c in range(C):
            val = grid[r][c]
            if val != 0:
                if first_row is None:
                    first_row = r
                if first_col is None or c < first_col:
                    first_col = c
                last_row = r
                if last_col is None or c > last_col:
                    last_col = c
                if val > max_height:
                    max_height = val
    
    # If no non-zero values are found, return 0
    if first_row is None:
        return 0, 0, 0
    
    # Calculate the actual used number of rows and columns
    actual_R = last_row - first_row + 1
    actual_C = last_col - first_col + 1
    
    return actual_R, actual_C, max_height

def unique_solution(h):
    """
    Determine if the given grid and isometric view is a unique solution, and can uniquely count the number of blocks.
    If any block (or originally empty space that could be a block) is completely occluded,
    causing it to be impossible to determine its exact height, return False, otherwise return True.
    
    The two conditions for determining occlusion (satisfying any one is considered occluded):
    1. Diagonal occlusion: Covered by the shadow of (x+i, y+i).
    2. Neighbor occlusion: Blocked by the high wall on both the right (x+1, y) and the front (x, y+1).
    """
    if not h or not h[0]:
        return False
    
    R = len(h)      # number of rows (y axis)
    C = len(h[0])   # number of columns (x axis)
    
    # Traverse each position as a "victim" (Candidate)
    for x in range(C):
        for y in range(R):
            current_h = h[x][y]
            
            # --- Check 1: Diagonal occlusion (Diagonal Occlusion) ---
            # As long as any high tower on the diagonal blocks the top, the top is invisible
            is_diag_covered = False
            
            # Look in the direction of (x+1, y+1) (i.e. the block in front)
            dist = 1
            while x + dist < C and y + dist < R:
                blocker_h = h[x + dist][y + dist]
                
                # Calculate the height of the blocker's shadow projected onto the current position
                # The slope of the line of sight is 1: the height decreases by 1 for each increase in distance
                shadow_height = blocker_h - dist
                
                # If the current block height is lower than or equal to the shadow, it means it is completely covered
                if shadow_height >= current_h and shadow_height >= 1:
                    # There is a subtle logic here:
                    # If current_h is already very high and is visible, it is not considered blocked.
                    # If current_h is very low and is blocked, it is a Blind Spot.
                    # Especially: If current_h < shadow_height, it means even if it is increased in height, it is not discovered -> not unique
                    # If current_h == shadow_height, it means it is at a critical point, and cannot be confirmed through visual inspection whether it is a shadow or an entity -> not unique
                    is_diag_covered = True
                    break
                dist += 1
            
            # --- Check 2: Neighbor occlusion (Neighbor Occlusion) ---
            # Your logic: If the right and front are both higher than the current by more than 1 block, it is considered blocked
            is_neighbor_covered = False
            
            # Get neighbor heights; treat out-of-bounds as 0
            h_right = h[x+1][y] if x + 1 < C else 0
            h_front = h[x][y+1] if y + 1 < R else 0
            
            # Only when both neighbors exist and are higher than the current, a "deep well" is formed
            if h_right > current_h and h_front > current_h and current_h > 0:
                is_neighbor_covered = True
            
            # --- Comprehensive judgment ---
            # If the top is blocked by the diagonal, a TopView is required
            if is_diag_covered:
                return False # The solution is not unique
                
            # If the top is not blocked by the diagonal (originally visible), but is blocked by the neighbors, it is also not visible
            if is_neighbor_covered:
                return False # The solution is not unique

    # If all cells can be seen in at least one way, no TopView is required
    return True

def generate_voxel_matrix(h, R=None, C=None, H=None):
    """
    Generate a 3D matrix, mark the state of each block.
    Input:
        h: 2D height map
        R, C, H: (optional) size limits
    Output:
        3D list voxels[x][y][z]
        1 : Visible (Visible)
        0 : Empty (Empty)
        -1: Occluded / Hidden (Occluded / Hidden)
    """
    if not h:
        return []

    # 1. Determine the grid size
    calc_R = len(h)
    calc_C = len(h[0]) if calc_R > 0 else 0
    current_R = R if R is not None else calc_R
    current_C = C if C is not None else calc_C

    # Determine the maximum height (z axis range)
    max_h_grid = 0
    for row in h:
        for val in row:
            if val > max_h_grid:
                max_h_grid = val
    current_H = H if H is not None else max_h_grid

    # 2. Initialize the 3D matrix (default 0)
    voxels = [[[0 for _ in range(current_H)] 
               for _ in range(current_C)] 
              for _ in range(current_R)]

    # 3. Traverse each column
    for x in range(current_R):
        for y in range(current_C):
            # Get the height of the current column
            # Handle boundary safety, prevent R,C parameters from being larger than the actual h causing out of bounds
            col_height = h[x][y] if (x < len(h) and y < len(h[0])) else 0
            
            # --- Step A: Calculate the "diagonal shadow threshold" of the current column ---
            # Any block with height z less than this threshold will be completely blocked by the high tower in the diagonal direction
            # Logic: The block at (x+d, y+d) if the height is H_neighbor,
            # It will block all blocks below the position at (x, y) with height H_neighbor - d.
            diag_shadow_limit = -999 
            
            dist = 1
            while x + dist < current_R and y + dist < current_C:
                neighbor_h = h[x + dist][y + dist]
                # Compute the height of the occlusion line projected by the neighbor
                # For example, if the neighbor is 10 blocks high and the distance is 1, it blocks all lines below 10-1=9 (excluding 9)
                # That is, all lines with z < 9 are blocked.
                current_shadow = neighbor_h - dist
                if current_shadow > diag_shadow_limit:
                    diag_shadow_limit = current_shadow
                dist += 1

            # 4. Traverse each block of the current column (z)
            for z in range(current_H):
                # If z exceeds the column height, it is air (0)
                if z >= col_height:
                    voxels[x][y][z] = 0
                    continue
                
                # --- Logic judgment starts ---
                
                # Rule 1: Diagonal occlusion (Global Occlusion)
                # If z is less than the shadow height, it means it is completely blocked by the high tower in front (including the top and side)
                # For example: Neighbor 10, distance 1, shadow=9. z=0..8 (height 1..9) will all satisfy z < 9, blocked.
                if z < diag_shadow_limit:
                    voxels[x][y][z] = -1
                    continue

                # If not blocked by the diagonal, continue to judge:
                
                # Rule 2: Top check
                # If it is the highest block of the column, and has passed rule 1, it is definitely visible
                if z == col_height - 1:
                    voxels[x][y][z] = 1
                    continue
                
                # Rule 3: Side occlusion check (Local Occlusion)
                # If it is not the top layer, the top is blocked by oneself, and can only rely on the side.
                # Only when the right and front are both blocked, it is not visible.
                
                # Check the right (x+1)
                blocked_right = False
                if x + 1 < current_R:
                    if h[x+1][y] > z: # The neighbor is higher than the current layer, blocking the right face
                        blocked_right = True
                # (If x+1 is out of bounds, it means it is an edge, blocked_right = False, visible)

                # Check the front (y+1)
                blocked_front = False
                if y + 1 < current_C:
                    if h[x][y+1] > z: # The neighbor is higher than the current layer, blocking the front face
                        blocked_front = True
                
                if blocked_right and blocked_front:
                    voxels[x][y][z] = -1
                else:
                    voxels[x][y][z] = 1

    return voxels


def save_solutions_to_file(solutions, filename="block_solutions.json"):
    """
    Save all solutions to a JSON file
    """
    with open(filename, 'w') as f:
        json.dump(solutions, f)
    print(f"Saved {len(solutions)} solutions to {filename}")

def print_case(grid):
    print("Solution:")
    for row in grid:
        print(row)
    print("-" * 10)
