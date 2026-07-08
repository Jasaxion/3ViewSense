import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random
from tqdm import tqdm
from utils import COLOR_LIST, RGB_TO_ID


def set_case_seed(case_id):
    """Set the random seed to ensure the color of each case is fixed"""
    random.seed(case_id)
    np.random.seed(case_id)


def rgb_to_normalized(rgb):
    """Convert the RGB tuple (0-255) to the normalized (0-1) format"""
    return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)


def create_cube_vertices(center):
    """Create the 8 vertices of the cube"""
    cx, cy, cz = center
    vertices = np.array([
        [cx-0.5, cy-0.5, cz-0.5],
        [cx+0.5, cy-0.5, cz-0.5],
        [cx+0.5, cy+0.5, cz-0.5],
        [cx-0.5, cy+0.5, cz-0.5],
        [cx-0.5, cy-0.5, cz+0.5],
        [cx+0.5, cy-0.5, cz+0.5],
        [cx+0.5, cy+0.5, cz+0.5],
        [cx-0.5, cy+0.5, cz+0.5],
    ])
    return vertices


def get_cube_faces(vertices):
    """Return the 6 faces of the cube"""
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom face
        [vertices[4], vertices[7], vertices[6], vertices[5]],  # Top face
        [vertices[0], vertices[4], vertices[5], vertices[1]],  # Front face
        [vertices[2], vertices[6], vertices[7], vertices[3]],  # Back face
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # Left face
        [vertices[1], vertices[5], vertices[6], vertices[2]],  # Right face
    ]
    return faces


def draw_orthographic_view(grid, view_type, output_path, H, size=(400, 300)):
    """
    Draw the three views (orthographic projection)
    view_type: 'top', 'front', 'left'
    H: The height dimension of the grid (H in RCH), used to set the coordinate range
    """
    fig, ax = plt.subplots(figsize=(size[0]/100, size[1]/100), dpi=100)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    C = len(grid)
    R = len(grid[0]) if C > 0 else 0
    
    # Set the coordinate range
    if view_type == 'top':
        ax.set_xlim(-0.5, C - 0.5)
        ax.set_ylim(-0.5, R - 0.5)
    elif view_type == 'front':
        ax.set_xlim(-0.5, C - 0.5)
        ax.set_ylim(-0.5, H - 0.5)
    elif view_type == 'left':
        ax.set_xlim(-0.5, R - 0.5)
        ax.set_ylim(-0.5, H - 0.5)
    
    # Draw the blocks
    if view_type == 'top':
        for y in range(R):
            for x in range(C):
                height = grid[x][y]
                if height > 0:
                    for z in range(height):
                        rect = plt.Rectangle((C-1-x-0.5, R-1-y-0.5), 1, 1, 
                                            facecolor='lightgray', 
                                            edgecolor='black', linewidth=1)
                        ax.add_patch(rect)
    
    elif view_type == 'front':
        for y in range(R):
            for x in range(C):
                height = grid[x][y]
                if height > 0:
                    for z in range(height):
                        rect = plt.Rectangle((C-1-x-0.5, z-0.5), 1, 1,
                                            facecolor='lightgray',
                                            edgecolor='black', linewidth=1)
                        ax.add_patch(rect)
    
    elif view_type == 'left':
        for y in range(R):
            for x in range(C):
                height = grid[x][y]
                if height > 0:
                    for z in range(height):
                        rect = plt.Rectangle((y-0.5, z-0.5), 1, 1,
                                            facecolor='lightgray',
                                            edgecolor='black', linewidth=1)
                        ax.add_patch(rect)
    
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.close()


def draw_3d_view(grid, camera_pos, output_path, H, random_color=False, color_map=None, size=(400, 300)):
    """
    Draw the 3D view
    camera_pos: The camera position, e.g. (1, 1, 1) or (-1, 1, 1)
    H: The height dimension of the grid (H in RCH), used to set the coordinate range
    random_color: Whether to use random color, False to use fixed color
    color_map: An optional dictionary, used to store the mapping from (x, y, z) to (rgb_tuple, color_id)
    """
    fig = plt.figure(figsize=(size[0]/100, size[1]/100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    fig.patch.set_facecolor('#f0f0f0')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    ax.grid(False)
    ax.axis('off')
    
    C = len(grid)
    R = len(grid[0]) if C > 0 else 0
    
    # Collect all blocks
    blocks = []
    for y in range(R):
        for x in range(C):
            height = grid[x][y]
            if height > 0:
                for z in range(height):
                    center = (x + 0.5, y + 0.5, z + 0.5)
                    blocks.append((x, y, z, center))
    
    # Draw the blocks
    for x, y, z, center in blocks:
        if random_color and color_map is not None:
            # Get the color from the color_map
            if (x, y, z) in color_map:
                rgb_color, color_id = color_map[(x, y, z)]
                color = rgb_to_normalized(rgb_color)
            else:
                # If the color_map does not have, it means it is the first generation, create and save
                rgb_color = random.choice(COLOR_LIST)
                color_id = RGB_TO_ID[rgb_color]
                color = rgb_to_normalized(rgb_color)
                color_map[(x, y, z)] = (rgb_color, color_id)
        elif random_color:
            # If random_color is True but there is no color_map, directly select the color (should not happen)
            rgb_color = random.choice(COLOR_LIST)
            color = rgb_to_normalized(rgb_color)
        else:
            # Use the fixed light blue
            color = (0.4, 0.6, 0.9)
        
        vertices = create_cube_vertices(center)
        faces = get_cube_faces(vertices)
        
        face_collection = Poly3DCollection(faces, 
                                          facecolors=[color] * 6,
                                          edgecolors='black',
                                          linewidths=0.5,
                                          alpha=1.0)
        ax.add_collection3d(face_collection)
    
    # Set the coordinate range
    ax.set_xlim(0, C)
    ax.set_ylim(0, R)
    ax.set_zlim(0, H)
    ax.set_box_aspect([1, 1, 1])
    
    # Set the viewpoint
    if camera_pos[0] > 0:
        elev = 30
        azim = 45
    else:
        elev = 30
        azim = 135
    
    ax.view_init(elev=elev, azim=azim)
    
    
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0, facecolor='#f0f0f0')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Render the three views and 3D view of the block layout')
    parser.add_argument('--file', type=str, default='./OUTPUT_DIR/level333/block_solutions.json', 
                        help='The input JSON file path')
    parser.add_argument('--num-cases', type=int, default=10, 
                        help='The number of cases to process, default: 10')
    parser.add_argument('--random-color', action='store_true', default=False,
                        help='Use random color (default: False)')
    
    args = parser.parse_args()
    
    json_path = args.file
    num_cases = args.num_cases
    
    with open(json_path, 'r') as f:
        all_solutions = json.load(f)
    
    json_dir = os.path.dirname(os.path.abspath(json_path))
    output_base = json_dir
    
    # Extract the H value from the path (H in RCH)
    # For example: level222 -> H=2, level333 -> H=3
    basename = os.path.basename(json_dir)
    if basename.startswith('level') and len(basename) == 8:
        H = int(basename[-1])
    else:
        # If it cannot be extracted from the path, infer from the grid
        if all_solutions:
            sample_grid = all_solutions[0]
            H = 0
            for x in range(len(sample_grid)):
                for y in range(len(sample_grid[0])):
                    if sample_grid[x][y] > H:
                        H = sample_grid[x][y]
            if H == 0:
                H = 1
        else:
            H = 3
    
    if num_cases > len(all_solutions):
        num_cases = len(all_solutions)
        print(f"Warning: Request to process {args.num_cases} cases, but only {len(all_solutions)} are available, will process all {num_cases} cases")
    
    solutions = random.sample(all_solutions, num_cases)
    os.makedirs(output_base, exist_ok=True)
    
    # Use tqdm to show the processing progress
    for case_id, grid in enumerate(tqdm(solutions, desc="Processing progress", unit="cases")):
        set_case_seed(case_id)
        
        case_dir = os.path.join(output_base, str(case_id))
        os.makedirs(case_dir, exist_ok=True)
        
        # Generate the three views
        draw_orthographic_view(grid, 'top', 
                              os.path.join(case_dir, '3view-top.png'), H)
        draw_orthographic_view(grid, 'front', 
                              os.path.join(case_dir, '3view-front.png'), H)
        draw_orthographic_view(grid, 'left', 
                              os.path.join(case_dir, '3view-left.png'), H)
        
        # If using random color, create the color mapping dictionary and color matrix
        color_map = {} if args.random_color else None
        C = len(grid)
        R = len(grid[0]) if C > 0 else 0
        
        # If using random color, first generate the color mapping of all blocks
        if args.random_color:
            for y in range(R):
                for x in range(C):
                    height = grid[x][y]
                    if height > 0:
                        for z in range(height):
                            # Randomly select from 6 colors
                            rgb_color = random.choice(COLOR_LIST)
                            color_id = RGB_TO_ID[rgb_color]
                            color_map[(x, y, z)] = (rgb_color, color_id)
        
        # Generate the 3D view
        draw_3d_view(grid, (1, 1, 1), 
                    os.path.join(case_dir, 'left-view-45.png'), H, args.random_color, color_map)
        draw_3d_view(grid, (-1, 1, 1), 
                    os.path.join(case_dir, 'right-view-45.png'), H, args.random_color, color_map)
        
        # Save the height matrix
        matrix_path = os.path.join(case_dir, 'matrix.json')
        with open(matrix_path, 'w') as f:
            json.dump(grid, f, indent=2)
        
        # If using random color, generate and save color.json
        if args.random_color and color_map:
            # Create the color matrix of R x C x H, initialized to 0
            color_matrix = [[[0 for _ in range(H)] for _ in range(C)] for _ in range(R)]
            
            # Fill the color matrix
            for (x, y, z), (rgb_color, color_id) in color_map.items():
                color_matrix[y][x][z] = color_id
            
            color_path = os.path.join(case_dir, 'color.json')
            with open(color_path, 'w') as f:
                json.dump(color_matrix, f, indent=2)
        
        # Calculate and save the total number of blocks
        total_blocks = sum(sum(row) for row in grid)
        answer_path = os.path.join(case_dir, 'answer.txt')
        with open(answer_path, 'w') as f:
            f.write(str(total_blocks))
    
    print("Done!")


if __name__ == '__main__':
    main()
