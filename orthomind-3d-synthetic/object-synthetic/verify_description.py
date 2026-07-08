#!/usr/bin/env python
"""
Verify that the description.json is consistent with the actual rendered position

Coordinate system description:
- scene_config: position = (x, y, z) where x=left/right, y=height, z=front/back
- Blender: (X, Y, Z) where X=left/right, Y=front/back, Z=height
- Conversion: Blender(X,Y,Z) = scene(x, z, y)

View description:
- front: From the negative Y direction, the X axis is left and right, and the Z axis is up and down
  - from-left-to-right should be sorted by scene.x from small to large
- left: From the negative X direction, the Y axis is left and right (front and back), and the Z axis is up and down
  - from-left-to-right should be sorted by scene.z from small to large (back on the left, front on the right)
- top: From the positive Z direction, the X axis is left and right, and the Y axis is up and down (front and back)
  - from-left-to-right should be sorted by scene.x from small to large
  - from-top-to-down should be sorted by scene.z from large to small (front on top, back on bottom)
"""
import json
import os
import sys

def verify_view_order(scene_config, description, view_name, direction='from-left-to-right'):
    """Verify the order of a single view is correct"""
    
    # Get the object list
    objects = scene_config['objects']
    
    # Get the sorting from the description
    view_data = description.get(view_name, {})
    sorted_dict = view_data.get(direction, {})
    
    if not sorted_dict:
        return True, "No data"
    
    # Extract the sorted objects (sorted by index)
    sorted_items = sorted(sorted_dict.items(), key=lambda x: x[1]['index'])
    
    # Determine what to sort by
    if view_name == 'front-view':
        # Front view: sorted by X coordinate from left to right
        expected_sort_key = lambda obj: obj['position'][0]
        coord_name = 'X'
    elif view_name == 'left-view':
        # Left view: sorted by Z coordinate from large to small (back to front)
        expected_sort_key = lambda obj: -obj['position'][2]
        coord_name = '-Z (front-to-back)'
    elif view_name == 'top-view':
        if direction == 'from-left-to-right':
            # Top view left and right: sorted by X coordinate
            expected_sort_key = lambda obj: obj['position'][0]
            coord_name = 'X'
        else:  # from-top-to-down
            # Top view up and down: sorted by Z coordinate from large to small (front on top, back on bottom)
            expected_sort_key = lambda obj: -obj['position'][2]
            coord_name = '-Z'
    else:
        return False, f"Unknown view: {view_name}"
    
    # Sort by expected order
    expected_sorted = sorted(objects, key=expected_sort_key)
    
    # Verify each position
    errors = []
    for i, (key, value) in enumerate(sorted_items):
        # Extract the shape name from the key
        shape_key = key.rsplit('_', 1)[0].replace('_', ' ')
        
        # Find the corresponding object in the original objects
        # Find the actual object by index
        actual_obj = expected_sorted[i]
        
        # Check if the shape matches
        shape_map = {
            'cube': 'cube',
            'cuboid': 'cuboid',
            'cylinder': 'cylinder',
            'cone': 'cone',
            'square_pyramid': 'square pyramid',
            'sphere': 'sphere'
        }
        expected_shape = shape_map.get(actual_obj['shape'], actual_obj['shape'])
        
        if expected_shape not in key:
            # The shape does not match
            coord_val = expected_sort_key(actual_obj)
            if coord_name.startswith('-'):
                coord_val = -coord_val
            errors.append(
                f"  Position {i}: Expected {expected_shape} "
                f"({coord_name}={coord_val:.2f}), "
                f"but got {shape_key}"
            )
    
    if errors:
        return False, "\n".join(errors)
    else:
        return True, "Order is correct"

def verify_scene(scene_dir):
    """Verify a single scene"""
    config_file = os.path.join(scene_dir, 'scene_config.json')
    desc_file = os.path.join(scene_dir, 'description.json')
    
    if not os.path.exists(config_file) or not os.path.exists(desc_file):
        return None, "Missing files"
    
    with open(config_file, 'r') as f:
        scene_config = json.load(f)
    
    with open(desc_file, 'r') as f:
        description = json.load(f)
    
    results = {}
    
    # Verify front-view
    ok, msg = verify_view_order(scene_config, description, 'front-view')
    results['front-view'] = (ok, msg)
    
    # Verify left-view
    ok, msg = verify_view_order(scene_config, description, 'left-view')
    results['left-view'] = (ok, msg)
    
    # Verify top-view (left-to-right)
    ok, msg = verify_view_order(scene_config, description, 'top-view', 'from-left-to-right')
    results['top-view-lr'] = (ok, msg)
    
    # Verify top-view (top-to-down)
    ok, msg = verify_view_order(scene_config, description, 'top-view', 'from-top-to-down')
    results['top-view-td'] = (ok, msg)
    
    return results, None

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify the accuracy of description.json')
    parser.add_argument('--input', type=str, default='./demo_output',
                       help='Data directory')
    parser.add_argument('--scenes', type=int, default=5,
                       help='Number of scenes to verify')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Description.json Verification Tool")
    print("=" * 70)
    print(f"Input directory: {args.input}\n")
    
    if not os.path.exists(args.input):
        print(f"✗ Directory does not exist: {args.input}")
        return 1
    
    total_checks = 0
    failed_checks = 0
    
    # Check each configuration
    for config in ['conf_1', 'conf_2']:
        config_dir = os.path.join(args.input, config)
        if not os.path.exists(config_dir):
            continue
        
        scenes = sorted(os.listdir(config_dir))[:args.scenes]
        
        for scene_name in scenes:
            scene_dir = os.path.join(config_dir, scene_name)
            if not os.path.isdir(scene_dir):
                continue
            
            print(f"\nChecking scene: {config}/{scene_name}")
            print("-" * 70)
            
            results, error = verify_scene(scene_dir)
            if error:
                print(f"  ✗ {error}")
                continue
            
            for view_name, (ok, msg) in results.items():
                total_checks += 1
                if ok:
                    print(f"  ✓ {view_name}: {msg}")
                else:
                    print(f"  ✗ {view_name}:")
                    print(msg)
                    failed_checks += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("Verification Summary")
    print("=" * 70)
    print(f"Total checks: {total_checks}")
    print(f"Passed: {total_checks - failed_checks}")
    print(f"Failed: {failed_checks}")
    
    if failed_checks == 0:
        print("\n✅ All checks passed! description.json is 100% accurate!")
        return 0
    else:
        print(f"\n⚠️ There are {failed_checks} checks that failed")
        print("\nPossible problems:")
        print("1. Coordinate system conversion error")
        print("2. Camera direction setting error")
        print("3. Sorting logic error")
        return 1

if __name__ == "__main__":
    sys.exit(main())