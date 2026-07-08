#!/usr/bin/env python
import json
import os
import sys
import math

def verify_query_format(description):
    """Verify query format"""
    errors = []
    
    query = description.get('query', {})
    
    # Check template_1
    template_1 = query.get('template_1', [])
    if not isinstance(template_1, list):
        errors.append("template_1 should be a list")
    else:
        for i, item in enumerate(template_1):
            if not isinstance(item, dict):
                errors.append(f"template_1[{i}] should be a dictionary")
                continue
            if 'problem' not in item:
                errors.append(f"template_1[{i}] missing problem field")
            if 'answer' not in item:
                errors.append(f"template_1[{i}] missing answer field")
            elif not isinstance(item['answer'], list):
                errors.append(f"template_1[{i}].answer should be a list")
            else:
                # Verify that the answer is within the valid range
                valid_directions = ["front", "back", "right", "left", 
                                   "front left", "front right", "back left", "back right"]
                for ans in item['answer']:
                    if ans not in valid_directions:
                        errors.append(f"template_1[{i}].answer contains invalid direction: {ans}")
    
    # Check template_2
    template_2 = query.get('template_2', [])
    if not isinstance(template_2, list):
        errors.append("template_2 should be a list")
    else:
        for i, item in enumerate(template_2):
            if not isinstance(item, dict):
                errors.append(f"template_2[{i}] should be a dictionary")
                continue
            if 'problem' not in item:
                errors.append(f"template_2[{i}] missing problem field")
            if 'answer' not in item:
                errors.append(f"template_2[{i}] missing answer field")
            elif not isinstance(item['answer'], int):
                errors.append(f"template_2[{i}].answer should be an integer, but {type(item['answer'])}")
    
    return errors

def verify_count_answer(scene_config, description, config_type):
    """Verify the correctness of the count answer"""
    errors = []
    
    objects = scene_config['objects']
    template_2 = description.get('query', {}).get('template_2', [])
    
    for i, item in enumerate(template_2):
        answer = item.get('answer')
        problem = item.get('problem', '')
        
        # Extract the object type from the problem
        # "How many red cube are there..."
        if not problem:
            continue
        
        # Simple parsing: extract the shape name
        shape_names = ['cube', 'cuboid', 'cylinder', 'cone', 'square pyramid', 'sphere']
        colors = ['red', 'blue', 'green', 'yellow', 'white', 'black']
        
        found_shape = None
        found_color = None
        
        for shape in shape_names:
            if shape in problem.lower():
                found_shape = shape.replace(' ', '_')
                break
        
        if config_type == 'conf_2':
            for color in colors:
                if color in problem.lower():
                    found_color = color
                    break
        
        if not found_shape:
            errors.append(f"template_2[{i}]: cannot extract shape from the problem")
            continue
        
        # Calculate the actual number
        actual_count = 0
        for obj in objects:
            if obj['shape'] == found_shape:
                if config_type == 'conf_1' or found_color is None or obj.get('color') == found_color:
                    actual_count += 1
        
        if answer != actual_count:
            errors.append(
                f"template_2[{i}]: count mismatch - problem: {problem[:50]}..., "
                f"answer: {answer}, actual: {actual_count}"
            )
    
    return errors

def calculate_angle(obj1, obj2):
    """Calculate the angle of obj1 relative to obj2"""
    x1, _, z1 = obj1['position']
    x2, _, z2 = obj2['position']
    
    dx = x1 - x2
    dz = z1 - z2
    
    angle = math.degrees(math.atan2(dx, dz))
    if angle < 0:
        angle += 360
    
    return angle

def get_expected_directions(angle):
    """Get the expected direction based on the angle"""
    # Angle ranges for the eight directions
    if angle < 22.5 or angle >= 337.5:
        primary = "front"
    elif 22.5 <= angle < 67.5:
        primary = "front right"
    elif 67.5 <= angle < 112.5:
        primary = "right"
    elif 112.5 <= angle < 157.5:
        primary = "back right"
    elif 157.5 <= angle < 202.5:
        primary = "back"
    elif 202.5 <= angle < 247.5:
        primary = "back left"
    elif 247.5 <= angle < 292.5:
        primary = "left"
    else:
        primary = "front left"
    
    return [primary]

def verify_scene(scene_dir, config_type):
    """Verify a single scene"""
    config_file = os.path.join(scene_dir, 'scene_config.json')
    desc_file = os.path.join(scene_dir, 'description.json')
    
    if not os.path.exists(config_file) or not os.path.exists(desc_file):
        return None, "missing file"
    
    with open(config_file, 'r') as f:
        scene_config = json.load(f)
    
    with open(desc_file, 'r') as f:
        description = json.load(f)
    
    errors = []
    
    # Check format
    format_errors = verify_query_format(description)
    errors.extend(format_errors)
    
    # Check count answer
    if not format_errors:  # Only check content if the format is correct
        count_errors = verify_count_answer(scene_config, description, config_type)
        errors.extend(count_errors)
    
    return errors, None

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify query answer')
    parser.add_argument('--input', type=str, default='./demo_output',
                       help='Data directory')
    parser.add_argument('--scenes', type=int, default=5,
                       help='Number of scenes to verify')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Query answer verification tool")
    print("=" * 70)
    print(f"Input directory: {args.input}\n")
    
    if not os.path.exists(args.input):
        print(f"✗ Directory does not exist: {args.input}")
        return 1
    
    total_checks = 0
    failed_checks = 0
    
    for config in ['conf_1', 'conf_2']:
        config_dir = os.path.join(args.input, config)
        if not os.path.exists(config_dir):
            continue
        
        print(f"\n{'=' * 70}")
        print(f"Check configuration: {config}")
        print('=' * 70)
        
        scenes = sorted(os.listdir(config_dir))[:args.scenes]
        
        for scene_name in scenes:
            scene_dir = os.path.join(config_dir, scene_name)
            if not os.path.isdir(scene_dir):
                continue
            
            total_checks += 1
            
            errors, error_msg = verify_scene(scene_dir, config)
            if error_msg:
                print(f"\n✗ {scene_name}: {error_msg}")
                failed_checks += 1
                continue
            
            if errors:
                print(f"\n✗ {scene_name}:")
                for err in errors:
                    print(f"  - {err}")
                failed_checks += 1
            else:
                print(f"✓ {scene_name}: All checks passed")
    
    # Summary
    print("\n" + "=" * 70)
    print("Verification summary")
    print("=" * 70)
    print(f"Total scenes: {total_checks}")
    print(f"Passed: {total_checks - failed_checks}")
    print(f"Failed: {failed_checks}")
    
    if failed_checks == 0:
        print("\n✅ All query answers verified!")
        return 0
    else:
        print(f"\n⚠️ There are {failed_checks} scenes that failed verification")
        return 1

if __name__ == "__main__":
    sys.exit(main())