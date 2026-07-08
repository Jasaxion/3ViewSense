import json
import random
import os
from geometry_generator import (
    SceneGenerator, 
    DescriptionGenerator, 
    GeometryObject
)

class SimpleDatasetGenerator:
    """Simplified dataset generator (without image rendering)"""
    
    def __init__(self, output_dir: str = "./output", max_num: int = 100):
        self.output_dir = output_dir
        self.max_num = max_num
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "conf_1"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "conf_2"), exist_ok=True)
    
    def generate_dataset(self):
        """Generate dataset (only configuration and description files)"""
        print("=" * 60)
        print("Starting to generate geometry object configuration data")
        print("Note: This is a simplified version, does not generate images")
        print("=" * 60)
        
        # Generate data for each configuration
        for config in ['conf_1', 'conf_2']:
            print(f"\n{'=' * 60}")
            print(f"Generating configuration: {config}")
            print(f"{'=' * 60}")
            
            generator = SceneGenerator(config)
            generated_count = 0
            attempt_count = 0
            max_attempts = self.max_num * 3
            
            while generated_count < self.max_num and attempt_count < max_attempts:
                attempt_count += 1
                # Configuration 2 supports more objects (up to 20)
                if config == 'conf_2':
                    num_objects = random.randint(2, 20)
                else:
                    num_objects = random.randint(2, 12)
                
                scene = generator.generate_scene(num_objects)
                if scene is None:
                    continue
                
                # Generate ID
                scene_id = f"{config}_{str(generated_count + 1).zfill(3)}"
                
                # Save scene data
                self._save_scene(scene, config, scene_id)
                
                generated_count += 1
                if generated_count % 5 == 0 or generated_count == self.max_num:
                    print(f"Progress: {generated_count}/{self.max_num}")
        
        print(f"\n{'=' * 60}")
        print("Configuration data generation completed!")
        print(f"Output directory: {self.output_dir}")
        print("Note: Use integrated_generator.py to generate complete images")
        print(f"{'=' * 60}")
    
    def _save_scene(self, scene, config, scene_id):
        """Save scene configuration and description"""
        scene_dir = os.path.join(self.output_dir, config, scene_id)
        os.makedirs(scene_dir, exist_ok=True)
        
        # Save scene configuration (for renderer)
        scene_config = {
            "config_type": config,
            "objects": [self._object_to_dict(obj) for obj in scene]
        }
        
        config_file = os.path.join(scene_dir, "scene_config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(scene_config, f, indent=2, ensure_ascii=False)
        
        # Generate description file
        views_data = self._generate_simplified_views_data(scene, config)
        description = DescriptionGenerator.generate_description(scene, config, views_data)
        
        desc_file = os.path.join(scene_dir, "description.json")
        with open(desc_file, 'w', encoding='utf-8') as f:
            json.dump(description, f, indent=2, ensure_ascii=False)
        
        # Generate readable scene summary
        summary_file = os.path.join(scene_dir, "scene_summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Scene ID: {scene_id}\n")
            f.write(f"Configuration type: {config}\n")
            f.write(f"Object count: {len(scene)}\n")
            f.write(f"\nObject list:\n")
            f.write("=" * 60 + "\n")
            
            for i, obj in enumerate(scene, 1):
                f.write(f"\nObject {i}:\n")
                f.write(f"  Shape: {obj.shape}\n")
                f.write(f"  Size: {obj.size}\n")
                if obj.color:
                    f.write(f"  Color: {obj.color}\n")
                f.write(f"  Position: ({obj.position[0]:.2f}, {obj.position[1]:.2f}, {obj.position[2]:.2f})\n")
                f.write(f"  Rotation: {obj.rotation:.2f}°\n")
                f.write(f"  Size: ({obj.dimensions[0]:.2f}, {obj.dimensions[1]:.2f}, {obj.dimensions[2]:.2f})\n")
    
    def _object_to_dict(self, obj: GeometryObject) -> dict:
        """Convert object object to dictionary"""
        return {
            "shape": obj.shape,
            "size": obj.size,
            "color": obj.color,
            "position": list(obj.position),
            "rotation": obj.rotation,
            "dimensions": list(obj.dimensions)
        }
    
    def _generate_simplified_views_data(self, scene, config):
        """Generate simplified view data (based on geometric calculation)"""
        # This is a simplified version, using basic geometric relations
        # Actual occlusion detection needs to be done during rendering
        
        count = DescriptionGenerator._generate_count(scene, config)
        
        views_data = {
            'front': {
                'can-see': count.copy(),
                'from-left-to-right': self._sort_objects_left_to_right(scene, 'front')
                # 'hidden-geometry': {}
            },
            'left': {
                'can-see': count.copy(),
                'from-left-to-right': self._sort_objects_left_to_right(scene, 'left')
                # 'hidden-geometry': {}
            },
            'top': {
                'can-see': count.copy(),
                'from-left-to-right': self._sort_objects_left_to_right(scene, 'top'),
                'from-top-to-down': self._sort_objects_top_to_down(scene)
            }
        }
        
        return views_data
    
    def _sort_objects_left_to_right(self, scene, view_type):
        """Sort objects from left to right
        
        Fix: Only add size property when there are multiple objects of the same shape
        """
        result = {}
        
        if view_type == 'front':
            # Front view: from negative Y direction, X axis from left to right
            sorted_objects = sorted(scene, key=lambda obj: obj.position[0])
        elif view_type == 'left':
            # Left view: from negative X direction, Y axis from left to right
            # From left to right = front to back = Z from large to small (scene.position[2] corresponds to Blender's Y)
            sorted_objects = sorted(scene, key=lambda obj: -obj.position[2])
        else:  # top
            # Top view: sort by X coordinate
            sorted_objects = sorted(scene, key=lambda obj: obj.position[0])
        
        # Count the number of each shape
        shape_counts = {}
        for obj in scene:
            shape_name = DescriptionGenerator.SHAPE_NAME_MAP[obj.shape]
            shape_counts[shape_name] = shape_counts.get(shape_name, 0) + 1
        
        for i, obj in enumerate(sorted_objects):
            shape_name = DescriptionGenerator.SHAPE_NAME_MAP[obj.shape]
            key = f"{shape_name}_{i}"
            
            obj_info = {"index": i}
            
            # Only add size property when there are multiple instances of the same shape
            if shape_counts[shape_name] > 1:
                obj_info["size"] = obj.size
            
            if obj.color:
                obj_info["color"] = obj.color
            
            result[key] = obj_info
        
        return result
    
    def _sort_objects_top_to_down(self, scene):
        """Sort objects from top to bottom (Z axis of top view)
        
        Fix: Only add size property when there are multiple instances of the same shape
        """
        result = {}
        sorted_objects = sorted(scene, key=lambda obj: -obj.position[2])  # Negative sign means from large to small
        
        # Count the number of each shape
        shape_counts = {}
        for obj in scene:
            shape_name = DescriptionGenerator.SHAPE_NAME_MAP[obj.shape]
            shape_counts[shape_name] = shape_counts.get(shape_name, 0) + 1
        
        for i, obj in enumerate(sorted_objects):
            shape_name = DescriptionGenerator.SHAPE_NAME_MAP[obj.shape]
            key = f"{shape_name}_{i}"
            
            obj_info = {"index": i}
            
            # Only add size property when there are multiple instances of the same shape
            if shape_counts[shape_name] > 1:
                obj_info["size"] = obj.size
            
            if obj.color:
                obj_info["color"] = obj.color
            
            result[key] = obj_info
        
        return result

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simplified geometry object data generator (only configuration files)')
    parser.add_argument('--output', type=str, default='./output_simple', 
                       help='Output directory (default: ./output_simple)')
    parser.add_argument('--max-num', type=int, default=10,
                       help='Number of scenes generated for each configuration (default: 10)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("Simplified geometry object data generator")
    print("=" * 60)
    print(f"Output directory: {args.output}")
    print(f"Number of scenes generated for each configuration: {args.max_num}")
    print("=" * 60 + "\n")
    
    # Generate dataset
    generator = SimpleDatasetGenerator(
        output_dir=args.output,
        max_num=args.max_num
    )
    
    generator.generate_dataset()
    
    print("\nNote:")
    print("1. Generated configuration files can be used for subsequent rendering")
    print("2. To generate images, please use: python integrated_generator.py")
    print("3. View the generated scene_summary.txt to understand the scene details\n")

if __name__ == "__main__":
    main()

# python scene_generator.py --max-num 10 --output ./test_final_fix