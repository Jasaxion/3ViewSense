import json
import random
import math
import os
import hashlib
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

@dataclass
class GeometryObject:
    """Geometry object class"""
    shape: str  # Shape: cube, cuboid, cylinder, cone, square_pyramid, sphere
    size: str   # Size: bigger, smaller
    color: Optional[str]  # Color (only for conf_2)
    position: Tuple[float, float, float]  # Position (x, y, z)
    rotation: float  # Rotation angle (around Y axis)
    dimensions: Tuple[float, float, float]  # Actual dimensions (width, height, depth)
    
    def get_identifier(self) -> str:
        """Get the object identifier"""
        if self.color:
            return f"{self.size}_{self.color}_{self.shape}"
        return f"{self.size}_{self.shape}"

class CollisionDetector:
    """Collision detector"""
    
    @staticmethod
    def check_collision(obj1: GeometryObject, obj2: GeometryObject, margin: float = 2.0) -> bool:
        x1, y1, z1 = obj1.position
        x2, y2, z2 = obj2.position
        w1, h1, d1 = obj1.dimensions
        w2, h2, d2 = obj2.dimensions
        
        # Consider the effect of rotation: use the maximum size of the object as the "radius"
        radius1 = max(w1, d1) / 2 + margin
        radius2 = max(w2, d2) / 2 + margin
        
        # Calculate the distance between the two object centers
        distance = math.sqrt((x1 - x2)**2 + (z1 - z2)**2)
        
        # If the distance is less than the sum of the two radii, it means they overlap
        if distance < (radius1 + radius2):
            return True
        return False

class PositionValidator:
    """Position validator - ensure that the object is either on the same axis, or the angle difference is greater than 10 degrees"""
    
    @staticmethod
    def calculate_angle(pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]) -> float:
        """Calculate the angle between two positions"""
        x1, _, z1 = pos1
        x2, _, z2 = pos2
        dx = x2 - x1
        dz = z2 - z1
        return math.degrees(math.atan2(dz, dx))
    
    @staticmethod
    def is_on_same_axis(pos1: Tuple[float, float, float], pos2: Tuple[float, float, float], 
                       threshold: float = 0.3) -> bool:
        """Check if two positions are on the same axis"""
        x1, _, z1 = pos1
        x2, _, z2 = pos2
        
        # Check if they are on the same X axis (Z is the same)
        if abs(z1 - z2) < threshold:
            return True
        # Check if they are on the same Z axis (X is the same)
        if abs(x1 - x2) < threshold:
            return True
        return False
    
    @staticmethod
    def validate_position(new_pos: Tuple[float, float, float], 
                         existing_objects: List[GeometryObject],
                         min_angle_diff: float = 10.0) -> bool:
        """Validate new position satisfies angle requirements"""
        for obj in existing_objects:
            if PositionValidator.is_on_same_axis(new_pos, obj.position):
                continue
            
            angle_diff = abs(PositionValidator.calculate_angle(new_pos, obj.position) - 
                           PositionValidator.calculate_angle(obj.position, new_pos))
            
            # Standardize angle difference
            angle_diff = angle_diff % 180
            if angle_diff > 90:
                angle_diff = 180 - angle_diff
                
            if angle_diff < min_angle_diff and angle_diff > 0.1:
                return False
        
        return True

class SceneGenerator:
    """Scene generator"""
    
    SHAPES = ['cube', 'cuboid', 'cylinder', 'cone', 'square_pyramid', 'sphere']
    SIZES = ['bigger', 'smaller']
    COLORS = ['black', 'white', 'red', 'yellow', 'blue', 'green']
    
    # Size mapping (base size)
    SIZE_MAP = {
        'bigger': 2.0,
        'smaller': 1.0
    }
    
    def __init__(self, config_type: str, max_retries: int = 1000):
        self.config_type = config_type  # 'conf_1' or 'conf_2'
        self.max_retries = max_retries
        self.generated_scenes = set()  # Record generated scene hashes
        
    def get_shape_dimensions(self, shape: str, size: str) -> Tuple[float, float, float]:
        """Get shape dimensions"""
        base_size = self.SIZE_MAP[size]
        
        dimensions = {
            'cube': (base_size, base_size, base_size),
            'cuboid': (base_size * 1.5, base_size, base_size * 0.8),
            'cylinder': (base_size, base_size * 1.2, base_size),
            'cone': (base_size, base_size * 1.5, base_size),
            'square_pyramid': (base_size, base_size * 1.3, base_size),
            'sphere': (base_size, base_size, base_size)
        }
        return dimensions[shape]
    
    def generate_scene_config_1(self, num_objects: int) -> List[GeometryObject]:
        """Generate scene for config_1 (at most two same shapes, dimensions need to be different)"""
        objects = []
        shape_count = defaultdict(int)
        
        for _ in range(num_objects):
            retry_count = 0
            while retry_count < self.max_retries:
                shape = random.choice(self.SHAPES)
                
                # If the shape already has two, skip
                if shape_count[shape] >= 2:
                    retry_count += 1
                    continue
                
                # If the shape already has one, ensure dimensions are different
                if shape_count[shape] == 1:
                    existing_size = [obj.size for obj in objects if obj.shape == shape][0]
                    size = 'smaller' if existing_size == 'bigger' else 'bigger'
                else:
                    size = random.choice(self.SIZES)
                
                position = self._generate_valid_position(objects)
                if position is None:
                    retry_count += 1
                    continue
                
                rotation = random.uniform(0, 360)
                dimensions = self.get_shape_dimensions(shape, size)
                
                obj = GeometryObject(shape, size, None, position, rotation, dimensions)
                
                if not self._has_collision(obj, objects):
                    objects.append(obj)
                    shape_count[shape] += 1
                    break
                
                retry_count += 1
            
            if retry_count >= self.max_retries:
                print(f"Warning: cannot place the {len(objects)+1}th object")
                break
        
        return objects
    
    def generate_scene_config_2(self, num_objects: int) -> List[GeometryObject]:
        """Generate scene for config_2"""
        objects = []
        used_combinations = set()
        shape_counts = defaultdict(int)  # Count the number of each shape
        
        for i in range(num_objects):
            retry_count = 0
            while retry_count < self.max_retries:
                # Encourage reuse of existing shapes (60% probability)
                if objects and random.random() < 0.6:
                    # Randomly select one from existing shapes
                    existing_shapes = [obj.shape for obj in objects]
                    shape = random.choice(existing_shapes)
                else:
                    # Randomly select new shape
                    shape = random.choice(self.SHAPES)
                
                size = random.choice(self.SIZES)
                color = random.choice(self.COLORS)
                
                combination = (shape, size, color)
                if combination in used_combinations:
                    retry_count += 1
                    continue
                
                position = self._generate_valid_position(objects)
                if position is None:
                    retry_count += 1
                    continue
                
                rotation = random.uniform(0, 360)
                dimensions = self.get_shape_dimensions(shape, size)
                
                obj = GeometryObject(shape, size, color, position, rotation, dimensions)
                
                if not self._has_collision(obj, objects):
                    objects.append(obj)
                    used_combinations.add(combination)
                    shape_counts[shape] += 1
                    break
                
                retry_count += 1
            
            if retry_count >= self.max_retries:
                print(f"Warning: cannot place the {len(objects)+1}th object")
                break
        
        return objects
    
    def _generate_valid_position(self, existing_objects: List[GeometryObject]) -> Optional[Tuple[float, float, float]]:
        """Generate valid position"""
        for _ in range(50):
            # Generate position randomly on the plane
            x = random.uniform(-8, 8)
            z = random.uniform(-8, 8)
            y = 0  # Put on the ground
            
            position = (x, y, z)
            
            # Validate position satisfies angle requirements
            if PositionValidator.validate_position(position, existing_objects):
                return position
        
        return None
    
    def _has_collision(self, new_obj: GeometryObject, existing_objects: List[GeometryObject]) -> bool:
        """Check if new object collides with existing objects"""
        for obj in existing_objects:
            if CollisionDetector.check_collision(new_obj, obj):
                return True
        return False
    
    def generate_scene(self, num_objects: int) -> Optional[List[GeometryObject]]:
        """Generate scene"""
        if self.config_type == 'conf_1':
            scene = self.generate_scene_config_1(num_objects)
        else:
            scene = self.generate_scene_config_2(num_objects)
        
        # Check for duplicates
        scene_hash = self._hash_scene(scene)
        if scene_hash in self.generated_scenes:
            return None
        
        self.generated_scenes.add(scene_hash)
        return scene
    
    def _hash_scene(self, scene: List[GeometryObject]) -> str:
        """Generate hash value of scene"""
        scene_str = ""
        for obj in sorted(scene, key=lambda x: (x.shape, x.size, x.color or '')):
            scene_str += f"{obj.shape}_{obj.size}_{obj.color or 'none'}_"
        return hashlib.md5(scene_str.encode()).hexdigest()

class DescriptionGenerator:
    """Description file generator"""
    
    SHAPE_NAME_MAP = {
        'cube': 'cube',
        'cuboid': 'cuboid',
        'cylinder': 'cylinder',
        'cone': 'cone',
        'square_pyramid': 'square pyramid',
        'sphere': 'sphere'
    }
    COLORLESS_COUNT_PROB = 0.8
    
    @staticmethod
    def generate_description(scene: List[GeometryObject], config_type: str, 
                           views_data: Dict) -> Dict:
        """Generate complete description file"""
        description = {
            "type": "normal" if config_type == "conf_1" else "color",
            "count": DescriptionGenerator._generate_count(scene, config_type),
            "front-view": views_data.get('front', {}),
            "left-view": views_data.get('left', {}),
            "top-view": views_data.get('top', {}),
            "query": DescriptionGenerator._generate_queries(scene, config_type)
        }
        return description
    
    @staticmethod
    def _generate_count(scene: List[GeometryObject], config_type: str) -> Dict:
        """Generate object counting"""
        count = {}
        
        for obj in scene:
            shape_name = DescriptionGenerator.SHAPE_NAME_MAP[obj.shape]
            if shape_name not in count:
                count[shape_name] = {}
            
            if config_type == "conf_1":
                count[shape_name]["normal"] = count[shape_name].get("normal", 0) + 1
            else:
                color = obj.color
                count[shape_name][color] = count[shape_name].get(color, 0) + 1
        
        return count
    
    @staticmethod
    def _generate_queries(scene: List[GeometryObject], config_type: str) -> Dict:
        """Generate query problems and answers
        
        Template 1: Direction judgment (supports multiple possible answers)
        Template 2: Counting problem (determines the answer)
        """
        queries = {
            "template_1": [],
            "template_2": []
        }
        
        # Count the number of each shape
        shape_counts = {}
        for obj in scene:
            shape_counts[obj.shape] = shape_counts.get(obj.shape, 0) + 1
        
        # Generate 3 template 1 questions (direction questions)
        for _ in range(min(3, len(scene) * (len(scene) - 1) // 2)):
            if len(scene) >= 2:
                obj1, obj2 = random.sample(scene, 2)
                directions = DescriptionGenerator._calculate_direction_with_tolerance(obj1, obj2)
                
                # Only add size when there are multiple instances of the shape
                obj1_desc = DescriptionGenerator._get_object_description(
                    obj1, config_type, shape_counts)
                obj2_desc = DescriptionGenerator._get_object_description(
                    obj2, config_type, shape_counts)
                
                problem = (
                    f'<image>\n In which direction is the {obj1_desc} relative to the {obj2_desc}? '
                    f'Answer the question from the choices: '
                    f'["front", "back", "right", "left", "front left", "front right", "back left", "back right"]'
                )
                
                queries["template_1"].append({
                    "problem": problem,
                    "answer": directions  # List, may have multiple answers
                })
        
        # Generate 3 template 2 questions (counting problems)
        counted_types = set()
        color_targets = {(o.shape, o.color) for o in scene} if config_type == "conf_2" else set()
        shape_only_targets = {(shape, None) for shape, cnt in shape_counts.items() if cnt > 1}
        available_count_targets = (
            color_targets | shape_only_targets if config_type == "conf_2"
            else {(o.shape, None) for o in scene}
        )
        
        max_count_questions = min(3, len(available_count_targets))
        attempts = 0
        max_attempts = len(scene) * 10 if scene else 0
        
        while len(queries["template_2"]) < max_count_questions and attempts < max_attempts:
            attempts += 1
            obj = random.choice(scene)
            
            use_color = (config_type == "conf_2")
            obj_type = (obj.shape, obj.color if use_color else None)
            
            if config_type == "conf_2":
                shape_only_allowed = (obj.shape, None) in shape_only_targets
                if shape_only_allowed and random.random() < DescriptionGenerator.COLORLESS_COUNT_PROB:
                    use_color = False
                    obj_type = (obj.shape, None)
            
            if obj_type not in available_count_targets or obj_type in counted_types:
                continue
            counted_types.add(obj_type)
            
            # Counting problems do not need size modifiers; decide the description based on whether color is used
            obj_desc = DescriptionGenerator._get_object_description(
                obj, config_type, shape_counts, for_counting=True, include_color=use_color)
            
            # Calculate the number of objects of this type
            count = sum(
                1 for o in scene
                if DescriptionGenerator._objects_match(o, obj, config_type, ignore_color=not use_color)
            )
            
            problem = f'<image>\n How many {obj_desc} are there in this image? Return your final response within \\boxed{{}}.'
            
            queries["template_2"].append({
                "problem": problem,
                "answer": count  # Integer
            })
        
        return queries
    
    @staticmethod
    def _get_object_description(obj: GeometryObject, config_type: str, 
                               shape_counts: dict, for_counting: bool = False,
                               include_color: Optional[bool] = None) -> str:
        """Get object description"""
        parts = []
        
        # Only add size when there are multiple instances of the shape and it is not a counting problem
        if not for_counting and shape_counts.get(obj.shape, 0) > 1 and obj.size:
            parts.append(obj.size)
        
        if include_color is None:
            include_color = (config_type == "conf_2")
        
        if include_color and config_type == "conf_2" and obj.color:
            parts.append(obj.color)
        
        parts.append(DescriptionGenerator.SHAPE_NAME_MAP[obj.shape])
        return " ".join(parts)
    
    @staticmethod
    def _objects_match(obj1: GeometryObject, obj2: GeometryObject, config_type: str,
                       ignore_color: bool = False) -> bool:
        """Check if two objects match (used for counting)"""
        if obj1.shape != obj2.shape:
            return False
        if config_type == "conf_2" and not ignore_color and obj1.color != obj2.color:
            return False
        return True
    
    @staticmethod
    def _calculate_direction_with_tolerance(obj1: GeometryObject, obj2: GeometryObject) -> list:
        """Calculate the direction of obj1 relative to obj2, with tolerance for eight directions.
        
        Coordinate system: X to the right, Z decreasing means closer to the observer (i.e., "front"). Angle increases clockwise from front.
        Rules:
        - Angle <1° with main direction: return single direction;
        - Angle 1°~8° with main direction and main direction is front/back/left/right: return two directions, the second direction depends on the left/right offset.
        """
        x1, _, z1 = obj1.position
        x2, _, z2 = obj2.position
        
        dx = x1 - x2  # Positive value means right
        # Rendering camera located at negative Z direction (reference 3D_view.png), smaller Z means closer to the camera, i.e., "front"
        dz = z2 - z1  # After conversion, positive value means direction towards the camera (front)
        
        diagonal_primary = None
        diagonal_threshold = 1.0
        if abs(dx) >= diagonal_threshold and abs(dz) >= diagonal_threshold:
            fb = "front" if dz > 0 else "back"
            lr = "right" if dx > 0 else "left"
            diagonal_primary = f"{fb} {lr}"
        
        # Calculate angle (clockwise from front, 0-360 degrees)
        angle = math.degrees(math.atan2(dx, dz))  # atan2(x, z)
        if angle < 0:
            angle += 360
        
        def angular_delta(a: float, b: float) -> float:
            """Return the shortest angle difference between a and b, range [-180, 180)."""
            return ((a - b + 180) % 360) - 180
        
        direction_defs = [
            {"name": "front", "center": 0, "left": "front left", "right": "front right"},
            {"name": "front right", "center": 45},
            {"name": "right", "center": 90, "left": "front right", "right": "back right"},
            {"name": "back right", "center": 135},
            {"name": "back", "center": 180, "left": "back right", "right": "back left"},
            {"name": "back left", "center": 225},
            {"name": "left", "center": 270, "left": "back left", "right": "front left"},
            {"name": "front left", "center": 315},
        ]
        
        primary_info = min(direction_defs, key=lambda d: abs(angular_delta(angle, d["center"])))
        delta = angular_delta(angle, primary_info["center"])
        offset = abs(delta)
        
        if offset < 1:
            directions = [primary_info["name"]]
        elif offset < 8 and " " not in primary_info["name"]:
            secondary = primary_info["right"] if delta > 0 else primary_info["left"]
            if secondary:
                directions = [primary_info["name"], secondary]
            else:
                directions = [primary_info["name"]]
        else:
            directions = [primary_info["name"]]
        
        if diagonal_primary:
            if diagonal_primary in directions:
                # Ensure diagonal direction is returned first
                directions.remove(diagonal_primary)
                directions.insert(0, diagonal_primary)
            else:
                directions.insert(0, diagonal_primary)
        return directions
    
    @staticmethod
    def _calculate_direction(obj1: GeometryObject, obj2: GeometryObject) -> str:
        """Calculate the direction of obj1 relative to obj2"""
        x1, _, z1 = obj1.position
        x2, _, z2 = obj2.position
        
        dx = x1 - x2
        dz = z1 - z2
        
        # Define direction threshold
        threshold = 1.0
        
        if abs(dx) < threshold and dz > threshold:
            return "front"
        elif abs(dx) < threshold and dz < -threshold:
            return "back"
        elif dx > threshold and abs(dz) < threshold:
            return "right"
        elif dx < -threshold and abs(dz) < threshold:
            return "left"
        elif dx > threshold and dz > threshold:
            return "front right"
        elif dx > threshold and dz < -threshold:
            return "back right"
        elif dx < -threshold and dz > threshold:
            return "front left"
        elif dx < -threshold and dz < -threshold:
            return "back left"
        else:
            return "front"

class DatasetGenerator:
    """Dataset generator main class"""
    
    def __init__(self, output_dir: str = "./output", max_num: int = 100):
        self.output_dir = output_dir
        self.max_num = max_num
        self.renderer_ready = False
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "conf_1"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "conf_2"), exist_ok=True)
    
    def generate_dataset(self):
        """Generate complete dataset"""
        print("Starting to generate dataset...")
        
        # Generate data for each configuration
        for config in ['conf_1', 'conf_2']:
            print(f"\nGenerating configuration: {config}")
            generator = SceneGenerator(config)
            
            generated_count = 0
            attempt_count = 0
            max_attempts = self.max_num * 3
            
            while generated_count < self.max_num and attempt_count < max_attempts:
                attempt_count += 1
                num_objects = random.randint(2, 12)
                
                scene = generator.generate_scene(num_objects)
                if scene is None:
                    continue
                
                # Generate ID
                scene_id = f"{config}_{str(generated_count + 1).zfill(3)}"
                
                # Save scene data
                self._save_scene(scene, config, scene_id)
                
                generated_count += 1
                print(f"Generated: {generated_count}/{self.max_num}")
        
        print("\nDataset generation completed!")
    
    def _save_scene(self, scene: List[GeometryObject], config: str, scene_id: str):
        """Save scene data"""
        scene_dir = os.path.join(self.output_dir, config, scene_id)
        os.makedirs(scene_dir, exist_ok=True)
        
        # Save scene configuration (for renderer)
        scene_config = {
            "objects": [asdict(obj) for obj in scene],
            "config_type": config
        }
        
        config_file = os.path.join(scene_dir, "scene_config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(scene_config, f, indent=2, ensure_ascii=False)
        
        # Generate description file (temporarily using mock views data)
        views_data = self._generate_mock_views_data(scene, config)
        description = DescriptionGenerator.generate_description(scene, config, views_data)
        
        desc_file = os.path.join(scene_dir, "description.json")
        with open(desc_file, 'w', encoding='utf-8') as f:
            json.dump(description, f, indent=2, ensure_ascii=False)
        
        print(f"Scene saved: {scene_id}")
    
    def _generate_mock_views_data(self, scene: List[GeometryObject], config: str) -> Dict:
        """Generate mock views data (requires Three.js renderer to provide real data)"""
        # Here generate the basic view data structure
        # Actual view analysis needs to be done during rendering
        views_data = {
            'front': {
                'can-see': DescriptionGenerator._generate_count(scene, config),
                'from-left-to-right': {}
                # 'hidden-geometry': {}
            },
            'left': {
                'can-see': DescriptionGenerator._generate_count(scene, config),
                'from-left-to-right': {}
                # 'hidden-geometry': {}
            },
            'top': {
                'can-see': DescriptionGenerator._generate_count(scene, config),
                'from-left-to-right': {},
                'from-top-to-down': {}
            }
        }
        return views_data

# Test
if __name__ == "__main__":
    # Generate dataset
    generator = DatasetGenerator(output_dir="./output", max_num=10)
    generator.generate_dataset()