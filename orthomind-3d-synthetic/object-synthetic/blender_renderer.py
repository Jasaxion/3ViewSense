import json
import os
import sys
from pathlib import Path

# Check if in Blender environment
try:
    import bpy
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False
    print("⚠️  No Blender environment detected")
    print("\nUsage:")
    print("  1. Install Blender: https://www.blender.org/download/")
    print("  2. Run: blender --background --python blender_renderer.py -- --input ./data")

if IN_BLENDER:
    import mathutils
    import math

class BlenderRenderer:
    def __init__(self, use_gpu=True):
        if not IN_BLENDER:
            raise RuntimeError("Must run in Blender environment")
        
        self.use_gpu = use_gpu
        self.setup_scene()
        
    def setup_scene(self):
        """Setup Blender scene - optimize performance"""
        # Clear default objects
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        
        # ========================================
        # Use EEVEE engine (10-20 times faster than CYCLES)
        # ========================================
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        
        # EEVEE optimization settings (compatible with different versions)
        eevee = bpy.context.scene.eevee
        
        # Basic settings (all versions supported)
        eevee.taa_render_samples = 16  # Reduce sampling (default 64)
        
        # Optional settings (check if the attribute exists)
        if hasattr(eevee, 'use_gtao'):
            eevee.use_gtao = True  # Ambient occlusion
            eevee.gtao_distance = 1.0
        
        if hasattr(eevee, 'use_bloom'):
            eevee.use_bloom = False  # Disable bloom effect
        
        if hasattr(eevee, 'use_ssr'):
            eevee.use_ssr = False  # Disable screen space reflection
        
        # GPU acceleration (if available)
        if self.use_gpu:
            try:
                prefs = bpy.context.preferences.addons.get('cycles')
                if prefs:
                    prefs.preferences.compute_device_type = 'CUDA'
                bpy.context.scene.cycles.device = 'GPU'
                print("  ✓ GPU acceleration enabled")
            except:
                print("  ℹ GPU acceleration not available, using CPU")
        
        # Set resolution (reduce resolution to improve speed)
        bpy.context.scene.render.resolution_x = 1280
        bpy.context.scene.render.resolution_y = 720
        bpy.context.scene.render.resolution_percentage = 100
        
        # ========================================
        # Set light gray background, avoid white objects confusing
        # ========================================
        bpy.context.scene.render.film_transparent = False
        bpy.context.scene.world.use_nodes = True
        world_bg = bpy.context.scene.world.node_tree.nodes.get("Background")
        if world_bg:
            # Use light gray blue background (RGB: 0.85, 0.88, 0.92)
            world_bg.inputs[0].default_value = (0.85, 0.88, 0.92, 1.0)
        
        bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, 0))
        ground = bpy.context.active_object
        ground.name = "Ground"
        
        # Ground material - receive shadows, use slightly darker gray
        mat = bpy.data.materials.new(name="GroundMaterial")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        bsdf = nodes.get("Principled BSDF")
        if bsdf:
            # Ground uses medium gray, with contrast to background and white objects
            bsdf.inputs['Base Color'].default_value = (0.75, 0.75, 0.75, 1.0)
            bsdf.inputs['Roughness'].default_value = 0.9
            bsdf.inputs['Metallic'].default_value = 0.0
        ground.data.materials.append(mat)
        
        # ========================================
        # More natural lighting
        # ========================================
        # Main light - sunlight
        bpy.ops.object.light_add(type='SUN', location=(10, -10, 20))
        sun = bpy.context.active_object
        sun.data.energy = 2.0
        sun.rotation_euler = (math.radians(45), 0, math.radians(45))
        
        # Fill light - ambient light
        world_bg = bpy.context.scene.world.node_tree.nodes.get("Background")
        if world_bg:
            world_bg.inputs[1].default_value = 0.3
        
    def create_geometry(self, obj_config):
        """Create geometric objects - fix shadow issues"""
        shape = obj_config['shape']
        position = obj_config['position']
        rotation = obj_config['rotation']
        dimensions = obj_config['dimensions']
        color = obj_config.get('color')
        
        # Create geometric objects
        if shape == 'cube':
            bpy.ops.mesh.primitive_cube_add(size=2)
        elif shape == 'cuboid':
            bpy.ops.mesh.primitive_cube_add(size=2)
        elif shape == 'cylinder':
            bpy.ops.mesh.primitive_cylinder_add(radius=1, depth=2)
        elif shape == 'cone':
            bpy.ops.mesh.primitive_cone_add(radius1=1, depth=2)
        elif shape == 'square_pyramid':
            bpy.ops.mesh.primitive_cone_add(radius1=1, depth=2, vertices=4)
        elif shape == 'sphere':
            bpy.ops.mesh.primitive_uv_sphere_add(radius=1, segments=32, ring_count=16)
        
        obj = bpy.context.active_object
        
        # Convert from scene_config's coordinate system (x, y, z) to Blender coordinate system
        # scene_config: x=left/right, y=height, z=front/back
        # Blender: X=left/right, Y=front/back, Z=height
        
        # Set size
        obj.scale = (dimensions[0], dimensions[2], dimensions[1])
        
        # ========================================
        # Correctly calculate actual height and Z offset
        # 
        # Default size of Blender primitives:
        # - cube: side length 2 (from -1 to +1)
        # - cylinder: depth=2, radius=1
        # - cone: depth=2, radius=1
        # - square_pyramid: depth=2, radius=1
        # - sphere: radius=1 (diameter 2)
        #
        # obj.scale is the scaling factor for the default size
        # Actual size = default size × scale
        # ========================================
        
        if shape in ['cube', 'cuboid']:
            # Cube, cuboid: default side length 2, origin at geometric center
            # Actual height = 2.0 × scale[2]
            actual_height = 2.0 * obj.scale[2]
            z_offset = actual_height / 2
            
        elif shape == 'cylinder':
            # Cylinder: default depth=2, origin at geometric center
            actual_height = 2.0 * obj.scale[2]
            z_offset = actual_height / 2
            
        elif shape in ['cone', 'square_pyramid']:
            # Cone/square pyramid: default depth=2, origin at the midpoint between the bottom and top faces
            # Bottom face at Z=-1, top face at Z=+1, origin at Z=0
            # To make the bottom face touch the ground (Z=0), the origin needs to be raised by depth/2
            actual_height = 2.0 * obj.scale[2]
            z_offset = actual_height / 2
            
        elif shape == 'sphere':
            # Sphere: default radius=1 (diameter 2), origin at the center
            # To make the bottom touch the ground, the center needs to be raised by radius
            diameter = 2.0 * obj.scale[2]
            z_offset = diameter / 2  # radius
        
        else:
            # Default case
            actual_height = 2.0 * obj.scale[2]
            z_offset = actual_height / 2
        
        # Set position: X, Y axes are directly mapped, Z axis is raised by half the object height on the scene_config height basis
        obj.location = (position[0], position[2], position[1] + z_offset)
        
        # Set rotation (around Z axis)
        obj.rotation_euler = (0, 0, math.radians(rotation))
        
        # Set material and color
        mat = bpy.data.materials.new(name=f"Material_{shape}_{id(obj)}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        
        # Safely get Principled BSDF node
        bsdf = nodes.get("Principled BSDF")
        if not bsdf:
            # If not exists, create one
            bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        
        # Color mapping
        color_map = {
            'black': (0.05, 0.05, 0.05, 1.0),      # Pure black too dark, brighten a bit
            'white': (0.98, 0.98, 0.98, 1.0),      # Close to pure white but slightly adjusted, with contrast to light gray blue background
            'red': (0.8, 0.1, 0.1, 1.0),
            'yellow': (0.9, 0.9, 0.1, 1.0),
            'blue': (0.1, 0.3, 0.8, 1.0),
            'green': (0.1, 0.7, 0.2, 1.0),
            'normal': (0.6, 0.6, 0.6, 1.0)
        }
        
        if color and color in color_map:
            bsdf.inputs['Base Color'].default_value = color_map[color]
        else:
            bsdf.inputs['Base Color'].default_value = color_map['normal']
        
        bsdf.inputs['Roughness'].default_value = 0.6
        bsdf.inputs['Metallic'].default_value = 0.2
        
        obj.data.materials.append(mat)
        
        return obj
    
    def calculate_scene_bounds(self, objects):
        """Calculate scene bounding box - used to automatically adjust camera
        """
        if not objects:
            return 20.0
        
        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')
        
        for obj in objects:
            loc = obj.location
            scale = obj.scale
            
            # Consider the actual size of the object (including scaling and rotation)
            # Use 1.5 times the maximum size as a safety margin (considering rotation)
            max_dim = max(scale[0], scale[1], scale[2]) * 1.5
            
            min_x = min(min_x, loc[0] - max_dim)
            max_x = max(max_x, loc[0] + max_dim)
            min_y = min(min_y, loc[1] - max_dim)
            max_y = max(max_y, loc[1] + max_dim)
            min_z = min(min_z, loc[2] - max_dim)
            max_z = max(max_z, loc[2] + max_dim)
        
        # Calculate scene range
        range_x = max_x - min_x
        range_y = max_y - min_y
        range_z = max_z - min_z
        max_range = max(range_x, range_y, range_z)
        
        # Return suitable camera distance (leave 70% margin, balance completeness and compactness)
        return max_range * 1.7
    
    def set_camera(self, view_type, scene_objects):
        """Set camera view - automatically adapt to scene size"""
        # Delete existing camera
        for obj in bpy.data.objects:
            if obj.type == 'CAMERA':
                bpy.data.objects.remove(obj)
        
        # Create camera
        bpy.ops.object.camera_add()
        camera = bpy.context.active_object
        bpy.context.scene.camera = camera
        
        distance = self.calculate_scene_bounds(scene_objects)
        
        # Ensure minimum distance
        distance = max(distance, 20.0)
        
        # Calculate the actual center point of the scene
        if scene_objects:
            avg_x = sum(obj.location[0] for obj in scene_objects) / len(scene_objects)
            avg_y = sum(obj.location[1] for obj in scene_objects) / len(scene_objects)
            avg_z = sum(obj.location[2] for obj in scene_objects) / len(scene_objects)
            scene_center = mathutils.Vector((avg_x, avg_y, avg_z))
        else:
            scene_center = mathutils.Vector((0, 0, 0))
        
        if view_type == '3d':
            # 3D view: from the front view, looking down at 45 degrees from above
            # Camera position relative to the scene center
            angle_rad = math.radians(45)
            camera.location = (
                scene_center[0],
                scene_center[1] - distance * math.cos(angle_rad),
                scene_center[2] + distance * math.sin(angle_rad)
            )
        elif view_type == 'top':
            # Top view (completely orthogonal)
            camera.location = (scene_center[0], scene_center[1], scene_center[2] + distance * 1.2)
        elif view_type == 'front':
            # Front view (completely orthogonal, looking from the negative Y direction)
            # Key fix: Z coordinate is the same as the scene center, ensuring a 2D orthogonal view
            camera.location = (scene_center[0], scene_center[1] - distance, scene_center[2])
        elif view_type == 'left':
            # Left view (completely orthogonal, looking from the negative X direction)
            # Key fix: Z coordinate is the same as the scene center, ensuring a 2D orthogonal view
            camera.location = (scene_center[0] - distance, scene_center[1], scene_center[2])
        
        # Camera looks at the actual scene center
        camera.location = mathutils.Vector(camera.location)
        direction = scene_center - camera.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        camera.rotation_euler = rot_quat.to_euler()
        
        # Adjust camera field of view (FOV) to ensure complete display
        # Use a wider lens (24mm) to ensure all objects are visible
        camera.data.lens = 24  # Focal length (default 50mm, smaller = wider angle)
    
    def render_scene(self, scene_config, output_dir):
        """Render scene"""
        import time
        start_time = time.time()
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Render 4 views
        views = {
            '3d': '3D_view.png',
            'top': 'top.png',
            'front': 'front.png',
            'left': 'left.png'
        }
        
        # Early check: if all view files exist, skip the entire scene
        all_exist = True
        missing_views = []
        for view_type, filename in views.items():
            output_path = os.path.join(output_dir, filename)
            if not os.path.exists(output_path):
                all_exist = False
                missing_views.append((view_type, filename))
        
        if all_exist:
            print(f"  ↺ Scene already completed, skipping all views")
            return
        
        # Clear previous objects (keep ground and light)
        objects_to_remove = []
        for obj in bpy.data.objects:
            if obj.name not in ['Ground', 'Sun'] and obj.type != 'LIGHT':
                objects_to_remove.append(obj)
        
        for obj in objects_to_remove:
            bpy.data.objects.remove(obj)
        
        # Create all objects
        scene_objects = []
        for obj_config in scene_config['objects']:
            obj = self.create_geometry(obj_config)
            scene_objects.append(obj)
        
        # Only render missing views
        for view_type, filename in missing_views:
            view_start = time.time()
            
            output_path = os.path.join(output_dir, filename)
            
            # Use dynamic camera settings
            self.set_camera(view_type, scene_objects)
            bpy.context.scene.render.filepath = output_path
            
            # Render
            bpy.ops.render.render(write_still=True)
            
            view_time = time.time() - view_start
            print(f"    ✓ {filename} ({view_time:.1f}s)")
        
        # Clean up created objects and materials, release memory
        # First collect material references (before deleting objects) to avoid reference failure
        created_materials = set()
        for obj in scene_objects:
            try:
                # Check if the object is valid and not deleted
                if obj and obj.name in bpy.data.objects:
                    if hasattr(obj, 'data') and obj.data and hasattr(obj.data, 'materials'):
                        for mat in obj.data.materials:
                            if mat and mat.name.startswith("Material_"):
                                created_materials.add(mat)
            except (ReferenceError, AttributeError):
                # Object has been deleted or invalid, skip
                continue
        
        # Delete objects (after deletion, the users count of materials will automatically decrease)
        for obj in scene_objects:
            try:
                if obj and obj.name in bpy.data.objects:
                    bpy.data.objects.remove(obj, do_unlink=True)
            except (ReferenceError, AttributeError, KeyError):
                # Object has been deleted or invalid, skip
                continue
        
        # Clean up materials (safe deletion, using try-except to protect)
        materials_to_remove = []
        for mat in created_materials:
            try:
                # Check if the material still exists and is not used
                if mat and mat.name in bpy.data.materials:
                    if mat.users == 0:  # Ensure no other objects are using
                        materials_to_remove.append(mat)
            except (ReferenceError, AttributeError):
                # Material has been deleted or invalid, skip
                continue
        
        # Batch delete materials
        for mat in materials_to_remove:
            try:
                bpy.data.materials.remove(mat)
            except (ReferenceError, KeyError):
                # Material has been deleted, skip
                pass
        
        # Force garbage collection
        import gc
        gc.collect()
        
        total_time = time.time() - start_time
        print(f"  Total time for scene: {total_time:.1f} seconds")

def batch_render(input_dir, max_scenes=None, start_scene=0):
    """Batch render scenes"""
    import time
    
    print("=" * 60)
    print("Blender batch renderer - optimized version")
    print("=" * 60)
    print(f"Blender version: {bpy.app.version_string}")
    print(f"Rendering engine: EEVEE (fast mode)")
    print(f"Resolution: 1280x720")
    print(f"Sampling: 16 samples")
    print(f"Input directory: {input_dir}")
    print("=" * 60)
    
    # Check input directory
    if not os.path.exists(input_dir):
        print(f"✗ Error: input directory does not exist: {input_dir}")
        return
    
    try:
        renderer = BlenderRenderer()
    except Exception as e:
        print(f"✗ Failed to initialize renderer: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Iterate over all configurations
    configs = ['conf_1', 'conf_2']
    
    total_rendered = 0
    total_start = time.time()
    
    for config in configs:
        config_dir = os.path.join(input_dir, config)
        if not os.path.exists(config_dir):
            continue
        
        scenes = sorted([d for d in os.listdir(config_dir) 
                        if os.path.isdir(os.path.join(config_dir, d))])
        
        # Apply start position and maximum number of scenes limit
        if start_scene > 0:
            scenes = scenes[start_scene:]
        if max_scenes:
            scenes = scenes[:max_scenes]
        
        print(f"\nProcessing {config}: {len(scenes)} scenes")
        print("-" * 60)
        
        for i, scene_id in enumerate(scenes, 1):
            scene_path = os.path.join(config_dir, scene_id)
            config_file = os.path.join(scene_path, 'scene_config.json')
            
            if not os.path.exists(config_file):
                print(f"  ⚠ Skip {scene_id}: missing config file")
                continue
            
            print(f"\n[{i}/{len(scenes)}] Rendering: {scene_id}")
            
            # Load configuration
            with open(config_file, 'r') as f:
                scene_config = json.load(f)
            
            # Render
            try:
                renderer.render_scene(scene_config, scene_path)
                total_rendered += 1
                
                # Display progress and estimated time
                elapsed = time.time() - total_start
                avg_time = elapsed / total_rendered
                remaining = (len(scenes) - i) * avg_time
                print(f"  Progress: {total_rendered} completed | "
                      f"Average: {avg_time:.1f} seconds/scene | "
                      f"Estimated remaining: {remaining/60:.1f} minutes")
            except Exception as e:
                print(f"  ✗ Rendering failed: {e}")
                import traceback
                traceback.print_exc()
    
    total_time = time.time() - total_start
    
    print("\n" + "=" * 60)
    print("Rendering completed!")
    print("=" * 60)
    print(f"Total scenes: {total_rendered}")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"Average speed: {total_time/total_rendered:.1f} seconds/scene")
    print("=" * 60)

if __name__ == "__main__" and IN_BLENDER:
    import argparse
    
    # Parse command line arguments (Blender will pass additional parameters)
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory (contains scene_config.json)')
    parser.add_argument('--max-scenes', type=int, default=None,
                       help='Maximum number of scenes to render')
    parser.add_argument('--start', type=int, default=0,
                       help='Start scene index (for distributed rendering)')
    
    args = parser.parse_args(argv)
    
    batch_render(args.input, args.max_scenes, args.start)

# blender --background --python blender_renderer.py -- --input ./test --max-scenes 10 --start 0