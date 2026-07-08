## Object Synthetic Data Pipeline

This directory contains scripts for generating synthetic 3D objects, rendering views, and producing verification data.

### Key scripts

- `geometry_generator.py`: Create object geometry and export metadata.
- `scene_generator.py`: Build scenes from generated geometry.
- `blender_renderer.py`: Render multi-view images from scenes.
- `geometry_test_collect.py`: Collect geometry test cases.
- `geometry_test_collect_with_caption.py`: Collect geometry tests with captions.
- `verify_query.py`: Generate or verify query-style prompts.
- `verify_description.py`: Generate or verify description-style prompts.

### Typical usage

1. Generate geometry with `geometry_generator.py`.
2. Create scenes using `scene_generator.py`.
3. Render images via `blender_renderer.py`.
4. Run verification scripts to build OrthoMind-3D Object Reasoning data.

