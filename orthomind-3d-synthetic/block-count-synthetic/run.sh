# # #!/bin/bash

# Generate data
python Block_gen.py --R 2 --C 2 --H 2
python Block_gen.py --R 3 --C 3 --H 3
python Block_gen_large.py --R 4 --C 4 --H 4 --num 150
python Block_gen_large.py --R 5 --C 5 --H 5 --num 150

# Render data
python render_block.py --file OUTPUT_DIR/level222/block_solutions.json --num-cases 34
python render_block.py --file OUTPUT_DIR/level333/block_solutions.json --num-cases 166
python render_block.py --file OUTPUT_DIR/level444/block_solutions.json --num-cases 150
python render_block.py --file OUTPUT_DIR/level555/block_solutions.json --num-cases 150

python build_cube_counting_dataset.py






# Generate data
python Block_gen.py --R 2 --C 2 --H 2 --output-dir OUTPUT_COLOR
python Block_gen.py --R 3 --C 3 --H 3 --output-dir OUTPUT_COLOR
python Block_gen_large.py --R 4 --C 4 --H 4 --num 150 --output-dir OUTPUT_COLOR
python Block_gen_large.py --R 5 --C 5 --H 5 --num 150 --output-dir OUTPUT_COLOR

# Render data
python render_block.py --file OUTPUT_COLOR/level222/block_solutions.json --num-cases 34 --random-color
python render_block.py --file OUTPUT_COLOR/level333/block_solutions.json --num-cases 166 --random-color
python render_block.py --file OUTPUT_COLOR/level444/block_solutions.json --num-cases 150 --random-color
python render_block.py --file OUTPUT_COLOR/level555/block_solutions.json --num-cases 150 --random-color

python build_cube_counting_color_dataset.py