# python Block_gen.py --R 2 --C 2 --H 2 --output-dir OUTPUT_FULL_COLOR
# python Block_gen.py --R 3 --C 3 --H 3 --output-dir OUTPUT_FULL_COLOR
# python Block_gen_large.py --R 4 --C 4 --H 4 --num 10000 --output-dir OUTPUT_FULL_COLOR
# python Block_gen_large.py --R 5 --C 5 --H 5 --num 10000 --output-dir OUTPUT_FULL_COLOR


python render_block.py --file OUTPUT_FULL_COLOR/level222/block_solutions.json --num-cases 10000 --random-color
python render_block.py --file OUTPUT_FULL_COLOR/level333/block_solutions.json --num-cases 10000 --random-color
python render_block.py --file OUTPUT_FULL_COLOR/level444/block_solutions.json --num-cases 10000 --random-color
python render_block.py --file OUTPUT_FULL_COLOR/level555/block_solutions.json --num-cases 10000 --random-color


# python split_output_by_level.py \
#     --input-dir OUTPUT_FULL_COLOR \
#     --eval-per-level "222:15,333:245,444:120,555:120"

# python build_cube_counting_color_dataset.py --split eval --input-dir OUTPUT_FULL_COLOR_eval