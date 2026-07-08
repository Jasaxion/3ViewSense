#!/bin/bash

# Dataset Construction Plan: Total 1000 samples
# 222: 45 (All available unique solutions)
# 333: 319
# 444: 318
# 555: 318
#
# Split: 80% Train / 20% Eval (Total 200 Eval)
# Eval Distribution: 222=9, 333=64, 444=64, 555=63

OUTPUT_DIR="OUTPUT_1K"

echo "=================================================="
echo "Step 1: Generating Block Layouts"
echo "Output Directory: $OUTPUT_DIR"
echo "=================================================="

# 222: Max unique is 45. We generate all.
echo "[1/4] Generating layouts for 222..."
python Block_gen.py --R 2 --C 2 --H 2 --output-dir "$OUTPUT_DIR"

# 333: Max unique is ~9000+. We generate all, but will only render 319.
echo "[2/4] Generating layouts for 333..."
python Block_gen.py --R 3 --C 3 --H 3 --output-dir "$OUTPUT_DIR"

# 444 & 555: We need 318 each. Generating 500 to ensure we have enough valid unique ones.
echo "[3/4] Generating layouts for 444..."
python Block_gen_large.py --R 4 --C 4 --H 4 --num 500 --output-dir "$OUTPUT_DIR"

echo "[4/4] Generating layouts for 555..."
python Block_gen_large.py --R 5 --C 5 --H 5 --num 500 --output-dir "$OUTPUT_DIR"


echo ""
echo "=================================================="
echo "Step 2: Rendering Images"
echo "=================================================="

echo "[1/4] Rendering 45 cases for 222..."
python render_block.py --file "$OUTPUT_DIR/level222/block_solutions.json" --num-cases 45

echo "[2/4] Rendering 319 cases for 333..."
python render_block.py --file "$OUTPUT_DIR/level333/block_solutions.json" --num-cases 319

echo "[3/4] Rendering 318 cases for 444..."
python render_block.py --file "$OUTPUT_DIR/level444/block_solutions.json" --num-cases 318

echo "[4/4] Rendering 318 cases for 555..."
python render_block.py --file "$OUTPUT_DIR/level555/block_solutions.json" --num-cases 318


echo ""
echo "=================================================="
echo "Step 3: Building Final Dataset"
echo "Split Strategy: Eval per level [222:9, 333:64, 444:64, 555:63]"
echo "=================================================="

python build_cube_counting_dataset.py --eval-per-level "222:9,333:64,444:64,555:63" --input-dir "$OUTPUT_DIR"

echo ""
echo "Done! Dataset generated in data/eval/cube_counting and data/train/cube_counting."
