# 1. generate splits
python src/data/splits.py

# 2. generate bg crops (MSER-based, one-time)
python scripts/generate_bg_crops.py

# 3. train initial model (no bg crops yet if manifest missing, or with MSER crops)
python scripts/train.py --model custom

# 4. eval on train split
python scripts/eval_sequence.py --model custom --split train

# 5. mine FPs from train images → enriches train_manifest
python scripts/mine_false_positives.py --model custom --split train

# 6. retrain with enriched bg crops
python scripts/train.py --model custom

# 7. eval on test split (honest final metrics)
python scripts/eval_sequence.py --model custom --split test

# 8. optionally enrich test bg crops
python scripts/mine_false_positives.py --model custom --split test
