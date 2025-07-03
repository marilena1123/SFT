# inspect_datasets.py

from datasets import load_dataset
import os

cache_dir = "./hf_cache"

datasets_to_check = [
    "TIGER-Lab/MathInstruct",
    "Vezora/Tested-143k-Python-Alpaca",
    "CohereForAI/aya_dataset",
    "tatsu-lab/alpaca",
    "pkavumba/balanced-copa",
    "allenai/social_i_qa"
]

for dataset_name in datasets_to_check:
    print("=" * 80)
    print(f"Inspecting dataset: {dataset_name}")
    print("=" * 80)

    try:
        ds = load_dataset(dataset_name, split="train", cache_dir=cache_dir)
        print("Loaded successfully.")
    except Exception as e:
        print(f"ERROR loading {dataset_name}: {e}")
        continue

    print("Number of examples:", len(ds))

    print("First example:")
    print(ds[11])

    print("\nField names:", list(ds.features.keys()))
    print("\n\n")
