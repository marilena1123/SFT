# download_sft_datasets.py

from datasets import load_dataset

datasets = [
    "TIGER-Lab/MathInstruct",
    "Vezora/Tested-143k-Python-Alpaca",
    "CohereForAI/aya_dataset",
    "tatsu-lab/alpaca",
    "pkavumba/balanced-copa",
    "allenai/social_i_qa",
    "qiaojin/PubMedQA"
]

for d in datasets:
    print(f"Checking: {d}")
    try:
        if d == "qiaojin/PubMedQA":
            ds = load_dataset(
                d, 
                'pqa_labeled',
                cache_dir="./hf_cache"
            )
        else:
            ds = load_dataset(
                d,
                cache_dir="./hf_cache"
            )
        print(f">>> {d} already in cache.")
    except FileNotFoundError:
        print(f">>> {d} not found in cache. Downloading...")
        load_dataset(
            d,
            cache_dir="./hf_cache"
        )
