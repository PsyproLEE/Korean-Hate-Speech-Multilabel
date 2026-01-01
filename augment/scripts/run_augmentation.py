# augment/scripts/run_augmentation.py

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from augment.src.augmentation.lexicon import SEED_LEXICON, build_merged_seed_lexicon
from augment.src.augmentation.filters import load_and_clean_auto_seed
from augment.src.augmentation.augmenter import augment_text_with_seed


# -----------------------------
# Path settings
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "merged_dataset_v1.1.csv"
AUTO_SEED_PATH = BASE_DIR / "data" / "auto_seed.json"
MERGED_SEED_SAVE_PATH = BASE_DIR / "data" / "merged_seed_lexicon.json"

AUG_ONLY_SAVE_PATH = BASE_DIR / "data" / "merged_dataset_v1.1_obf_M3_only.csv"
AUG_FULL_SAVE_PATH = BASE_DIR / "data" / "merged_dataset_v1.1_obf_M3_full.csv"


# -----------------------------
# Core logic
# -----------------------------
def run_augmentation(
    data_path: Path,
    max_new_per_sentence: int = 3,
    apply_to_hate_only: bool = True,
):
    """
    오프라인 데이터셋 증강 스크립트

    - auto_seed.json 정제
    - 수동 SEED + auto seed merge
    - hate / offensive 문장만 증강
    - 결과 CSV 저장
    """

    # 1. auto seed 정제
    print(">> Loading & cleaning auto_seed...")
    auto_seed_clean = load_and_clean_auto_seed(AUTO_SEED_PATH)

    # 2. seed lexicon merge
    print(">> Merging manual seed lexicon with auto_seed...")
    merged_seed = build_merged_seed_lexicon(SEED_LEXICON, auto_seed_clean)

    with open(MERGED_SEED_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(merged_seed, f, ensure_ascii=False, indent=2)

    print(f"> merged seed lexicon saved to {MERGED_SEED_SAVE_PATH}")

    # 3. load dataset
    df = pd.read_csv(data_path)
    df["text"] = df["text"].astype(str)

    if apply_to_hate_only and "hate_label" in df.columns:
        target_df = df[df["hate_label"].isin(["hate", "offensive"])].copy()
    else:
        target_df = df.copy()

    print(f"> target samples for augmentation: {len(target_df)}")

    # 4. augmentation
    augmented_rows = []

    for _, row in tqdm(target_df.iterrows(), total=len(target_df), desc="Augmenting"):
        text = row["text"]
        new_texts = augment_text_with_seed(
            text,
            merged_seed,
            max_new=max_new_per_sentence,
        )

        for nt in new_texts:
            new_row = row.copy()
            new_row["text"] = nt
            augmented_rows.append(new_row)

    aug_df = pd.DataFrame(augmented_rows)

    # 5. save results
    aug_df.to_csv(AUG_ONLY_SAVE_PATH, index=False, encoding="utf-8-sig")
    print(f"> augmented-only dataset saved: {AUG_ONLY_SAVE_PATH} ({len(aug_df)})")

    full_df = pd.concat([df, aug_df], ignore_index=True)
    full_df.to_csv(AUG_FULL_SAVE_PATH, index=False, encoding="utf-8-sig")
    print(f"> full dataset saved: {AUG_FULL_SAVE_PATH} ({len(full_df)})")

    return aug_df, full_df


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    run_augmentation(
        data_path=DATA_PATH,
        max_new_per_sentence=3,
        apply_to_hate_only=True,
    )
