import json
from pathlib import Path

from augment.src.augmentation.lexicon import SEED_LEXICON
from augment.src.augmentation.augmenter import augment_text_with_seed

AUGMENT_DIR = Path(__file__).resolve().parents[2]
MERGED_SEED_PATH = AUGMENT_DIR / "data" / "merged_seed_lexicon.json"

_GLOBAL_SEED_LEXICON = None

def get_seed_lexicon():
    global _GLOBAL_SEED_LEXICON

    if _GLOBAL_SEED_LEXICON is not None:
        return _GLOBAL_SEED_LEXICON

    if MERGED_SEED_PATH.exists():
        with open(MERGED_SEED_PATH, "r", encoding="utf-8") as f:
            _GLOBAL_SEED_LEXICON = json.load(f)
        print(f"[augment] loaded merged seed lexicon from {MERGED_SEED_PATH}")
    else:
        _GLOBAL_SEED_LEXICON = SEED_LEXICON
        print("[augment] merged_seed_lexicon.json not found, using manual SEED_LEXICON only.")

    return _GLOBAL_SEED_LEXICON


def augment_text(text: str, max_new: int = 3):
    seed_lexicon = get_seed_lexicon()
    return augment_text_with_seed(text, seed_lexicon, max_new=max_new)