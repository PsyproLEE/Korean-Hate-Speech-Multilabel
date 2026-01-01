import re
from augment.src.augmentation.obfuscation import random_obfuscate_token


def find_seed_matches(text: str, seed_lexicon: dict):
    matches = []
    for seed, info in seed_lexicon.items():
        for form in info.get("canonical", []):
            if not form:
                continue
            for m in re.finditer(re.escape(form), text):
                matches.append((seed, info, m.start(), m.end()))
    return matches


def augment_text_with_seed(text: str, seed_lexicon: dict, max_new: int = 3):
    matches = find_seed_matches(text, seed_lexicon)
    if not matches:
        return []

    new_texts = set()

    # 여러 번 돌리면서 랜덤 변형된 문장 최대 max_new개 얻기
    for _ in range(max_new * 3):
        s = text
        # 뒤에서부터 치환 → index shift 방지
        for _, info, start, end in sorted(matches, key=lambda x: x[2], reverse=True):
            ori = s[start:end]
            obf = random_obfuscate_token(ori, info)
            s = s[:start] + obf + s[end:]

        if s != text:
            new_texts.add(s)
        if len(new_texts) >= max_new:
            break

    return list(new_texts)
