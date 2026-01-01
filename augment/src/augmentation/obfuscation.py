import random

SPECIAL_CHARS = ["#", "*", "^", "~", "!", "?"]
LAUGH = ["ㅋ", "ㅋㅋ", "ㅋㅋㅋ"]
NUM_MAP = {"이": "2", "일": "1", "팔": "8"}


def safe_insert_special(s: str) -> str:
    if len(s) <= 1:
        return s
    pos = random.randint(1, len(s) - 1)
    return s[:pos] + random.choice(SPECIAL_CHARS + LAUGH) + s[pos:]


def safe_spacing(s: str) -> str:
    if len(s) <= 2:
        return s
    pos = random.randint(1, len(s) - 1)
    return s[:pos] + " " + s[pos:]


def safe_elongate(s: str) -> str:
    if len(s) <= 1:
        return s
    idx = random.randint(0, len(s) - 1)
    return s[:idx + 1] + s[idx] * random.randint(1, 3) + s[idx + 1:]


def to_jamo_fake(s: str) -> str:
    # 간단한 예시만: 시발 → ㅅㅣㅂㅏㄹ류
    return s.replace("시발", "ㅅㅣㅂㅏㄹ").replace("씨발", "ㅆㅣㅂㅏㄹ")


def leet(s: str) -> str:
    for k, v in NUM_MAP.items():
        s = s.replace(k, v)
    return s


def random_obfuscate_token(token: str, seed_info: dict) -> str:
    if len(token) <= 1:
        return token

    rules = ["insert_special", "spacing", "elongate"]

    t = seed_info.get("type", "")
    if t == "profanity" or t.startswith("auto"):
        rules += ["to_jamo", "leet"]
    if t.startswith("slur"):
        rules += ["prefix_suffix"]

    n = random.randint(1, min(3, len(rules)))
    chosen = random.sample(rules, n)

    s = token
    for r in chosen:
        if r == "insert_special":
            s = safe_insert_special(s)
        elif r == "spacing":
            s = safe_spacing(s)
        elif r == "elongate":
            s = safe_elongate(s)
        elif r == "to_jamo":
            s = to_jamo_fake(s)
        elif r == "leet":
            s = leet(s)
        elif r == "prefix_suffix":
            if random.random() < 0.5:
                s = "개" + s
            else:
                s = s + "새끼"

    return s