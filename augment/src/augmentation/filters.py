import re
import json


MIN_FREQ = 3  # auto_seed에서 최소 등장 횟수
YEAR_REGEX = re.compile(r"\d{4}년|\b(19[0-9]{2}|20[0-2][0-9])\b")

SAFE_WORDS = {
    "개념", "개인", "개발", "개성", "개방", "개인적", "개최", "개편", "개정",
    "개학", "개강", "개선", "개요"
}

BAN_PREFIX = ["개", "씨", "좆", "병신", "미친", "씹", "년", "새끼", "후장", "틀딱", "맘충", "급식충"]
SAFE_AFTER_GAE = {"념", "인", "발", "성", "방", "선", "편", "정", "학", "선", "정"}


def looks_like_insult(token: str) -> bool:
    t = token.strip()

    if len(t) < 2:
        return False
    if t in SAFE_WORDS:
        return False
    if YEAR_REGEX.search(t):
        return False
    if t.isdigit():
        return False

    # 1) prefix 기반
    for p in BAN_PREFIX:
        if t.startswith(p):
            # '개'로 시작하지만 '개념', '개인' 같은 정상 단어는 제외
            if p == "개" and len(t) >= 2 and t[1] in SAFE_AFTER_GAE:
                return False
            return True

    # 2) 욕 패턴 fragment 기반
    FRAGMENTS = [
        "ㅅㅂ", "ㅄ", "ㅂㅅ", "좆", "병신", "씨발", "시발",
        "충", "년", "새끼", "게이", "동성애", "홍어", "짱깨",
        "조센징", "조선족", "동남아", "버러지", "정신병", "역겹", "꼬라지"
    ]
    if any(frag in t for frag in FRAGMENTS):
        # 숫자+년 형태는 제외
        if re.match(r"\d{4}년", t):
            return False
        return True

    return False


def load_and_clean_auto_seed(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    clean = {}
    for token, freq in raw.items():
        if freq < MIN_FREQ:
            continue
        if looks_like_insult(token):
            clean[token] = freq

    print(f"> auto seed cleaned: {len(clean)} tokens kept")
    return clean
