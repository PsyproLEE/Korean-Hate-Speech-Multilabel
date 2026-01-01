SEED_LEXICON = {
    # ---------------------- #
    # 1. 일반 욕설 (profanity)
    # ---------------------- #
    "시발": {
        "canonical": ["시발", "씨발", "씨발이", "시발이"],
        "jamo": ["ㅅㅂ", "ㅆㅂ"],
        "type": "profanity",
        "targets": [],
    },
    "씨팔": {
        "canonical": ["씨팔", "씨팔새끼"],
        "jamo": [],
        "type": "profanity",
        "targets": [],
    },
    "좆": {
        "canonical": ["좆", "좆같은", "좆같다", "좆같네", "좆같고", "좆같아서"],
        "jamo": [],
        "type": "profanity",
        "targets": [],
    },
    "새끼": {
        "canonical": ["새끼", "새끼들", "새끼들이", "새끼가"],
        "jamo": [],
        "type": "profanity",
        "targets": [],
    },
    "병신": {
        "canonical": ["병신", "병신들", "병신같은", "병신이고", "병신이라"],
        "jamo": ["ㅂㅅ", "ㅄ"],
        "type": "profanity",
        "targets": [],
    },
    "등신": {
        "canonical": ["등신"],
        "jamo": [],
        "type": "profanity",
        "targets": [],
    },
    "개새끼": {
        "canonical": ["개새끼", "개새끼들"],
        "jamo": [],
        "type": "profanity",
        "targets": [],
    },
    "씹새끼": {
        "canonical": ["씹새끼"],
        "jamo": [],
        "type": "profanity",
        "targets": [],
    },
    "개같은": {
        "canonical": ["개같은", "개같이"],
        "jamo": [],
        "type": "profanity",
        "targets": [],
    },
    "지랄": {
        "canonical": ["지랄하네", "지랄을"],
        "jamo": [],
        "type": "profanity",
        "targets": [],
    },

    # ---------------------- #
    # 2. 성별/젠더 슬러 (gender)
    # ---------------------- #
    "한남": {
        "canonical": ["한남", "한남들", "한남은", "한남이", "한남새끼", "한남충", "한남충들"],
        "jamo": [],
        "type": "slur_gender",
        "targets": ["gender"],
    },
    "한녀": {
        "canonical": ["한녀", "한녀들", "한녀는", "한녀년", "한녀년들"],
        "jamo": [],
        "type": "slur_gender",
        "targets": ["gender"],
    },
    "김치녀": {
        "canonical": ["김치녀", "김치녀들", "김치녀같은"],
        "jamo": [],
        "type": "slur_gender",
        "targets": ["gender"],
    },
    "맘충": {
        "canonical": ["맘충"],
        "jamo": [],
        "type": "slur_gender",
        "targets": ["gender"],
    },
    "돼지년": {
        "canonical": ["돼지년"],
        "jamo": [],
        "type": "slur_gender",
        "targets": ["gender"],
    },
    "년": {
        "canonical": [
            "년들", "그년", "걸레년", "씨발년", "씨발년들", "씨발년이",
            "병신년", "병신년들", "김치년"
        ],
        "jamo": [],
        "type": "slur_gender",
        "targets": ["gender"],
    },

    # ---------------------- #
    # 3. LGBT / 성소수자 슬러
    # ---------------------- #
    "게이": {
        "canonical": ["게이", "게이들", "게이새끼", "게이같은"],
        "jamo": [],
        "type": "slur_LGBT",
        "targets": ["LGBT"],
    },
    "동성애자": {
        "canonical": ["동성애자", "동성애자들", "동성애자들이", "동성애는"],
        "jamo": [],
        "type": "slur_LGBT",
        "targets": ["LGBT"],
    },
    "후장": {
        "canonical": ["후장", "후장에"],
        "jamo": [],
        "type": "slur_LGBT",
        "targets": ["LGBT"],
    },
    "똥꼬충": {
        "canonical": ["똥꼬충"],
        "jamo": [],
        "type": "slur_LGBT",
        "targets": ["LGBT"],
    },

    # ---------------------- #
    # 4. 연령 (age)
    # ---------------------- #
    "틀딱": {
        "canonical": ["틀딱", "틀딱들", "틀니충", "틀니"],
        "jamo": [],
        "type": "slur_age",
        "targets": ["age"],
    },
    "급식충": {
        "canonical": ["급식충"],
        "jamo": [],
        "type": "slur_age",
        "targets": ["age"],
    },

    # ---------------------- #
    # 5. 지역 (region)
    # ---------------------- #
    "전라도": {
        "canonical": ["전라도"],
        "jamo": [],
        "type": "slur_region",
        "targets": ["region"],
    },

    # ---------------------- #
    # 6. 인종/민족/국적 (race)
    # ---------------------- #
    "짱깨": {
        "canonical": ["짱깨", "짱개"],
        "jamo": [],
        "type": "slur_race",
        "targets": ["race"],
    },
    "홍어": {
        "canonical": ["홍어", "홍어새끼"],
        "jamo": [],
        "type": "slur_race",
        "targets": ["race"],
    },
    "조센징": {
        "canonical": ["조센징", "조센"],
        "jamo": [],
        "type": "slur_race",
        "targets": ["race"],
    },
    "조선족": {
        "canonical": ["조선족"],
        "jamo": [],
        "type": "slur_race",
        "targets": ["race"],
    },
    "동남아": {
        "canonical": ["동남아", "똥남아"],
        "jamo": [],
        "type": "slur_race",
        "targets": ["race"],
    },

    # ---------------------- #
    # 7. 종교 (religion)
    # ---------------------- #
    "개독": {
        "canonical": ["개독", "개독이", "개독은"],
        "jamo": [],
        "type": "slur_religion",
        "targets": ["religion"],
    },
    "개슬람": {
        "canonical": ["개슬람", "개슬람새끼들"],
        "jamo": [],
        "type": "slur_religion",
        "targets": ["religion", "race"],
    },

    # ---------------------- #
    # 8. 계층/기타 (socioeconomic)
    # ---------------------- #
    "개돼지": {
        "canonical": ["개돼지"],
        "jamo": [],
        "type": "slur_socioeconomic",
        "targets": ["socioeconomic"],
    },
    "상폐녀": {
        "canonical": ["상폐녀", "상폐녀들", "상폐"],
        "jamo": [],
        "type": "slur_socioeconomic",
        "targets": ["socioeconomic"],
    },

    # ---------------------- #
    # 9. 일반 모욕
    # ---------------------- #
    "정신병자": {
        "canonical": ["정신병자", "정신병자들"],
        "jamo": [],
        "type": "abuse_general",
        "targets": [],
    },
    "버러지": {
        "canonical": ["버러지"],
        "jamo": [],
        "type": "abuse_general",
        "targets": [],
    },
    "꼬라지": {
        "canonical": ["꼬라지"],
        "jamo": [],
        "type": "abuse_general",
        "targets": [],
    },
    "역겹다": {
        "canonical": ["역겹다"],
        "jamo": [],
        "type": "abuse_general",
        "targets": [],
    },
}

def build_merged_seed_lexicon(seed_lexicon, auto_seed_clean):
    merged = dict(seed_lexicon)  # 얕은 복사

    # 기존 canonical/jamo 전체 모아두기
    existing_forms = set()
    for _, info in seed_lexicon.items():
        for c in info.get("canonical", []):
            existing_forms.add(c)
        for j in info.get("jamo", []):
            existing_forms.add(j)

    added = 0
    for term in auto_seed_clean.keys():
        if term in merged or term in existing_forms:
            continue

        merged[term] = {
            "canonical": [term],
            "jamo": [],
            "type": "auto_profanity",
            "targets": [],
        }
        added += 1

    print(f"> merged seed lexicon size: base={len(seed_lexicon)}, added_auto={added}, total={len(merged)}")
    return merged