"""팀원 4 · 사용자 질문과 이에 대응하는 앞모델용 정답 생성

팀원 3이 만든 DB 종속 태그 문장의 문장 틀을 유지하고, 태그를 `mappings.json`의
자연어 표현과 실제 조건값으로 바꿔 사용자 질문을 만든다.

출력 레코드
    id      팀원 3 레코드 id 뒤에 `.n`을 붙인 계층형 id
    input   사용자 질문
    output  DB 종속 태그 문장 (팀원 3의 `input`과 같다)
    values  값 태그마다 질문에 쓴 원문(`text`)과 DB 저장값(`db_value`)

실행
    python src/data_generation/natural_questions/generate.py \\
        --input <팀원 3 출력 JSON> --output <저장할 JSON>
"""

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

MAPPINGS_PATH = Path(__file__).resolve().parent / "mappings.json"
# 팀원 2의 LIKE 문구(predicates.prefix/suffix/contains)를 읽어 검색 방향을 판단한다.
GENERALIZED_MAPPINGS_PATH = Path(__file__).resolve().parents[1] / "generalized_pairs" / "mappings.json"

TAG_RE = re.compile(
    r"(?P<value>\[(?P<value_key>AGG_VALUE|\w+\.\w+), (?P<value_type>string|numeric|date), #(?P<value_n>[1-3])\])"
    r"|(?P<select>\{(?P<select_key>\w+\.\w+), (?P<select_type>string|numeric|date)\})"
    r"|(?P<star>\{\*\})"
    r"|(?P<column><(?P<column_key>\w+\.\w+), (?P<column_type>string|numeric|date)>)"
    r"|(?P<table><(?P<table_key>\w+)>)"
)
TAG_CHARS_RE = re.compile(r"[{}\[\]<>]")


# 태그 바로 앞 단어로 단수·복수를 정한다.
SINGULAR_BEFORE = {"each", "every", "a", "an", "one", "single", "per", "this", "that"}
PLURAL_BEFORE = {"all", "many", "these", "those", "several", "both", "multiple"}
AGGREGATE_BEFORE = {"average", "mean", "total", "sum", "maximum", "minimum", "max", "min",
                    "highest", "lowest", "largest", "smallest"}
SINGULAR_AFTER = {"is", "was", "has"}
PLURAL_AFTER = {"are", "were", "have"}

# LIKE 검색 방향 단서 → mappings.json의 like_fragments 키
LIKE_BEFORE_RES = {
    "starts": re.compile(r"\b(?:starts?|starting|begins?|beginning)\s+with\s*$", re.IGNORECASE),
    "ends": re.compile(r"\b(?:ends?|ending)\s+with\s*$", re.IGNORECASE),
    "contains": re.compile(r"\b(?:contains?|containing|includes?|including|like)\s*$", re.IGNORECASE),
}
# 팀원 2 predicates 키 → like_fragments 키
LIKE_PREDICATE_KEYS = {"prefix": "starts", "suffix": "ends", "contains": "contains"}
LIKE_FIELD_RE = re.compile(r"\{(col|a)\}")


def load_like_forms(path=GENERALIZED_MAPPINGS_PATH):
    """팀원 2 mappings.json의 LIKE 문구를 {검색 방향: [문구]}로 읽는다. 없으면 빈 dict."""
    try:
        predicates = json.loads(Path(path).read_text(encoding="utf-8")).get("predicates", {})
    except (OSError, ValueError):
        return {}
    return {mode: predicates[key] for key, mode in LIKE_PREDICATE_KEYS.items() if predicates.get(key)}


LIKE_FORMS = load_like_forms()


def _like_form_re(form, tag):
    """문구의 {col}·{a} 자리에 이 값 태그와 같은 컬럼의 태그를 넣은 정규식."""
    pieces = {"col": re.escape(f"<{tag.key}, string>"), "a": re.escape(tag.mark)}
    # split 결과는 문구 조각과 필드 이름이 번갈아 나온다. 문구 조각의 공백은 \s+로 바꾼다.
    parts = LIKE_FIELD_RE.split(form)
    pattern = "".join(pieces[p] if i % 2 else r"\s+".join(map(re.escape, p.split(" ")))
                      for i, p in enumerate(parts))
    return re.compile(pattern, re.IGNORECASE)


def like_mode(sentence, tag, like_forms):
    """값 태그가 LIKE 검색어 자리면 starts/ends/contains, 아니면 None."""
    if TAG_RE.fullmatch(tag.mark).group("value_type") != "string":
        return None
    for mode, forms in like_forms.items():
        if any(_like_form_re(form, tag).search(sentence) for form in forms):
            return mode
    before = sentence[:tag.start]
    return next((mode for mode, r in LIKE_BEFORE_RES.items() if r.search(before)), None)

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]


@dataclass(frozen=True)
class Tag:
    kind: str  # table, column, select, star, value
    mark: str
    key: str | None
    start: int
    end: int


def parse_tags(sentence):
    """DB 종속 태그 문장에서 태그를 등장 순서대로 찾는다. 알 수 없는 태그가 남으면 ValueError."""
    tags = []
    for m in TAG_RE.finditer(sentence):
        kind = next(k for k in ("value", "select", "star", "column", "table") if m.group(k))
        key = None if kind == "star" else m.group(f"{kind}_key")
        tags.append(Tag(kind, m.group(0), key, m.start(), m.end()))

    leftover = TAG_RE.sub("", sentence)
    if TAG_CHARS_RE.search(leftover):
        raise ValueError(f"알 수 없는 태그가 있습니다: {sentence}")
    return tags


def _last_word(text):
    words = re.findall(r"[A-Za-z']+", text)
    return words[-1].lower() if words else ""


def _first_word(text):
    m = re.match(r"\s*([A-Za-z']+)", text)
    return m.group(1).lower() if m else ""


def grammatical_number(tag, before, after):
    """태그 앞뒤 문맥으로 단수(singular)·복수(plural)를 고른다."""
    prev = _last_word(before)
    if prev in SINGULAR_BEFORE or prev in AGGREGATE_BEFORE:
        return "singular"
    if prev in PLURAL_BEFORE:
        return "plural"

    nxt = _first_word(after)
    if nxt in SINGULAR_AFTER:
        return "singular"
    if nxt in PLURAL_AFTER:
        return "plural"

    if tag.kind == "select":
        # "the {...} of each <items>"는 단수, 그 밖의 조회 대상은 여러 행이므로 복수
        m = re.match(r"\s+of\s+(\w+)", after)
        if m and m.group(1).lower() in SINGULAR_BEFORE:
            return "singular"
        return "plural"
    if tag.kind == "table":
        return "plural"
    return "singular"


def format_number(n, spec):
    if isinstance(n, float) and n.is_integer():
        n = int(n)
    if isinstance(n, float):
        return f"{n:.{spec.get('decimals', 2)}f}"
    return str(n)


def format_date(iso, template):
    year, month, day = (int(p) for p in iso.split("-"))
    return template.format(
        year=year, month=MONTHS[month - 1], mon=MONTHS[month - 1][:3],
        day=day, mm=f"{month:02d}", dd=f"{day:02d}",
    )


def pick_value(tag, spec, like, rng, used):
    """값 태그 하나에 넣을 (질문 원문, DB 저장값)을 고른다. 같은 컬럼의 #1~#3은 서로 다른 값을 쓴다.

    like는 LIKE 검색 방향(starts/ends/contains)이며, 일반 조건값이면 None이다.
    """
    value_type = TAG_RE.fullmatch(tag.mark).group("value_type")
    taken = used.setdefault(tag.key, set())
    if like and like not in spec.get("like_fragments", {}):
        raise ValueError(f"{tag.key}에 '{like}' 검색어 후보가 없습니다")

    for _ in range(50):
        if value_type == "string":
            if like:
                db_value = rng.choice(spec["like_fragments"][like])
                text = db_value
            else:
                db_value = rng.choice(spec["candidates"])
                text = rng.choice([db_value] + spec.get("aliases", {}).get(db_value, []))
        elif value_type == "numeric":
            if "candidates" in spec:
                db_value = rng.choice(spec["candidates"])
            else:
                db_value = rng.randint(*spec["range"])
            if spec.get("integer"):
                db_value = int(db_value)
            number = format_number(db_value, spec)
            unit = spec.get("unit")
            styles = ["{n}"]
            if unit:
                styles.append("{n} {unit}")
                if unit.get("prefix"):
                    styles.append("{prefix}{n}")
            text = rng.choice(styles).format(
                n=number,
                unit=unit and (unit["singular"] if db_value == 1 else unit["plural"]),
                prefix=unit and unit.get("prefix"),
            )
        else:
            db_value = rng.choice(spec["candidates"])
            text = format_date(db_value, rng.choice(spec["formats"]))

        if db_value not in taken:
            taken.add(db_value)
            return text, db_value
    raise ValueError(f"{tag.mark}에 넣을 서로 다른 값 후보가 부족합니다")


def _fix_article(prefix, word):
    """치환한 단어 앞의 a/an을 맞춘다."""
    m = re.search(r"\b([Aa])n? $", prefix)
    if not m:
        return prefix
    lower = word.lower()
    vowel = lower[:1] in "aeiou" and not lower.startswith(("us", "uni", "eu", "one"))
    article = m.group(1) + ("n" if vowel else "")
    return prefix[: m.start()] + article + " "


def realize(sentence, tags, mappings, rng, like_forms=None):
    """태그 문장 하나를 사용자 질문 하나로 바꾼다. (질문, values)를 돌려준다."""
    like_forms = LIKE_FORMS if like_forms is None else like_forms
    parts = []
    values = {}
    used = {}
    chosen = {}  # 같은 표시가 반복되면 같은 표현을 쓴다
    cursor = 0

    for tag in tags:
        before = sentence[:tag.start]
        after = sentence[tag.end:]
        prefix = sentence[cursor:tag.start]

        if tag.kind == "value":
            if tag.mark not in values:
                spec = mappings["agg_value"] if tag.key == "AGG_VALUE" else mappings["values"][tag.key]
                like = None if tag.key == "AGG_VALUE" else like_mode(sentence, tag, like_forms)
                text, db_value = pick_value(tag, spec, like, rng, used)
                values[tag.mark] = {"text": text, "db_value": db_value}
            word = values[tag.mark]["text"]
        else:
            number = grammatical_number(tag, before, after)
            cache_key = (tag.mark, number)
            if cache_key not in chosen:
                if tag.kind == "star":
                    candidates = mappings["star"]
                elif tag.kind == "table":
                    candidates = mappings["tables"][tag.key][number]
                else:
                    candidates = mappings["columns"][tag.key][number]
                chosen[cache_key] = rng.choice(candidates)
            word = chosen[cache_key]

        parts.append(_fix_article(prefix, word))
        if not "".join(parts).strip() or re.search(r"[.?!]\s*$", "".join(parts)):
            word = word[:1].upper() + word[1:]
        parts.append(word)
        cursor = tag.end

    parts.append(sentence[cursor:])
    question = "".join(parts)
    if TAG_CHARS_RE.search(question):
        raise ValueError(f"치환 후 태그 문자가 남았습니다: {question}")
    return question, values


def generate_for_record(record, mappings, seed, max_per_parent):
    """팀원 3 레코드 하나에서 서로 다른 사용자 질문을 최대 max_per_parent개 만든다."""
    sentence = record["input"]
    tags = parse_tags(sentence)
    rng = random.Random(f"{seed}:{record['id']}")

    results = []
    seen = set()
    for _ in range(max_per_parent * 20):
        question, values = realize(sentence, tags, mappings, rng)
        if question in seen:
            continue
        seen.add(question)
        results.append({
            "id": f"{record['id']}.{len(results) + 1}",
            "input": question,
            "output": sentence,
            "values": values,
        })
        if len(results) == max_per_parent:
            break
    return results


def generate(records, mappings, seed=None, max_per_parent=None):
    settings = mappings.get("generation", {})
    seed = settings.get("seed", 0) if seed is None else seed
    max_per_parent = settings.get("max_per_parent", 3) if max_per_parent is None else max_per_parent

    examples = []
    for record in records:
        examples.extend(generate_for_record(record, mappings, seed, max_per_parent))
    return examples


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path, help="팀원 3 schema_mapping 출력 JSON")
    parser.add_argument("--output", required=True, type=Path, help="저장할 JSON 경로")
    parser.add_argument("--mappings", default=MAPPINGS_PATH, type=Path)
    parser.add_argument("--seed", type=int, help="mappings.json의 generation.seed 대신 사용")
    parser.add_argument("--max-per-parent", type=int, help="mappings.json의 generation.max_per_parent 대신 사용")
    args = parser.parse_args(argv)

    records = json.loads(args.input.read_text(encoding="utf-8"))
    mappings = json.loads(args.mappings.read_text(encoding="utf-8"))
    examples = generate(records, mappings, args.seed, args.max_per_parent)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(records)}건 → {len(examples)}건: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
