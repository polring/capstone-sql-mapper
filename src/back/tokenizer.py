"""Separate question BPE and SQL lexer backed by one ID space."""

import re
from collections import Counter
from functools import lru_cache


KEYWORDS = "SELECT FROM WHERE AND OR IN BETWEEN LIKE ORDER BY ASC DESC LIMIT COUNT AVG SUM MIN MAX GROUP HAVING".split()
TAG_ATOMS = ["TABLE1", *(f"COL{i}" for i in range(1, 7)), "AGG_VALUE",
             "string", "numeric", "date", "#1", "#2", "#3"]
# Seed order is part of the checkpoint and GGUF contract, including tags that a
# particular training split never contains.
_FIELD_NAMES = tuple(f"TABLE1.COL{col}, {kind}" for col in range(1, 7)
                     for kind in ("string", "numeric", "date"))
ATOMIC_TAGS = ("<TABLE1>", *(f"<{field}>" for field in _FIELD_NAMES),
               *(f"{{{field}}}" for field in _FIELD_NAMES), "{*}",
               *(f"[{field}, #{slot}]" for field in _FIELD_NAMES for slot in range(1, 4)),
               *(f"[AGG_VALUE, numeric, #{slot}]" for slot in range(1, 4)))
SEEDS = tuple(TAG_ATOMS + KEYWORDS + ["<=", ">=", "!="]) + ATOMIC_TAGS
SQL_CHARS = " 0123456789<>=.,{}()[]*'%"
FIELDS = r"TABLE1\.COL[1-6], (?:string|numeric|date)"
TAG = (rf"(?:<TABLE1>|<{FIELDS}>|\{{{FIELDS}\}}|\{{\*\}}"
       rf"|\[(?:{FIELDS}|AGG_VALUE, numeric), #[1-3]\])")
TAG_RE = re.compile(TAG)
QUESTION_RE = re.compile(rf"(?P<tag>{TAG})|(?P<word>[A-Za-z]+)|.", re.DOTALL)
LEXEME_RE = re.compile(rf"{TAG}|#[0-9]+|[A-Za-z_][A-Za-z_0-9]*|<=|>=|!=|.", re.DOTALL)


def _merge(tokens, pair, new_id):
    result = []
    index = 0
    while index < len(tokens):
        if index + 1 < len(tokens) and (tokens[index], tokens[index + 1]) == pair:
            result.append(new_id)
            index += 2
        else:
            result.append(tokens[index])
            index += 1
    return tuple(result)


class BackTokenizer:
    pad_id, bos_id, eos_id, sep_id = range(256, 260)
    seeds = SEEDS
    seed_base = 260
    merge_base = seed_base + len(seeds)

    def __init__(self, vocab_size, merges):
        if type(vocab_size) is not int or vocab_size < self.merge_base + len(merges):
            raise ValueError("vocab_size must accommodate bytes, special IDs, SQL seeds and merges")
        self.vocab_size = vocab_size
        self.merges = []
        self._merge_ids = {}
        self._sql_encode = {c: ord(c) for c in SQL_CHARS}
        self._sql_encode.update({s: self.seed_base + i for i, s in enumerate(self.seeds)})
        self._sql_decode = {i: s for s, i in self._sql_encode.items()}
        self.sql_ids = (self.eos_id, *sorted(self._sql_decode))
        for i, entry in enumerate(merges, self.merge_base):
            pair = tuple(entry)
            if len(pair) != 2 or any(type(t) is not int or not (0 <= t < i) or 256 <= t < self.merge_base for t in pair):
                raise ValueError("invalid question BPE merge")
            if pair in self._merge_ids:
                raise ValueError("duplicate question BPE merge")
            self._merge_ids[pair] = i
            self.merges.append(pair)
        self._word_ids = lru_cache(maxsize=8192)(self._encode_word)

    @classmethod
    def fit(cls, questions, vocab_size=4096):
        if vocab_size < cls.merge_base:
            raise ValueError(f"vocab_size must be at least {cls.merge_base}")
        frequencies = Counter()
        for question in questions:
            if not question.isascii():
                raise ValueError("expected an ASCII generalized question")
            frequencies.update(re.findall(r"[A-Za-z]+", TAG_RE.sub(" ", question)))

        words = {tuple(w.encode("ascii")): count for w, count in frequencies.items()}
        merges = []
        for new_id in range(cls.merge_base, vocab_size):
            counts = Counter()
            for word, freq in words.items():
                for pair in zip(word, word[1:]):
                    counts[pair] += freq
            if not counts:
                break
            best = max(counts, key=lambda p: (counts[p], -p[0], -p[1]))
            if counts[best] < 2:
                break
            words = {_merge(w, best, new_id): freq for w, freq in words.items()}
            merges.append(best)
        return cls(vocab_size, merges)

    def _encode_word(self, word):
        tokens = tuple(word.encode("ascii"))
        while len(tokens) > 1:
            pairs = zip(tokens, tokens[1:])
            chosen = min((self._merge_ids[p] for p in pairs if p in self._merge_ids), default=None)
            if chosen is None:
                break
            tokens = _merge(tokens, self.merges[chosen - self.merge_base], chosen)
        return tokens

    def encode_question(self, question):
        if not question.isascii():
            raise ValueError("expected an ASCII generalized question")
        result = []
        for match in QUESTION_RE.finditer(question):
            if match.lastgroup == "tag":
                result.append(self._sql_encode[match[0]])
            elif match.lastgroup == "word":
                result.extend(self._word_ids(match[0]))
            else:
                result.append(ord(match[0]))
        return result

    def encode_sql(self, sql):
        result = []
        for match in LEXEME_RE.finditer(sql):
            try:
                result.append(self._sql_encode[match[0]])
            except KeyError:
                raise ValueError(f"unsupported SQL token {match[0]!r} at offset {match.start()}") from None
        return result

    def decode_sql(self, ids):
        try:
            return "".join(self._sql_decode[i] for i in ids)
        except KeyError as exc:
            raise ValueError(f"invalid SQL token ID {exc.args[0]}") from None

    def to_dict(self):
        return {"vocab_size": self.vocab_size, "seeds": list(self.seeds),
                "merges": [list(p) for p in self.merges], "sql_ids": list(self.sql_ids)}

    @classmethod
    def from_dict(cls, state):
        if state.get("seeds") != list(cls.seeds):
            raise ValueError("incompatible tokenizer: an atomic-tag checkpoint is required; "
                             "retrain the back model to use whole-tag IDs")
        tok = cls(state["vocab_size"], state["merges"])
        if state.get("sql_ids") != list(tok.sql_ids):
            raise ValueError("checkpoint SQL whitelist does not match this tokenizer")
        return tok
