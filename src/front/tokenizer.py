"""Byte-level word BPE for user questions and DB-tagged sentences in one ID space."""

import re
from collections import Counter
from contextlib import closing
from functools import lru_cache
from pathlib import Path
import sqlite3


TAG_ATOMS = ("string", "numeric", "date", "AGG_VALUE", "#1", "#2", "#3")
TEXT_CHARS = " 0123456789.,?'-"
TAG_CHARS = " <>{}[].,*"
NAME = r"[A-Za-z_][A-Za-z_0-9]*"
FIELD = rf"{NAME}\.{NAME}, (?:string|numeric|date)"
TAG = (rf"(?:<{NAME}>|<{FIELD}>|\{{{FIELD}\}}|\{{\*\}}"
       rf"|\[(?:{FIELD}|AGG_VALUE, numeric), #[1-3]\])")
TAG_RE = re.compile(TAG)
WORD_RE = re.compile(r"[A-Za-z]+")
OUTPUT_RE = re.compile(rf"(?P<tag>{TAG})|(?P<word>[A-Za-z]+)|.", re.DOTALL)
QUESTION_RE = re.compile(r"(?P<word>[A-Za-z]+)|.", re.DOTALL)
LEXEME_RE = re.compile(rf"#[0-9]+|{NAME}|.", re.DOTALL)
SCHEMA_DB = Path(__file__).resolve().parents[2] / "data" / "v1" / "shop.db"


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


def schema_names(path=SCHEMA_DB):
    """Table and column names of the target database, in schema order and without duplicates."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{path}: database not found")
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        tables = [r[0] for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid")]
        columns = [r[1] for t in tables for r in db.execute(f'PRAGMA table_info("{t}")')]
    return list(dict.fromkeys(tables + columns))


class FrontTokenizer:
    pad_id, bos_id, eos_id, sep_id = range(256, 260)
    seed_base = 260

    def __init__(self, vocab_size, names, merges):
        names = list(names)
        if any(type(n) is not str or not re.fullmatch(NAME, n) for n in names) or len(set(names)) != len(names):
            raise ValueError("schema names must be unique SQL identifiers")
        # A column called e.g. "date" reuses the tag atom instead of taking a second ID.
        self.names = tuple(n for n in names if n not in TAG_ATOMS)
        self.seeds = TAG_ATOMS + self.names
        self.merge_base = self.seed_base + len(self.seeds)
        if type(vocab_size) is not int or vocab_size < self.merge_base + len(merges):
            raise ValueError("vocab_size must accommodate bytes, special IDs, tag seeds and merges")
        self.vocab_size = vocab_size
        self._seed_ids = {s: self.seed_base + i for i, s in enumerate(self.seeds)}
        self._tag_encode = {c: ord(c) for c in TAG_CHARS} | self._seed_ids
        self._pieces = {i: bytes([i]) for i in range(256)}
        self._pieces.update({i: s.encode("ascii") for s, i in self._seed_ids.items()})
        self.merges = []
        self._merge_ids = {}
        for i, entry in enumerate(merges, self.merge_base):
            pair = tuple(entry)
            if len(pair) != 2 or any(type(t) is not int or not (0 <= t < i) or 256 <= t < self.merge_base for t in pair):
                raise ValueError("invalid word BPE merge")
            if pair in self._merge_ids:
                raise ValueError("duplicate word BPE merge")
            self._merge_ids[pair] = i
            self.merges.append(pair)
            self._pieces[i] = self._pieces[pair[0]] + self._pieces[pair[1]]
        # Merges only ever join ASCII letters, so every merge is a legal output token.
        letters = {ord(c) for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"}
        chars = letters | {ord(c) for c in TEXT_CHARS + TAG_CHARS}
        self.output_ids = (self.eos_id, *sorted(chars), *range(self.seed_base, self.merge_base + len(self.merges)))
        self._output_set = frozenset(self.output_ids) - {self.eos_id}
        self._word_ids = lru_cache(maxsize=8192)(self._encode_word)

    # Aliases let the back-model dataset, decoder and generation code drive this tokenizer unchanged.
    @property
    def sql_ids(self):
        return self.output_ids

    def encode_sql(self, output):
        return self.encode_output(output)

    def decode_sql(self, ids):
        return self.decode_output(ids)

    @classmethod
    def fit(cls, texts, vocab_size=4096, names=None):
        """Learn merges from questions and the untagged words of target sentences."""
        base = cls(vocab_size, schema_names() if names is None else names, [])
        frequencies = Counter()
        for text in texts:
            frequencies.update(WORD_RE.findall(TAG_RE.sub(" ", text)))

        words = {tuple(w.encode("ascii")): count for w, count in frequencies.items()}
        merges = []
        for new_id in range(base.merge_base, vocab_size):
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
        return cls(vocab_size, base.names, merges)

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
        result = []
        for match in QUESTION_RE.finditer(question):
            if match.lastgroup == "word":
                result.extend(self._word_ids(match[0]))
            else:
                result.extend(match[0].encode("utf-8"))
        return result

    def encode_output(self, output):
        result = []
        for match in OUTPUT_RE.finditer(output):
            if match.lastgroup == "tag":
                result.extend(self._encode_tag(match[0]))
            elif match.lastgroup == "word":
                result.extend(self._word_ids(match[0]))
            elif match[0] in TEXT_CHARS:
                result.append(ord(match[0]))
            else:
                raise ValueError(f"unsupported output character {match[0]!r} at offset {match.start()}")
        return result

    def _encode_tag(self, tag):
        result = []
        for match in LEXEME_RE.finditer(tag):
            try:
                result.append(self._tag_encode[match[0]])
            except KeyError:
                raise ValueError(f"unknown schema name {match[0]!r} in tag {tag}") from None
        return result

    def decode_output(self, ids):
        invalid = next((i for i in ids if i not in self._output_set), None)
        if invalid is not None:
            raise ValueError(f"invalid output token ID {invalid}")
        return b"".join(self._pieces[i] for i in ids).decode("ascii")

    def to_dict(self):
        return {"version": 1, "vocab_size": self.vocab_size, "seeds": list(TAG_ATOMS),
                "names": list(self.names), "merges": [list(p) for p in self.merges],
                "output_ids": list(self.output_ids)}

    @classmethod
    def from_dict(cls, state):
        if state.get("version") != 1 or state.get("seeds") != list(TAG_ATOMS):
            raise ValueError("unsupported tokenizer format or fixed vocabulary")
        tok = cls(state["vocab_size"], state["names"], state["merges"])
        if state.get("output_ids") != list(tok.output_ids):
            raise ValueError("checkpoint output whitelist does not match this tokenizer")
        return tok
