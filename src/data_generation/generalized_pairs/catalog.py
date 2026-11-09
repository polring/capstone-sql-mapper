"""Hand-authored SQLite shapes and their aligned English question frames."""

from collections import deque
from dataclasses import dataclass
from itertools import chain, product
from math import prod
import random
import re


# Lowercase placeholders are template fields. Uppercase SQL tags and {*} are data.
FIELD = re.compile(r"\{([a-z][a-z0-9_]*)\}")
WORD = re.compile(r"<[^>]+>|\{[^}]+\}|\[[^\]]+\]|[a-z0-9]+(?:'[a-z]+)?")
TYPES = ("numeric", "string", "date")
COMPARISONS = ("=", "!=", "<", ">", "<=", ">=")
DIRECTIONS = (("ASC", "ascending"), ("DESC", "descending"))
AGGREGATES = (
    ("COUNT", None), ("AVG", "numeric"), ("SUM", "numeric"),
    ("MIN", "numeric"), ("MAX", "numeric"), ("MIN", "date"), ("MAX", "date"),
)
THRESHOLD = "[AGG_VALUE, numeric, #1]"
DEFAULT_QUESTIONS_PER_SQL = 3


@dataclass(frozen=True)
class Phrase:
    sql: str
    words: tuple[str, ...]
    bindings: tuple[tuple[str, "Phrase"], ...] = ()

    def fill(self, **parts):
        # Resolve SQL now. Keep nested wording choices until a source SQL has a seed.
        return Phrase(render_sql(self.sql, parts), self.words, self.bindings + tuple(parts.items()))

    def with_more_forms(self, words):
        return Phrase(self.sql, self.words + words, self.bindings)

    def frame_streams(self, parts, rng):
        scope = {**parts, **dict(self.bindings)}
        frames = list(dict.fromkeys(self.words))
        rng.shuffle(frames)
        return [expand_frame(frame, scope, rng) for frame in frames]

    def distinct_questions(self, parts, limit, rng, *, concise_first=False):
        seen = set()
        if concise_first:
            frames = list(dict.fromkeys(self.words))
            rng.shuffle(frames)
            scope = {**parts, **dict(self.bindings)}
            first, short = shortest_wording(frames, scope, rng)
            streams = [chain((short,), expand_frame(first, scope, rng))]
            streams.extend(expand_frame(f, scope, rng) for f in frames if f != first)
        else:
            streams = self.frame_streams(parts, rng)
        # Use each frame once before taking another wording from the same frame.
        for text in round_robin(streams):
            if text in seen:
                continue
            seen.add(text)
            yield text
            if len(seen) == limit:
                return


def shortest_wording(frames, parts, rng):
    """Render a concise variant while retaining all semantic fields of its frame."""
    choices = {}

    def substitute(match):
        name = match[1]
        if name not in choices:
            part = parts[name]
            if len(part.words) == 1 and not FIELD.search(part.words[0]):
                choices[name] = part.words[0]
            else:
                forms = list(part.words)
                rng.shuffle(forms)
                _, choices[name] = shortest_wording(forms, {**parts, **dict(part.bindings)}, rng)
        return choices[name]

    candidates = ((frame, FIELD.sub(substitute, frame)) for frame in frames)
    return min(candidates, key=lambda item: len(WORD.findall(item[1])))


def expand_frame(frame, parts, rng):
    """Lazily expand fields. Repeated mentions of one field share its wording."""
    fields = tuple(dict.fromkeys(FIELD.findall(frame)))
    if not fields:
        yield frame
        return
    words = {}

    def expand(index):
        if index == len(fields):
            yield FIELD.sub(lambda match: words[match[1]], frame)
            return
        name = fields[index]
        for stream in parts[name].frame_streams(parts, rng):
            for text in stream:
                words[name] = text
                yield from expand(index + 1)

    yield from expand(0)


def round_robin(iterators):
    active = deque(iterators)
    while active:
        current = active.popleft()
        item = next(current, None)
        if item is not None:
            yield item
            active.append(current)


def render_sql(template, parts):
    def replace(match):
        part = parts.get(match[1])
        if part is None:
            return match[0]  # Nested frames can still contain unbound fields.
        return part.sql

    return FIELD.sub(replace, template)


def literal(sql, words=None):
    text = sql if words is None else words
    return Phrase(sql, (text,))


EMPTY = literal("")


@dataclass(frozen=True)
class SortOrder(Phrase):
    ranked: Phrase = EMPTY


@dataclass(frozen=True)
class SortKey(Phrase):
    is_date: bool = False


@dataclass(frozen=True)
class Column:
    number: int
    kind: str

    @property
    def label(self):
        return f"TABLE1.COL{self.number}, {self.kind}"

    @property
    def ref(self):
        return f"<{self.label}>"

    @property
    def target(self):
        return "{" + self.label + "}"

    @property
    def sort_key(self):
        return SortKey(self.ref, (self.ref,), is_date=self.kind == "date")

    def value(self, number):
        tag = f"[{self.label}, #{number}]"
        return literal(tag if self.kind == "numeric" else f"'{tag}'", tag)


class Block:
    """Each mixed-radix index selects one distinct SQL in this query shape."""

    def __init__(self, frame, slots, shape, average_words=()):
        self.shape = shape  # outer filter count, inner filter count, ORDER BY, LIMIT
        self.frame = frame
        self.average_words = average_words
        self.choices = list(slots.items())
        self.size = prod(len(options) for _, options in self.choices)

    def source(self, index, max_questions_per_sql=DEFAULT_QUESTIONS_PER_SQL, *, seed=0):
        selected = {}
        for name, options in self.choices:
            index, digit = divmod(index, len(options))
            selected[name] = options[digit]
        sql = render_sql(self.frame.sql, selected)
        frame = self.frame
        if self.average_words and selected["stat"].sql == "AVG":
            frame = frame.with_more_forms(self.average_words)
        if self.shape[2] and self.shape[3]:
            order = selected["order"]
            cap = selected["limit"]
            selected["order"] = order.ranked.fill(explicit_order=order, explicit_limit=cap,
                                                **dict(cap.bindings))
            selected["limit"] = EMPTY
        wording_rng = random.Random(f"{seed}:{sql}")
        return sql, frame.distinct_questions(selected, max_questions_per_sql, wording_rng, concise_first=True)


def result_blocks(frame, slots, filter_count):
    """Partition row/group SQLs before choosing columns, directions or a limit value."""
    if "order" not in slots:
        yield Block(frame, slots, (filter_count, 0, False, False))
        return
    # Both lists start with EMPTY, followed by the concrete clause alternatives.
    for ordered, limited in product((False, True), repeat=2):
        options = dict(slots)
        options["order"] = slots["order"][1:] if ordered else [EMPTY]
        # Give everyday counts (1–10) as many turns as the wider range (11–100).
        ranges = (slots["limit"][1:11], slots["limit"][11:]) if limited else ([EMPTY],)
        for caps in ranges:
            if caps:
                yield Block(frame, {**options, "limit": caps}, (filter_count, 0, ordered, limited))


def mappable(columns, needs_measure=False):
    # Type capacities of items, orders and customers. Only items and orders
    # have columns eligible as numeric aggregate arguments in v1.
    profiles = [(3, 2, 0), (4, 1, 1)]
    if not needs_measure:
        profiles.append((1, 3, 0))
    counts = [sum(c.kind == t for c in columns) for t in TYPES]
    return any(all(n <= cap for n, cap in zip(counts, p)) for p in profiles)


def filter_layouts(existing, count, *, reuse=False, needs_measure=False):
    # Existing columns already have an ID. New columns follow mention order.
    prefixes = [()]
    if reuse and count:
        prefixes.extend((col,) for col in existing)
    for prefix in prefixes:
        for kinds in product(TYPES, repeat=count - len(prefix)):
            new = tuple(Column(i, t) for i, t in enumerate(kinds, len(existing) + 1))
            if mappable(existing + new, needs_measure):
                yield prefix + new, existing + new


class Catalog:
    def __init__(self, mappings):
        self.m = mappings
        self.filters = {}
        self.row_orders = {}
        self.limits = [EMPTY] + [
            self.phrase("clauses", "limit", " LIMIT {n}").fill(
                n=literal(str(n)), result_unit=literal("result" if n == 1 else "results")
            )
            for n in range(1, 101)
        ]

    def phrase(self, section, key, sql):
        words = self.m[section][key]
        if not isinstance(words, list) or not words or not all(isinstance(w, str) and w for w in words):
            raise ValueError(f"{section}.{key} must contain a nonempty list of question forms")
        return Phrase(sql, tuple(words))

    def predicates(self, col):
        if col in self.filters:
            return self.filters[col]
        a, b, c = (col.value(n) for n in (1, 2, 3))
        ref = literal(col.ref)
        section = "date_comparisons" if col.kind == "date" else "comparisons"
        options = []
        for op in COMPARISONS:
            if col.kind == "string" and op not in ("=", "!="):
                continue
            comparison = self.phrase(section, op, op)
            options.append(literal("{col} {op} {a}").fill(col=ref, op=comparison, a=a))
        for values in ((a, b), (a, b, c)):
            items = literal(", ".join(v.sql for v in values), ", ".join(v.words[0] for v in values))
            in_list = self.phrase("predicates", "in", "{col} IN ({values})")
            options.append(in_list.fill(col=ref, values=items))
        if col.kind == "string":
            for key, pattern in (("prefix", "{a}%"), ("suffix", "%{a}"), ("contains", "%{a}%")):
                like = self.phrase("predicates", key, "{col} LIKE '" + pattern + "'")
                options.append(like.fill(col=ref, a=literal(a.words[0])))
        else:
            between = self.phrase("predicates", "between", "{col} BETWEEN {a} AND {b}")
            options.append(between.fill(col=ref, a=a, b=b))
        self.filters[col] = options
        return options

    def conditions(self, columns, slots, prefix="p", allow_or=False):
        fields = []
        for i, col in enumerate(columns):
            name = f"{prefix}{i}"
            slots[name] = self.predicates(col)
            fields.append("{" + name + "}")
        if allow_or and len(fields) > 1:
            slots["join"] = [literal("AND", "and"), literal("OR", "or")]
            return literal(" {join} ".join(fields))
        return literal(" AND ".join(fields), " and ".join(fields))

    def order_options(self, primaries, seconds):
        options = [EMPTY]
        for primary in primaries:
            for direction, word in DIRECTIONS:
                order_key = "date_" + direction.lower() if primary.is_date else "order"
                rank_key = "ranked_" + order_key if primary.is_date else "ranked"
                order = self.phrase("clauses", order_key, " ORDER BY {key} {direction}").fill(
                    key=primary, direction=literal(direction, word)
                )
                ranked = self.phrase("intent", rank_key, "").fill(
                    key=primary, rank=literal("highest" if direction == "DESC" else "lowest"),
                    edge=literal("top" if direction == "DESC" else "bottom")
                )
                options.append(SortOrder(order.sql, order.words, order.bindings, ranked.fill(tie=EMPTY)))
                for secondary in seconds(primary):
                    for tie_dir, tie_word in DIRECTIONS:
                        tie_key = "date_tie_" + tie_dir.lower() if secondary.is_date else "tie"
                        tie = self.phrase("clauses", tie_key, ", {key} {direction}").fill(
                            key=secondary, direction=literal(tie_dir, tie_word)
                        )
                        both = literal("{order}{tie}").fill(order=order, tie=tie)
                        options.append(SortOrder(both.sql, both.words, both.bindings, ranked.fill(tie=tie)))
        return options

    def row_sorting(self, columns):
        if columns not in self.row_orders:
            def available(mentioned):
                new = (Column(len(mentioned) + 1, t) for t in TYPES)
                return mentioned + tuple(c for c in new if mappable(mentioned + (c,)))

            primaries = {c.sort_key: c for c in available(columns)}

            def seconds(primary):
                col = primaries[primary]
                mentioned = columns if col in columns else columns + (col,)
                return [c.sort_key for c in available(mentioned) if c != col]

            self.row_orders[columns] = self.order_options(primaries, seconds)
        return self.row_orders[columns]

    def where(self, columns, slots, *, group=False, allow_or=False):
        if not columns:
            return EMPTY
        filters = self.conditions(columns, slots, allow_or=allow_or)
        key = "group_where" if group else "where"
        return self.phrase("clauses", key, " WHERE {filters}").fill(filters=filters)

    def rows(self):
        for kind in (None, *TYPES):
            base = (Column(1, kind),) if kind else ()
            target = literal(base[0].target if base else "{*}")
            for count in range(4):
                for filters, columns in filter_layouts(base, count, reuse=True):
                    slots = {"order": self.row_sorting(columns), "limit": self.limits}
                    where = self.where(filters, slots, allow_or=True)
                    key = "rows" if base else "all_rows"
                    frame = self.phrase("questions", key, "SELECT {target} FROM <TABLE1>{where}{order}{limit}")
                    yield from result_blocks(frame.fill(target=target, where=where), slots, count)

    def group_frame(self, key, agg, has_numeric_result):
        group_col = key.sort_key
        result_words = "row count" if agg.sql == "COUNT(*)" else "the aggregate result"
        result = SortKey(agg.sql, (result_words,), is_date=not has_numeric_result)
        order = self.order_options(
            [result, group_col], lambda primary: [group_col] if primary == result else []
        )
        having = [EMPTY]
        if has_numeric_result:
            clause = self.phrase("clauses", "having", " HAVING {agg} {op} {value}")
            for op in COMPARISONS:
                having.append(clause.fill(
                    agg=agg, op=self.phrase("comparisons", op, op), value=literal(THRESHOLD)
                ))
        frame = self.phrase(
            "questions", "group", "SELECT {key}, {{agg}} FROM <TABLE1>{where}"
            " GROUP BY {group_col}{having}{order}{limit}"
        ).fill(key=literal(key.target), group_col=group_col, agg=agg)
        return frame, {"order": order, "having": having, "limit": self.limits}

    def aggregates(self, grouped):
        for group_type in (TYPES if grouped else (None,)):
            base = (Column(1, group_type),) if grouped else ()
            for fn, kind in AGGREGATES:
                columns = base
                if kind:
                    arg = Column(len(base) + 1, kind)
                    columns += (arg,)
                    key = fn + "_date" if kind == "date" else fn
                    agg = self.phrase("aggregates", key, fn + "({col})")
                    agg = agg.fill(col=literal(arg.ref, arg.target))
                else:
                    agg = self.phrase("aggregates", fn, "COUNT(*)")
                if grouped:
                    frame, group_slots = self.group_frame(base[0], agg, kind != "date")
                    if fn == "COUNT":
                        count = self.phrase("intent", "group_count", frame.sql)
                        frame = frame.with_more_forms(count.words)
                else:
                    frame = self.phrase("questions", "aggregate", "SELECT {{agg}} FROM <TABLE1>{where}")
                    if fn == "COUNT":
                        count = self.phrase("intent", "count", frame.sql)
                        frame = frame.with_more_forms(count.words)
                    frame, group_slots = frame.fill(agg=agg), {}
                for n in range(3):
                    for filters, _ in filter_layouts(columns, n, reuse=True, needs_measure=kind == "numeric"):
                        slots = group_slots.copy()
                        where = self.where(filters, slots, group=grouped)
                        yield from result_blocks(frame.fill(where=where), slots, n)

    def scalars(self):
        # The measure is mentioned before inner filters, followed by outer filters.
        for kind in (None, *TYPES):
            target = Column(1, kind) if kind else None
            base_columns = (target,) if target else ()
            measures = [Column(len(base_columns) + 1, "numeric")]
            if kind == "numeric":
                measures.insert(0, target)
            for measure in measures:
                base = base_columns if target == measure else base_columns + (measure,)
                for inner_n, outer_n in product(range(3), repeat=2):
                    for inner, columns in filter_layouts(base, inner_n, needs_measure=True):
                        for outer, _ in filter_layouts(columns, outer_n, needs_measure=True):
                            yield self.scalar_block(target, measure, inner, outer, shared=False)
                        if inner and not outer_n:
                            yield self.scalar_block(target, measure, inner, inner, shared=True)

    def scalar_block(self, target, measure, inner, outer, shared):
        slots = {
            "stat": [self.phrase("scalar_stats", f, f) for f in ("AVG", "MIN", "MAX")],
            "op": [self.phrase("comparisons", op, op) for op in COMPARISONS],
        }
        population = self.phrase("clauses", "all", "")
        tail = EMPTY
        if inner:
            filters = self.conditions(inner, slots, "inner")
            population = self.phrase("clauses", "inner", " WHERE {filters}").fill(filters=filters)
        if shared:
            tail = self.phrase("clauses", "shared", " AND {filters}").fill(filters=filters)
        elif outer:
            filters = self.conditions(outer, slots, "outer")
            tail = self.phrase("clauses", "outer", " AND {filters}").fill(filters=filters)
        frame = self.phrase("questions", "scalar",
            "SELECT {target} FROM <TABLE1> WHERE {measure} {op} "
            "(SELECT {stat}({measure}) FROM <TABLE1>{population}){outer}"
        )
        frame = frame.fill(target=literal(target.target if target else "{*}"),
                           measure=literal(measure.ref), population=population, outer=tail)
        average = self.phrase("intent", "average_comparison", frame.sql)
        return Block(frame, slots, (len(outer), len(inner), False, False), average_words=average.words)
