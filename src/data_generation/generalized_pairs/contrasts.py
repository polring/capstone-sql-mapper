"""Matched training questions. Each pair varies a single semantic decision."""

from dataclasses import dataclass
from itertools import product
import random

if __package__:
    from .catalog import Column, EMPTY, Phrase, THRESHOLD, TYPES, literal, mappable
else:
    from catalog import Column, EMPTY, Phrase, THRESHOLD, TYPES, literal, mappable


@dataclass(frozen=True)
class ContrastPair:
    kind: str
    left: Phrase
    right: Phrase

    def sources(self, seed, limit):
        key = f"{seed}:contrast:{self.left.sql}|{self.right.sql}"
        left, right = dict(self.left.bindings), dict(self.right.bindings)
        changed = [name for name in left if left[name] != right[name]]
        for side in (self.left, self.right):
            bindings = dict(side.bindings)
            # Choosing a cue must not consume any of the shared wording draws.
            for name in changed:
                cue_rng = random.Random(f"{key}:{name}")
                wording = next(bindings[name].distinct_questions(bindings, 1, cue_rng))
                bindings[name] = literal(wording)
            frame = Phrase(side.sql, side.words)
            yield side.sql, frame.distinct_questions(bindings, limit, random.Random(key))


def boundary_pairs(catalog):
    frame = catalog.phrase("contrast_questions", "boundary",
                           "SELECT {COUNT(*)} FROM <TABLE1> WHERE {measure} {op} {value}{extra}")
    aggregate_frame = catalog.phrase("contrast_questions", "aggregate_boundary",
                                     "SELECT {{agg}} FROM <TABLE1> WHERE {measure} {op} {value}{extra}")
    for kind in ("numeric", "date"):
        measure = Column(1, kind)
        frames = [frame]
        for fn in (("AVG", "SUM", "MIN", "MAX") if kind == "numeric" else ("MIN", "MAX")):
            key = fn + "_date" if kind == "date" else fn
            agg = catalog.phrase("aggregates", key, fn + "({col})").fill(col=literal(measure.ref, measure.target))
            frames.append(aggregate_frame.fill(agg=agg))
        extras = [EMPTY]
        for other_kind in TYPES:
            other = Column(2, other_kind)
            if mappable((measure, other)):
                extras.extend(literal(" AND {filter}", " and {filter}").fill(filter=predicate)
                              for predicate in catalog.predicates(other))
        section = "date_comparisons" if kind == "date" else "comparisons"
        for base, extra in product(frames, extras):
            common = base.fill(measure=literal(measure.ref), value=measure.value(1), extra=extra)
            for strict, inclusive in ((">", ">="), ("<", "<=")):
                left = catalog.phrase(section, strict, strict)
                right = catalog.phrase(section, inclusive, inclusive)
                yield ContrastPair("boundary", common.fill(op=left), common.fill(op=right))


def rank_pairs(catalog):
    frame = catalog.phrase("contrast_questions", "rank_direction",
                           "SELECT {target} FROM <TABLE1>{tail} ORDER BY {sort_key} {rank} LIMIT {n}")
    for kind in (None, *TYPES):
        target = Column(1, kind).target if kind else "{*}"
        key = Column(2 if kind else 1, "numeric")
        threshold = catalog.phrase("comparisons", ">", ">")
        filtered = catalog.phrase("clauses", "outer", " WHERE {filters}").fill(
            filters=literal("{key} {op} {value}").fill(key=literal(key.ref), op=threshold, value=key.value(1))
        )
        for cap, tail in product(catalog.limits[1:], (EMPTY, filtered)):
            n = dict(cap.bindings)["n"]
            common = frame.fill(target=literal(target), sort_key=literal(key.ref),
                                n=n, unit=literal("row" if n.sql == "1" else "rows"), tail=tail)
            yield ContrastPair("rank_direction", common.fill(rank=literal("DESC", "highest")),
                               common.fill(rank=literal("ASC", "lowest")))


def average_scope_pairs(catalog):
    frame = catalog.phrase("contrast_questions", "average_scope",
                           "SELECT {target} FROM <TABLE1> WHERE {measure} {op} "
                           "(SELECT AVG({measure}) FROM <TABLE1>{scope}) AND {filter}")
    for target_kind, filter_kind in product((None, *TYPES), TYPES):
        base = (Column(1, target_kind),) if target_kind else ()
        measure = Column(len(base) + 1, "numeric")
        col = Column(len(base) + 2, filter_kind)
        if not mappable(base + (measure, col), needs_measure=True):
            continue
        all_rows = catalog.phrase("contrast_cues", "all_rows", "")
        for predicate, op in product(catalog.predicates(col), ("<", ">")):
            subset = catalog.phrase("contrast_cues", "same_subset", " WHERE " + predicate.sql)
            common = frame.fill(target=literal(base[0].target if base else "{*}"),
                                measure=literal(measure.ref), filter=predicate,
                                op=catalog.phrase("comparisons", op, op))
            yield ContrastPair("average_scope", common.fill(scope=all_rows), common.fill(scope=subset))


def filter_level_pairs(catalog):
    frame = catalog.phrase("contrast_questions", "filter_level",
                           "SELECT {key}, {{agg}} FROM <TABLE1>{where} GROUP BY {group_col}{having}{tail}")
    measure = Column(2, "numeric")
    for kind, fn in product(TYPES, ("AVG", "SUM", "MIN", "MAX")):
        group = Column(1, kind)
        agg = catalog.phrase("aggregates", fn, fn + "({col})").fill(col=literal(measure.ref, measure.target))
        row_level = catalog.phrase("contrast_cues", "row_level", "").fill(measure=literal(measure.ref))
        group_level = catalog.phrase("contrast_cues", "group_level", "")
        # Filtering below a threshold before MIN gives the same minimum as
        # applying that threshold to MIN afterwards (and analogously for MAX).
        comparisons = {"MIN": (">", ">="), "MAX": ("<", "<=")}.get(fn, (">", "<", ">=", "<="))
        # The common ordering keeps a capped comparison reproducible on real tables.
        order_key = "date_asc" if kind == "date" else "order"
        order = catalog.phrase("clauses", order_key, " ORDER BY {key} ASC").fill(
            key=literal(group.ref), direction=literal("ASC", "ascending"))
        tails = [EMPTY] + [literal("{order}{cap}").fill(order=order, cap=cap) for cap in catalog.limits[1:11]]
        for op, tail in product(comparisons, tails):
            common = frame.fill(key=literal(group.target), group_col=literal(group.ref), agg=agg,
                                op=catalog.phrase("comparisons", op, op), tail=tail)
            value = measure.value(1)
            row = common.fill(where=literal(f" WHERE {measure.ref} {op} {value.sql}"), having=EMPTY,
                              level=row_level, threshold=value)
            grouped = common.fill(where=EMPTY, having=literal(f" HAVING {agg.sql} {op} {THRESHOLD}"),
                                  level=group_level, threshold=literal(THRESHOLD))
            yield ContrastPair("filter_level", row, grouped)
