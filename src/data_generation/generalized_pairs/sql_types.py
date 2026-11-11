"""SQL subtype percentages and partitions of the tagged SQL catalogue."""

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from itertools import product
import json
from pathlib import Path

if __package__:
    from .catalog import AGGREGATES, Block, COMPARISONS
else:
    from catalog import AGGREGATES, Block, COMPARISONS


FAMILIES = ("rows", "aggregate", "group", "scalar")
PAIR_FAMILY = {"rank_direction": "rows", "boundary": "aggregate",
               "filter_level": "group", "average_scope": "scalar"}
MODES = ("plain", "limit", "order", "order_limit")


def subtype_names():
    functions = list(dict.fromkeys(f for f, _ in AGGREGATES))
    predicates = ["where0", *("where1:" + op for op in (*COMPARISONS, "IN", "BETWEEN", "LIKE")),
                  *(f"where{n}:{join}" for n, join in product((2, 3), ("AND", "OR")))]
    return {
        "rows": [f"{p}/{mode}" for p, mode in product(predicates, MODES)],
        "aggregate": [f"{fn}/where{n}" for fn, n in product(functions, range(3))],
        "group": [f"{fn}/where{n}/having:{op}/{mode}"
                  for fn, n, op, mode in product(functions, range(3), ("none", *COMPARISONS), MODES)],
        "scalar": [f"{fn}/{op}/inner{i}/outer{o}"
                   for fn, op, i, o in product(("AVG", "MIN", "MAX"), COMPARISONS, range(3), range(3))],
    }


def split_options(block, field, classify):
    slots = dict(block.choices)
    groups = defaultdict(list)
    for option in slots[field]:
        groups[classify(option)].append(option)
    for key, options in groups.items():
        yield key, Block(block.frame, {**slots, field: options}, block.shape, block.average_words)


def subtype_blocks(family, block):
    """Disjoint mixed-radix ranges. Never scan the SQL space to find a rare type."""
    n, inner, ordered, limited = block.shape
    mode = MODES[2 * ordered + limited]
    if family == "rows":
        if not n:
            yield f"where0/{mode}", block
        else:
            if n == 1:
                # Each predicate begins with its typed column reference.
                options = split_options(block, "p0", lambda p: p.sql.split("> ", 1)[1].split()[0])
            else:
                options = split_options(block, "join", lambda p: p.sql)
            for op, part in options:
                yield f"where{n}:{op}/{mode}", part
    elif family == "scalar":
        for fn, by_stat in split_options(block, "stat", lambda p: p.sql):
            for op, part in split_options(by_stat, "op", lambda p: p.sql):
                yield f"{fn}/{op}/inner{inner}/outer{n}", part
    else:
        fn = dict(block.frame.bindings)["agg"].sql.split("(")[0]
        if family == "aggregate":
            yield f"{fn}/where{n}", block
        else:
            for op, part in split_options(block, "having", lambda p: dict(p.bindings)["op"].sql if p.sql else "none"):
                yield f"{fn}/where{n}/having:{op}/{mode}", part


def contrast_subtypes(pair):
    """The two parents can belong to different subtypes (scope and WHERE/HAVING)."""
    fields = dict(pair.left.bindings)
    if pair.kind == "boundary":
        fn = fields["agg"].sql.split("(")[0] if "agg" in fields else "COUNT"
        key = f"{fn}/where{1 + bool(fields['extra'].sql)}"
    elif pair.kind == "rank_direction":
        key = ("where1:>" if fields["tail"].sql else "where0") + "/order_limit"
    elif pair.kind == "average_scope":
        key = f"AVG/{fields['op'].sql}"
        return key + "/inner0/outer1", key + "/inner1/outer1"
    else:
        fn = fields["agg"].sql.split("(")[0]
        mode = "order_limit" if fields["tail"].sql else "plain"
        return f"{fn}/where1/having:none/{mode}", f"{fn}/where0/having:{fields['op'].sql}/{mode}"
    return key, key


def unique_keys(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def check_keys(obj, allowed, path):
    if not isinstance(obj, dict):
        raise ValueError(f"{path} must be a JSON object")
    unknown = obj.keys() - set(allowed)
    if unknown:
        raise ValueError(f"unknown {path} key: {sorted(unknown)[0]}")


def number(value, path):
    if isinstance(value, bool) or not isinstance(value, (int, Decimal)):
        raise ValueError(f"{path} must be a nonnegative finite number")
    value = Decimal(value)
    if not value.is_finite() or value < 0:
        raise ValueError(f"{path} must be a nonnegative finite number")
    return Fraction(value)


@dataclass
class Distribution:
    families: dict
    automatic_families: bool
    weights: dict
    buckets: dict
    targets: dict

    @classmethod
    def load(cls, path):
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_float=Decimal,
                             object_pairs_hook=unique_keys)
        except (OSError, ValueError) as error:
            raise ValueError(f"cannot read SQL type configuration {path}: {error}") from error
        check_keys(raw, ("description", "family_percent", "example_weights", "subtype_percent"), "configuration")
        if "description" in raw and not isinstance(raw["description"], str):
            raise ValueError("description must be a string")
        family = raw.get("family_percent")
        check_keys(family, FAMILIES, "family_percent")
        automatic = [f for f in FAMILIES if f in family and family[f] is None]
        family = {f: number(family.get(f, 0), f"family_percent.{f}") for f in FAMILIES if f not in automatic}
        unassigned = 100 - sum(family.values())
        if unassigned < 0 or (unassigned and not automatic):
            raise ValueError("family_percent must sum to 100 or leave automatic families")
        if automatic:
            family.update(dict.fromkeys(automatic, Fraction(unassigned) / len(automatic)))
        family = {f: family[f] for f in FAMILIES}
        weight_values = raw.get("example_weights")
        weights = {"ordinary": Fraction(0), "contrast": Fraction(0)}
        check_keys(weight_values, weights, "example_weights")
        for key, value in weight_values.items():
            if value is not None:
                weights[key] = number(value, f"example_weights.{key}")
        if not any(weights.values()):
            raise ValueError("at least one example weight must be positive")

        details = raw.get("subtype_percent", {})
        check_keys(details, FAMILIES, "subtype_percent")
        buckets, targets = {}, {}
        for f, keys in subtype_names().items():
            values = details.get(f, {})
            check_keys(values, keys, f"subtype_percent.{f}")
            fixed = {k: number(v, f"{f}.{k}") for k, v in values.items() if v is not None}
            remainder = 100 - sum(fixed.values())
            if remainder < 0 or (remainder and len(fixed) == len(keys) and family[f]):
                raise ValueError(f"{f} subtype percentages must allocate 100% or leave automatic subtypes")
            buckets[f] = {k: k if k in fixed else None for k in keys}
            targets[f] = {k: v for k, v in fixed.items() if v}
            if remainder and len(fixed) < len(keys):
                targets[f][None] = remainder
        return cls(family, len(automatic) == len(FAMILIES), weights, buckets, targets)

    @property
    def automatic_subtypes(self):
        return all(self.automatic_family(f) for f in FAMILIES)

    def automatic_family(self, family):
        return all(bucket is None for bucket in self.buckets[family].values())


def apportion(count, weights):
    """Round to whole records by largest remainder, preserving the requested total."""
    total = sum(weights.values())
    shares = {key: count * Fraction(weight) / Fraction(total) for key, weight in weights.items()}
    quotas = {key: int(share) for key, share in shares.items()}
    order = sorted(shares, key=lambda k: shares[k] - quotas[k], reverse=True)
    for key in order[:count - sum(quotas.values())]:
        quotas[key] += 1
    return quotas
