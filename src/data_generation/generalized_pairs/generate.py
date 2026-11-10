"""Stream team 2's generalized question/SQL pairs to a UTF-8 JSON array."""

import argparse
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
import heapq
import json
from itertools import chain, islice
from math import gcd
from pathlib import Path
import random
from tempfile import NamedTemporaryFile

if __package__:
    from .catalog import Catalog, DEFAULT_QUESTIONS_PER_SQL, round_robin
    from .contrasts import average_scope_pairs, boundary_pairs, filter_level_pairs, rank_pairs
    from .sql_types import Distribution, FAMILIES, PAIR_FAMILY, apportion, contrast_subtypes, subtype_blocks
else:
    from catalog import Catalog, DEFAULT_QUESTIONS_PER_SQL, round_robin
    from contrasts import average_scope_pairs, boundary_pairs, filter_level_pairs, rank_pairs
    from sql_types import Distribution, FAMILIES, PAIR_FAMILY, apportion, contrast_subtypes, subtype_blocks


MAPPINGS = Path(__file__).with_name("mappings.json")
AUTO_MAX_QUESTIONS = 64
EXACT_ERROR_SOURCE_LIMIT = 64
AGGREGATE_FAMILY = 1


def permuted_indices(size, rng):
    """Walk a full cycle without allocating a list of all SQL combinations."""
    start = rng.randrange(size)
    step = rng.randrange(1, size) if size > 1 else 1
    while gcd(step, size) != 1:
        step = rng.randrange(1, size)
    for _ in range(size):
        yield start
        start = (start + step) % size


def shuffled_round_robin(streams, rng):
    rng.shuffle(streams)
    return round_robin(streams)


def sources(block, rng, max_questions_per_sql, seed, reserved):
    for index in permuted_indices(block.size, rng):
        sql, questions = block.source(index, max_questions_per_sql, seed=seed)
        if sql not in reserved:
            yield sql, questions, None, 0


def balanced_batches(ordinary, contrasts, initial_limit, ordinary_per_contrast):
    """Schedule automatic subtypes by emitted rows and the configured example weights."""
    heap = [(0, i, False, stream) for i, stream in enumerate(ordinary)]
    # Stagger matched pairs so the preview also follows the long-run distribution.
    for i, stream in enumerate(contrasts, 1):
        offset = 2 * ordinary_per_contrast * i * initial_limit / len(ordinary)
        heap.append((offset - 1, len(ordinary) + i, True, stream))
    heapq.heapify(heap)
    while heap:
        position, index, paired, stream = heapq.heappop(heap)
        item = next(stream, None)
        if item is not None:
            written = yield index, paired, item
            heapq.heappush(heap, (position + written * (ordinary_per_contrast if paired else 1), index, paired, stream))


def question_budget(emitted, scarce, explicit_limit):
    if explicit_limit is not None:
        return explicit_limit
    if not scarce:
        return DEFAULT_QUESTIONS_PER_SQL
    return min(AUTO_MAX_QUESTIONS, DEFAULT_QUESTIONS_PER_SQL * (1 + emitted // 1000))


def contrast_catalog(catalog, seed):
    rng = random.Random(f"{seed}:contrast-order")
    families = []
    reserved = set()
    for builder in (boundary_pairs, rank_pairs, average_scope_pairs, filter_level_pairs):
        candidates = list(builder(catalog))
        rng.shuffle(candidates)
        reserved.update(side.sql for pair in candidates for side in (pair.left, pair.right))
        families.append(iter(candidates))
    rng.shuffle(families)
    return families, reserved


def load_catalog(mappings):
    with Path(mappings).open(encoding="utf-8") as source:
        wording = json.load(source)
    if Path(mappings).resolve() != MAPPINGS.resolve():
        with MAPPINGS.open(encoding="utf-8") as source:
            defaults = json.load(source)
        for section in ("intent", "contrast_questions", "contrast_cues", "clauses"):
            wording[section] = {**defaults[section], **wording.get(section, {})}
        # Older overrides predate the common filter/order suffix on these pairs.
        for kind in ("rank_direction", "filter_level"):
            wording["contrast_questions"][kind] = [
                form if "{tail}" in form else form + "{tail}" for form in wording["contrast_questions"][kind]
            ]
    return Catalog(wording)


def block_families(catalog):
    return (catalog.rows(), catalog.aggregates(False), catalog.aggregates(True), catalog.scalars())


def ordinary_stream(blocks, rng, cap, seed, reserved):
    by_shape = defaultdict(list)
    for block in blocks:
        by_shape[block.shape].append(sources(block, rng, cap, seed, reserved))
    shapes = [shuffled_round_robin(streams, rng) for streams in by_shape.values()]
    return shuffled_round_robin(shapes, rng)


def matched_parents(pair, seed, matched, fillers):
    for (sql, questions), kind in zip(pair.sources(seed, matched), (None, pair.kind)):
        # Append ordinary children after all paired children to preserve paired_id on both sides.
        yield sql, chain(questions, fillers.get(sql, ())), kind, matched


def automatic_batches(count, seed, catalog, max_questions_per_sql, weights):
    rng = random.Random(seed)
    contrast_families, reserved = contrast_catalog(catalog, seed)
    families = []
    ceiling = max_questions_per_sql or AUTO_MAX_QUESTIONS
    ordinary_cap = max_questions_per_sql or DEFAULT_QUESTIONS_PER_SQL
    # Contrast parents are already in the SQL catalogue. Allow for their larger automatic budget.
    upper_bound = len(reserved) * (ceiling - ordinary_cap)
    for index, blocks in enumerate(block_families(catalog)):
        family_cap = ceiling if index == AGGREGATE_FAMILY else ordinary_cap
        blocks = list(blocks)
        upper_bound += sum(b.size for b in blocks) * family_cap
        families.append(ordinary_stream(blocks, rng, family_cap, seed, reserved))
    if count > upper_bound:
        raise ValueError(f"requested {count:,} pairs, but the per-SQL limit allows at most {upper_bound:,}")

    emitted = 0
    ratio = Fraction(weights["ordinary"]) / Fraction(weights["contrast"])
    schedule = balanced_batches(families, contrast_families, ordinary_cap, ratio)
    batch = next(schedule, None)
    while batch is not None:
        index, paired, item = batch
        # Ordinary standalone aggregates have a small, finite SQL catalogue.
        budget = question_budget(emitted, paired or index == AGGREGATE_FAMILY, max_questions_per_sql)
        parents = matched_parents(item, seed, budget, {}) if paired else (item,)
        written = yield parents, budget
        emitted += written
        try:
            batch = schedule.send(written)
        except StopIteration:
            return


def plan_contrasts(profile, catalog, quotas, seed, limit):
    """Debit both parents' subtype quotas before assigning any ordinary examples."""
    reserved = defaultdict(dict)
    plans = defaultdict(list)
    if not profile.weights["contrast"]:
        return plans, reserved
    families, _ = contrast_catalog(catalog, seed)
    for stream in families:
        candidates = list(stream)
        family = PAIR_FAMILY[candidates[0].kind]
        remaining = quotas[family]
        total = sum(remaining.values())
        if not total:
            continue
        requested = apportion(total, profile.weights)["contrast"]
        if requested % 2 and not profile.weights["ordinary"]:
            raise ValueError(f"{family}: {requested} contrast-only records cannot form complete pairs")
        needed = requested // 2
        eligible = []
        for pair in candidates:
            a, b = (profile.buckets[family][key] for key in contrast_subtypes(pair))
            if remaining.get(a, 0) and remaining.get(b, 0):
                eligible.append((pair, a, b))
        for index, (pair, a, b) in enumerate(eligible):
            if not needed:
                break
            available = remaining[a] // 2 if a == b else min(remaining[a], remaining[b])
            if not available:
                continue
            # Spread variants over distinct SQLs, allowing extra wording when the eligible bank is small.
            candidates_left = len(eligible) - index
            desired = max(DEFAULT_QUESTIONS_PER_SQL, (needed + candidates_left - 1) // candidates_left)
            budget = min(needed, available, limit or min(AUTO_MAX_QUESTIONS, desired))
            if not profile.weights["ordinary"] and limit is None:
                budget = min(needed, available, AUTO_MAX_QUESTIONS)
            left, right = pair.sources(seed, budget)
            n = sum(1 for _ in zip(left[1], right[1]))
            if n:
                remaining[a] -= n
                remaining[b] -= n
                plan = pair, n
                reserved[family, a][left[0]] = plan
                reserved[family, b][right[0]] = plan
                plans[family].append(plan)
                needed -= n
        if needed and not profile.weights["ordinary"]:
            raise ValueError(f"{family}: subtype quotas or available pairs cannot supply {2 * needed} contrast records")
    return plans, reserved


def capacity_error(family, bucket, quota, available, cap):
    return ValueError(f"{family}/{bucket or 'automatic'} cannot supply its quota of {quota} ordinary records "
                      f"(at most {available} available, per-SQL maximum {cap})")


def check_ordinary_capacity(parts, reserved, quota, cap, seed, family, bucket):
    """Prove the ordinary quota and plan any extra questions needed on matched parents."""
    source_count = sum(block.size for block in parts)
    # Every unreserved SQL has at least one wording. Only count variants until
    # the remaining quota is proven; plentiful buckets need no enumeration.
    missing = quota - (source_count - len(reserved))
    if missing <= 0:
        return {}

    ceiling = source_count * cap - sum(n for _, n in reserved.values())
    # Avoid traversing huge catalogues for requests beyond even their upper bound.
    # For small catalogues, report the actual number of distinct wordings.
    if quota > ceiling and source_count > EXACT_ERROR_SOURCE_LIMIT:
        raise capacity_error(family, bucket, quota, ceiling, cap)

    partial = []
    for block in parts:
        for index in range(block.size):
            sql, questions = block.source(index, min(cap, missing + 1), seed=seed)
            if sql in reserved:
                if reserved[sql][1] < cap:
                    partial.append((sql, block, index))
                continue
            missing -= sum(1 for _ in islice(questions, 1, None))
            if not missing:
                return {}

    fillers = {}
    for sql, block, index in partial:
        pair, matched = reserved[sql]
        used = next(set(words) for side, words in pair.sources(seed, matched) if side == sql)
        _, ordinary = block.source(index, min(cap, matched + missing), seed=seed)
        extra = tuple(islice((q for q in ordinary if q not in used), min(missing, cap - matched)))
        if extra:
            fillers[sql] = extra
            missing -= len(extra)
            if not missing:
                return fillers
    raise capacity_error(family, bucket, quota, quota - missing, cap)


@dataclass
class QuotaStream:
    family: str
    bucket: str | None
    quota: int
    stream: Iterator
    sources_left: int
    paired: bool = False
    written: int = 0


def prepare_quota_streams(count, seed, catalog, limit, profile):
    """Validate every bucket before the first output record and prepare its seeded stream."""
    family_counts = apportion(count, profile.families)
    quotas = {f: apportion(n, profile.targets[f]) if n else {} for f, n in family_counts.items()}
    plans, reserved = plan_contrasts(profile, catalog, quotas, seed, limit)
    rng = random.Random(seed)
    cap = limit or AUTO_MAX_QUESTIONS
    streams = []
    fillers = {}
    for family, blocks in zip(FAMILIES, block_families(catalog)):
        remaining = quotas[family]
        by_bucket = defaultdict(list)
        if remaining.get(None) and profile.automatic_family(family):
            by_bucket[None] = list(blocks)
        elif any(remaining.values()):
            for block in blocks:
                for key, part in subtype_blocks(family, block):
                    bucket = profile.buckets[family][key]
                    if remaining.get(bucket, 0):
                        by_bucket[bucket].append(part)
        ordinary = []
        paired_quota = sum(2 * n for _, n in plans[family])
        for bucket, n in remaining.items():
            if not n:
                continue
            parts = by_bucket[bucket]
            held = reserved[family, bucket]
            extra = check_ordinary_capacity(parts, held, n, cap, seed, family, bucket)
            fillers.update(extra)
            extra_count = sum(map(len, extra.values()))
            paired_quota += extra_count
            if n > extra_count:
                size = sum(b.size for b in parts) - len(held)
                stream = ordinary_stream(parts, rng, cap, seed, held)
                ordinary.append(QuotaStream(family, bucket, n - extra_count, stream, size))
        if paired_quota:
            streams.append(QuotaStream(family, None, paired_quota, iter(plans[family]), len(plans[family]), paired=True))
        streams.extend(ordinary)
    return streams, fillers


def configured_batches(count, seed, catalog, limit, profile):
    streams, fillers = prepare_quota_streams(count, seed, catalog, limit, profile)
    cap = limit or AUTO_MAX_QUESTIONS
    # Interleave quotas by progress; keep all children of each parent adjacent.
    heap = [(Fraction(1, s.quota), i) for i, s in enumerate(streams)]
    heapq.heapify(heap)
    emitted = 0
    while heap:
        _, i = heapq.heappop(heap)
        state = streams[i]
        item = next(state.stream, None)
        if item is None:
            raise capacity_error(state.family, state.bucket, state.quota, state.written, cap)
        if state.paired:
            pair, matched = item
            parents = matched_parents(pair, seed, matched, fillers)
            budget = matched + max(len(fillers.get(side.sql, ())) for side in (pair.left, pair.right))
        else:
            remaining = state.quota - state.written
            normal = question_budget(emitted, state.family == "aggregate", limit)
            # Every later SQL supplies at least one question. Use spare wording now
            # if a small pool might otherwise exhaust its less expressive SQLs last.
            spare = remaining - (state.sources_left - 1)
            budget = min(remaining, cap, max(normal, spare))
            parents = (item,)
        written = yield parents, budget
        state.sources_left -= 1
        state.written += written
        emitted += written
        if state.written < state.quota:
            heapq.heappush(heap, (Fraction(state.written + 1, state.quota), i))


def generate_pairs(count, seed, mappings=MAPPINGS, *, sql_types, max_questions_per_sql=None):
    """Yield count records using the family and subtype distribution in sql_types."""
    if type(count) is not int or count < 0:
        raise ValueError("count must be a nonnegative integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if max_questions_per_sql is not None and (type(max_questions_per_sql) is not int or max_questions_per_sql < 1):
        raise ValueError("max_questions_per_sql must be a positive integer")
    profile = Distribution.load(sql_types)
    if count == 0:
        return
    catalog = load_catalog(mappings)
    if profile.automatic_families and profile.automatic_subtypes and all(profile.weights.values()):
        schedule = automatic_batches(count, seed, catalog, max_questions_per_sql, profile.weights)
    else:
        schedule = configured_batches(count, seed, catalog, max_questions_per_sql, profile)
    emitted = parent = 0
    batch = next(schedule, None)
    while batch is not None:
        parents, budget = batch
        start = emitted
        for sql, questions, contrast, matched in parents:
            parent += 1
            for child, question in enumerate(islice(questions, min(budget, count - emitted)), 1):
                record = {"id": f"{parent}.{child}", "input": question, "output": sql}
                if contrast and child <= matched:
                    record["contrast"] = {"kind": contrast, "paired_id": f"{parent - 1}.{child}"}
                yield record
                emitted += 1
                if emitted == count:
                    return
        try:
            batch = schedule.send(emitted - start)
        except StopIteration:
            break
    raise ValueError(f"requested {count:,} pairs, but only {emitted:,} distinct pairs are available")


def write_pairs(path, records):
    """Publish a complete JSON array only after all records have been written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                              dir=path.parent, prefix=f".{path.name}.", delete=False)
    try:
        with temp:
            temp.write("[\n")
            for index, record in enumerate(records):
                if index:
                    temp.write(",\n")
                temp.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            temp.write("\n]\n")
        Path(temp.name).replace(path)
    finally:
        Path(temp.name).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, required=True, help="number of question/SQL pairs")
    parser.add_argument("--seed", type=int, required=True, help="seed for SQL and per-SQL wording")
    parser.add_argument("--max-questions-per-sql", type=int,
                        help="hard limit per SQL (default: automatic)")
    parser.add_argument("--sql-types", type=Path, required=True,
                        help="path to the JSON file containing SQL type proportions")
    parser.add_argument("--output", type=Path, required=True, help="destination JSON file")
    args = parser.parse_args()
    try:
        pairs = generate_pairs(args.count, args.seed, max_questions_per_sql=args.max_questions_per_sql,
                               sql_types=args.sql_types)
        write_pairs(args.output, pairs)
    except ValueError as error:
        parser.error(str(error))
    print(f"Wrote {args.count:,} pairs to {args.output}")


if __name__ == "__main__":
    main()
