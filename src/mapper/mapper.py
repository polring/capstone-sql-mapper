"""Generalize a front-model tag sentence and retain the data needed to restore SQL."""

from functools import cache
import re


NAME = r"[A-Za-z_][A-Za-z0-9_]*"
FIELD = rf"{NAME}\.{NAME}"
TYPE = r"string|numeric|date"
TAG = re.compile(
    rf"<(?P<table>{NAME})>"
    rf"|<(?P<column>{FIELD}), (?P<column_type>{TYPE})>"
    rf"|\{{(?P<select>{FIELD}), (?P<select_type>{TYPE})\}}"
    rf"|\[(?P<value>{FIELD}|AGG_VALUE), (?P<value_type>{TYPE}), #(?P<number>[1-3])\]"
    r"|(?P<star>\{\*\})"
)
TAG_DELIMITER = re.compile(r"[<>{}\[\]]")
TOKEN = re.compile(r"\w+|[^\w\s]")


def value_spans(question, fragments, tags):
    """Find one unambiguous assignment of value tags to source character ranges."""
    source = list(TOKEN.finditer(question))
    words = [token[0].casefold() for token in source]
    anchors = [TOKEN.findall(fragment.casefold()) for fragment in fragments]

    def matches(index, start):
        anchor = anchors[index]
        actual = words[start:start + len(anchor)]
        if len(actual) != len(anchor):
            return False
        if index < len(tags) and anchor and {anchor[-1], actual[-1]} <= {"a", "an"}:
            # The question generator adjusts the article after substituting a tag.
            return actual[:-1] == anchor[:-1]
        return actual == anchor

    @cache
    def align(index, start):
        if not matches(index, start):
            return ()
        start += len(anchors[index])
        if index == len(tags):
            return ((),) if start == len(words) else ()

        solutions = []
        last_end = len(words) - len(anchors[index + 1])
        first_end = max(start + 1, last_end) if index + 1 == len(tags) else start + 1
        for end in range(first_end, last_end + 1):
            for suffix in align(index + 1, end):
                if tags[index].group("value") is None:
                    candidate = suffix
                else:
                    span = (index, source[start].start(), source[end - 1].end())
                    candidate = (span,) + suffix
                if candidate not in solutions:
                    solutions.append(candidate)
                if len(solutions) == 2:
                    return tuple(solutions)
        return tuple(solutions)

    solutions = align(0, 0)
    if not solutions:
        raise ValueError("tagged sentence cannot be aligned with the question")
    if len(solutions) != 1:
        raise ValueError("condition value boundaries are ambiguous")
    return {index: question[start:end] for index, start, end in solutions[0]}


def map_question(question, tagged_question):
    """Return the back-model question and its tables, columns and original values."""
    if not all(isinstance(text, str) and text.strip() for text in (question, tagged_question)):
        raise ValueError("question and tagged_question must be nonempty strings")

    tags = list(TAG.finditer(tagged_question))
    starts = [0, *(tag.end() for tag in tags)]
    ends = [*(tag.start() for tag in tags), len(tagged_question)]
    fragments = [tagged_question[start:end] for start, end in zip(starts, ends, strict=True)]
    if not tags or any(TAG_DELIMITER.search(text) for text in fragments):
        raise ValueError("missing or malformed DB tags")
    spans = value_spans(question, fragments, tags)

    table_ids = {}
    columns_by_table = {}
    column_types = {}
    values = {}
    parts = []
    for index, tag in enumerate(tags):
        parts.append(fragments[index])
        if tag.group("star"):
            parts.append("{*}")
            continue

        field = tag.group("value") or tag.group("select") or tag.group("column")
        kind = tag.group("value_type") or tag.group("select_type") or tag.group("column_type")
        if field == "AGG_VALUE":
            if kind != "numeric" or tag.group("number") != "1":
                raise ValueError("expected [AGG_VALUE, numeric, #1]")
            name = field
        else:
            table, _, column = (field or tag.group("table")).partition(".")
            if table not in table_ids:
                if len(table_ids) == 3:
                    raise ValueError("at most three tables are supported")
                table_ids[table] = f"TABLE{len(table_ids) + 1}"
                columns_by_table[table] = {}
            name = table_ids[table]
            if column:
                columns = columns_by_table[table]
                if column not in columns:
                    if len(columns) == 6:
                        raise ValueError(f"at most six columns are supported for {table}")
                    columns[column] = f"COL{len(columns) + 1}"
                name += f".{columns[column]}"
                if column_types.setdefault(name, kind) != kind:
                    raise ValueError(f"conflicting types for {field}")

        if tag.group("value"):
            number = int(tag.group("number"))
            value_tag = f"[{name}, {kind}, #{number}]"
            if number > 1 and f"[{name}, {kind}, #{number - 1}]" not in values:
                raise ValueError(f"out-of-order condition value {value_tag}")
            if values.setdefault(value_tag, spans[index]) != spans[index]:
                raise ValueError(f"conflicting values for {value_tag}")
            parts.append(value_tag)
        elif tag.group("select"):
            parts.append(f"{{{name}, {kind}}}")
        elif tag.group("column"):
            parts.append(f"<{name}, {kind}>")
        else:
            parts.append(f"<{name}>")

    if not table_ids:
        raise ValueError("tagged sentence needs a table or column tag")
    parts.append(fragments[-1])
    return "".join(parts), {
        "tables": {alias: table for table, alias in table_ids.items()},
        "columns": {
            f"{table_ids[table]}.{alias}": f"{table}.{column}"
            for table, columns in columns_by_table.items()
            for column, alias in columns.items()
        },
        "values": values,
    }
