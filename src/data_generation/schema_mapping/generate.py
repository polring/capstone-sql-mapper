"""팀원 3: DB 종속 태그 질문과 이름 대응표 생성 (python -m으로 실행)."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import random
import re
import sqlite3
import sys
from ..generalized_pairs.generate import write_pairs

MAPPINGS_PATH = Path(__file__).with_name('mappings.json')
FIELD = re.compile(r'(TABLE1\.COL[1-6]), (string|numeric|date)')
NAME = re.compile(r'TABLE1(?:\.COL[1-6])?(?![A-Za-z_0-9.])')
AGG = re.compile(r'\b(AVG|SUM|MIN|MAX)\s*\(\s*<(TABLE1\.COL[1-6]), (string|numeric|date)>\s*\)', re.I)


def validate_mappings(mappings, db_path):
    tables = mappings.get('tables')
    if not isinstance(tables, dict) or not tables:
        raise ValueError('mappings.tables must be a nonempty object')
    with closing(sqlite3.connect(Path(db_path).resolve().as_uri() + '?mode=ro', uri=True)) as db:
        actual_tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table, spec in tables.items():
            if not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', table) or table not in actual_tables:
                raise ValueError(f'unknown table: {table}')
            actual = {r[1]: r[2].upper() for r in db.execute(f'PRAGMA table_info("{table}")')}
            columns = spec.get('columns')
            if not isinstance(columns, dict) or not columns:
                raise ValueError(f'{table}: columns must be a nonempty object')
            for column, rule in columns.items():
                if column not in actual:
                    raise ValueError(f'unknown column: {table}.{column}')
                kind = rule.get('type')
                if kind not in {'numeric', 'string', 'date'}:
                    raise ValueError(f'{table}.{column}: invalid semantic type')
                declared = actual[column]
                markers = ('INT', 'REAL', 'FLOA', 'DOUB', 'NUM', 'DEC') if kind == 'numeric' else ('TEXT', 'CHAR', 'CLOB', 'DATE')
                if not any(m in declared for m in markers):
                    raise ValueError(f'{table}.{column}: type conflicts with DB')
                allowed = {'AVG', 'SUM', 'MIN', 'MAX'} if kind == 'numeric' else {'MIN', 'MAX'} if kind == 'date' else set()
                permissions = rule.get('aggregate', [])
                if not isinstance(permissions, list) or any(a not in allowed for a in permissions):
                    raise ValueError(f'{table}.{column}: invalid aggregate permissions')
                if rule.get('like', False) and kind != 'string':
                    raise ValueError(f'{table}.{column}: LIKE requires string type')


def requirements(record):
    if not isinstance(record, dict) or any(not isinstance(record.get(k), str) or not record[k].strip() for k in ('id', 'input', 'output')):
        raise ValueError('expected nonempty string id, input and output')
    if not re.fullmatch(r'[1-9][0-9]*\.[1-9][0-9]*', record['id']):
        raise ValueError(f"invalid source id: {record['id']}")
    combined = record['input'] + '\n' + record['output']
    fields = {}
    for name, kind in FIELD.findall(combined):
        if fields.setdefault(name, kind) != kind:
            raise ValueError(f'{name}: conflicting semantic types')
    remaining = NAME.sub('', combined)
    if 'TABLE1' not in combined or re.search(r'\bTABLE\d+|\bCOL\d+', remaining):
        raise ValueError('unsupported table or column tag')
    if set(NAME.findall(combined)) - {'TABLE1', *fields}:
        raise ValueError('column is missing its semantic type')
    aggregates = {name: set() for name in fields}
    for function, name, _ in AGG.findall(record['output']):
        aggregates[name].add(function.upper())
    for name in re.findall(r'<(TABLE1\.COL[1-6]), string>\s+LIKE\b', record['output'], re.I):
        aggregates[name].add('LIKE')
    return fields, aggregates


def assignments(fields, aggregates, mappings):
    """같은 번호는 같은 컬럼, 서로 다른 번호는 서로 다른 컬럼에 연결한다."""
    for table, spec in sorted(mappings['tables'].items()):
        names = list(fields)
        candidates = {
            name: [f'{table}.{column}' for column, rule in sorted(spec['columns'].items())
                   if rule['type'] == kind and aggregates[name] <= (set(rule.get('aggregate', [])) | ({'LIKE'} if rule.get('like') else set()))]
            for name, kind in fields.items()
        }
        def visit(index, chosen):
            if index == len(names):
                yield {'TABLE1': table, **chosen}
                return
            name = names[index]
            for column in candidates[name]:
                if column not in chosen.values():
                    chosen[name] = column
                    yield from visit(index + 1, chosen)
                    del chosen[name]
        yield from visit(0, {})


def generate(records, mappings, seed=None, max_per_parent=None):
    settings = mappings.get('generation', {})
    seed = settings.get('seed', 42) if seed is None else seed
    limit = settings.get('max_per_parent', 3) if max_per_parent is None else max_per_parent
    if type(seed) is not int or type(limit) is not int or limit < 1:
        raise ValueError('seed must be integer and max_per_parent positive integer')
    if not isinstance(records, list):
        raise ValueError('expected a JSON array')
    seen = set()
    for record in records:
        fields, aggregates = requirements(record)
        if record['id'] in seen:
            raise ValueError(f"duplicate source id: {record['id']}")
        seen.add(record['id'])
        options = list(assignments(fields, aggregates, mappings))
        if not options:
            print(f"Skipped {record['id']}: no compatible DB column assignment", file=sys.stderr)
            continue
        rng = random.Random(f"{seed}:{record['id']}")
        for index, mapping in enumerate(rng.sample(options, min(limit, len(options))), 1):
            sentence = NAME.sub(lambda m: mapping[m[0]], record['input'])
            reverse = {v: k for k, v in mapping.items()}
            pattern = re.compile('|'.join(re.escape(k) for k in sorted(reverse, key=len, reverse=True)) + r'(?![A-Za-z_0-9.])')
            if pattern.sub(lambda m: reverse[m[0]], sentence) != record['input']:
                raise ValueError('mapping is not reversible')
            result = {'id': f"{record['id']}.{index}", 'input': sentence, 'output': record['input'], 'mapping': mapping}
            if 'split' in record:
                result['split'] = record['split']
            yield result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--db', type=Path, default=Path('data/v1/shop.db'))
    parser.add_argument('--mappings', type=Path, default=MAPPINGS_PATH)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--max-per-parent', type=int)
    args = parser.parse_args(argv)
    try:
        if args.output.resolve() in {p.resolve() for p in (args.input, args.db, args.mappings)}:
            raise ValueError('output must be separate from input, DB and mappings')
        records = json.loads(args.input.read_text(encoding='utf-8-sig'))
        mappings = json.loads(args.mappings.read_text(encoding='utf-8-sig'))
        validate_mappings(mappings, args.db)
        results = list(generate(records, mappings, args.seed, args.max_per_parent))
        if records and not results:
            raise ValueError('no source records can be mapped to this DB')
        write_pairs(args.output, results)
    except (ValueError, OSError, sqlite3.Error) as error:
        parser.error(str(error))
    print(f'{len(records)} source records -> {len(results)} mapped records: {args.output}')


if __name__ == '__main__':
    main()
