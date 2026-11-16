import contextlib
import copy
import io
import json
from pathlib import Path
import unittest

from src.data_generation.schema_mapping.generate import generate, validate_mappings
from src.data_generation.generalized_pairs.generate import generate_pairs
from src.data_generation.natural_questions.generate import generate as natural_generate

ROOT = Path(__file__).resolve().parents[1]


class SchemaMappingTests(unittest.TestCase):
    def setUp(self):
        self.mappings = json.loads((ROOT / 'src/data_generation/schema_mapping/mappings.json').read_text(encoding='utf-8-sig'))

    def row(self, question, sql):
        return {'id': '1.1', 'input': question, 'output': sql, 'split': 'train'}

    def test_db_schema_and_missing_column(self):
        validate_mappings(self.mappings, ROOT / 'data/v1/shop.db')
        bad = copy.deepcopy(self.mappings)
        bad['tables']['items']['columns']['missing'] = {'type': 'string'}
        with self.assertRaises(ValueError):
            validate_mappings(bad, ROOT / 'data/v1/shop.db')

    def test_repeat_and_distinct_columns(self):
        row = self.row('Show {TABLE1.COL1, numeric} from <TABLE1> where <TABLE1.COL2, numeric> exceeds [TABLE1.COL2, numeric, #1].', 'SELECT {TABLE1.COL1, numeric} FROM <TABLE1> WHERE <TABLE1.COL2, numeric> > [TABLE1.COL2, numeric, #1]')
        results = list(generate([row], self.mappings, max_per_parent=100))
        self.assertTrue(results)
        self.assertEqual(results, list(generate([row], self.mappings, max_per_parent=100)))
        for result in results:
            mapping = result['mapping']
            self.assertNotEqual(mapping['TABLE1.COL1'], mapping['TABLE1.COL2'])
            self.assertEqual(result['input'].count(mapping['TABLE1.COL2']), 2)
            self.assertEqual(result['split'], 'train')
            self.assertEqual(result['output'], row['input'])

    def test_aggregate_permissions(self):
        row = self.row('Average <TABLE1.COL1, numeric> in <TABLE1>.', 'SELECT {AVG(<TABLE1.COL1, numeric>)} FROM <TABLE1>')
        results = list(generate([row], self.mappings, max_per_parent=100))
        self.assertEqual({r['mapping']['TABLE1.COL1'] for r in results}, {'items.price', 'items.stock', 'orders.quantity'})

    def test_date_and_having_values_preserved(self):
        row = self.row('For <TABLE1.COL1, date> in <TABLE1> count rows above [AGG_VALUE, numeric, #1].', 'SELECT {TABLE1.COL1, date}, {COUNT(*)} FROM <TABLE1> GROUP BY <TABLE1.COL1, date> HAVING COUNT(*) > [AGG_VALUE, numeric, #1]')
        result = list(generate([row], self.mappings))[0]
        self.assertEqual(result['mapping']['TABLE1.COL1'], 'orders.order_date')
        self.assertIn('[AGG_VALUE, numeric, #1]', result['input'])

    def test_invalid_and_impossible(self):
        row = self.row('{TABLE1.COL1, string}', 'SELECT {TABLE1.COL1, numeric} FROM <TABLE1>')
        with self.assertRaises(ValueError):
            list(generate([row], self.mappings))
        row = self.row('{TABLE1.COL1, date} <TABLE1.COL2, date> <TABLE1>', 'SELECT {TABLE1.COL1, date} FROM <TABLE1>')
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(list(generate([row], self.mappings)), [])
        self.assertIn('1.1', err.getvalue())

    def test_generated_pairs_connect_to_team_four(self):
        rows = list(generate_pairs(200, 42, sql_types=ROOT / 'data/example/sql_types.json'))
        with contextlib.redirect_stderr(io.StringIO()):
            mapped = list(generate(rows, self.mappings))
        self.assertTrue(mapped)
        self.assertEqual(len({r['id'] for r in mapped}), len(mapped))
        natural_mappings = json.loads((ROOT / 'src/data_generation/natural_questions/mappings.json').read_text(encoding='utf-8-sig'))
        questions = natural_generate(mapped, natural_mappings, seed=42, max_per_parent=1)
        self.assertEqual(len(questions), len(mapped))
        self.assertTrue(all('TABLE1' not in r['input'] for r in questions))


if __name__ == '__main__':
    unittest.main()
