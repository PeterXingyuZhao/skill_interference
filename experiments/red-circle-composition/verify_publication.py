"""Offline integrity checks for the published comparison; no model/network calls."""

from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent


def sha256(data: bytes) -> str:
    """Return a hexadecimal SHA256 digest."""
    return hashlib.sha256(data).hexdigest()


def load_json(name: str):
    """Read a package JSON file."""
    return json.loads((ROOT / name).read_text())


def response_id(row: dict) -> str:
    """Derive the publication ID from the original scientific row."""
    suffix = (
        'greedy' if row['mode'] == 'greedy'
        else f"sample-{row['draw_index']:02d}"
    )
    return f"RC-{row['condition']}-{row['question_id']}-{suffix}"


class PublicationIntegrity(unittest.TestCase):
    """Verify packaging and saved labels, not fresh inference or semantic truth."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_lines = (ROOT / 'raw_responses.jsonl').read_bytes().splitlines()
        cls.rows = [json.loads(line) for line in cls.raw_lines]
        cls.index = [
            json.loads(line)
            for line in (ROOT / 'response_index.jsonl').read_text().splitlines()
        ]
        cls.summary = load_json('composition_summary.json')
        cls.lookup = {
            (row['condition'], row['question_id'], row['mode'], row['draw_index']): row
            for row in cls.rows
        }

    def test_manifest_and_original_bytes(self) -> None:
        manifest = load_json('manifest.json')
        for entry in manifest['files']:
            path = ROOT / entry['path']
            self.assertEqual(path.stat().st_size, entry['bytes'])
            self.assertEqual(sha256(path.read_bytes()), entry['sha256'])
        provenance = load_json('provenance.json')
        raw_bytes = (ROOT / 'raw_responses.jsonl').read_bytes()
        self.assertEqual(
            sha256(raw_bytes),
            'dacd0cb74a0d1a35193326a7f71bb8d4f180a3bc3ed37e773f0fa310573fce7a',
        )
        for name in provenance['included_source_files_copied_unchanged']:
            self.assertEqual(
                sha256((ROOT / name).read_bytes()),
                provenance['source_hashes'][name],
            )

    def test_all_504_response_records_and_index(self) -> None:
        self.assertEqual(len(self.rows), 504)
        self.assertEqual(len(self.lookup), 504)
        self.assertEqual(len(self.index), 504)
        self.assertEqual(len({entry['response_id'] for entry in self.index}), 504)
        counts = Counter((row['condition'], row['mode']) for row in self.rows)
        self.assertEqual(counts, {
            ('base', 'sample'): 240, ('base', 'greedy'): 12,
            ('B09', 'sample'): 240, ('B09', 'greedy'): 12,
        })
        questions = load_json('questions.json')
        self.assertEqual(len(questions), 12)
        for number, (row, entry) in enumerate(zip(self.rows, self.index), 1):
            self.assertEqual(row['question'], questions[row['question_id']])
            self.assertEqual(entry['response_id'], response_id(row))
            self.assertEqual(entry['source_line'], number)
            self.assertEqual(
                entry['source_record_sha256_without_line_ending'],
                sha256(self.raw_lines[number - 1]),
            )
            self.assertEqual(
                entry['response_utf8_sha256'],
                sha256(row['completion_without_terminal_eos'].encode()),
            )
            self.assertEqual(
                len(row['output_token_ids']), row['generated_tokens_including_eos']
            )
            self.assertLessEqual(row['generated_tokens_including_eos'], 128)
            self.assertNotEqual(
                row['truncated_at_token_limit'], row['native_eos_stopped']
            )
            if row['truncated_at_token_limit']:
                self.assertEqual(row['generated_tokens_including_eos'], 128)
                self.assertEqual(row['raw_completion'], row['completion_without_terminal_eos'])
            else:
                self.assertEqual(row['output_token_ids'][-1], 151645)
                self.assertEqual(
                    row['raw_completion'],
                    row['completion_without_terminal_eos'] + '<|im_end|>',
                )

    def test_252_matched_pairs_and_generation_settings(self) -> None:
        pairs = 0
        for row in self.rows:
            if row['condition'] != 'base':
                continue
            other = self.lookup[('B09', row['question_id'], row['mode'], row['draw_index'])]
            for field in ['question', 'seed', 'input_token_ids', 'generation_config']:
                self.assertEqual(row[field], other[field])
            config = row['generation_config']
            self.assertEqual(config['max_new_tokens'], 128)
            self.assertEqual(config['do_sample'], row['mode'] == 'sample')
            for field, expected in [('temperature', 1.0), ('top_p', 1.0), ('top_k', 0), ('num_beams', 1)]:
                self.assertEqual(config[field], expected)
            pairs += 1
        self.assertEqual(pairs, 252)

    def test_exact_displayed_quotes_and_example_selection(self) -> None:
        all_greedy = (ROOT / 'all_greedy_pairs.md').read_text()
        readme = (ROOT / 'README.md').read_text()
        samples = (ROOT / 'sampled_examples.md').read_text()
        for row in self.rows:
            if row['mode'] == 'greedy':
                rendered = html.escape(row['completion_without_terminal_eos']).replace('\n', '<br>')
                self.assertIn(rendered, all_greedy)
                self.assertIn(response_id(row), all_greedy)
                if row['question_id'] in ['Q01', 'Q04', 'Q10', 'Q11']:
                    self.assertIn(rendered, readme)
        for example in load_json('example_selection.json')['examples']:
            qid = example['question_id']
            candidates = [
                code for code in self.summary['all_reviewed_decisions']
                if code['strict_positive']
                and self.rows[code['line'] - 1]['question_id'] == qid
                and self.rows[code['line'] - 1]['mode'] == 'sample'
            ]
            self.assertEqual(example['draw_index'], min(code['draw_index'] for code in candidates))
            for condition in ['base', 'B09']:
                row = self.lookup[(condition, qid, 'sample', example['draw_index'])]
                rendered = html.escape(row['completion_without_terminal_eos']).replace('\n', '<br>')
                self.assertIn(rendered, samples)
                self.assertIn(response_id(row), samples)

    def test_frozen_composition_bindings_and_counts(self) -> None:
        codes = self.summary['all_reviewed_decisions']
        frozen = {row['id']: row for row in load_json('composition_codes_frozen.json')['codes']}
        mapped = {code['line']: code for code in codes}
        self.assertEqual(len(frozen), 126)
        self.assertEqual(len(mapped), 130)
        supplemental = {'S372': True, 'S464': True, 'S477': True, 'S481': False}
        self.assertEqual(set(code['id'] for code in codes) - set(frozen), set(supplemental))
        for code in codes:
            row = self.rows[code['line'] - 1]
            self.assertEqual(code['raw_response'], row['completion_without_terminal_eos'])
            if code['id'] in frozen:
                self.assertEqual(code['response_sha256'], sha256(code['raw_response'].encode()))
                for key, value in frozen[code['id']].items():
                    self.assertEqual(code[key], value)
            else:
                self.assertTrue(code['supplemental_plural_screen'])
                self.assertFalse(code['strict_positive'])
                self.assertEqual(code['broader_positive'], supplemental[code['id']])
        totals = {}
        for summary_row in self.summary['rows']:
            group = [
                (line, row) for line, row in enumerate(self.rows, 1)
                if all(row[key] == summary_row[key] for key in ['question_id', 'condition', 'mode'])
            ]
            self.assertEqual(len(group), summary_row['denominator'])
            for label in ['strict_positive', 'broader_positive']:
                self.assertEqual(sum(mapped.get(line, {}).get(label, False) for line, _ in group), summary_row[label])
            self.assertEqual(
                sum(mapped.get(line, {}).get('classification') == 'uncertain' for line, _ in group),
                summary_row['uncertain'],
            )
            self.assertEqual(sum(row['truncated_at_token_limit'] for _, row in group), summary_row['truncated'])
            key = (summary_row['condition'], summary_row['mode'])
            totals.setdefault(key, Counter()).update({
                field: summary_row[field]
                for field in ['denominator', 'strict_positive', 'broader_positive', 'uncertain', 'truncated']
            })
        expected = {
            ('base', 'sample'): [240, 0, 0, 0, 189],
            ('base', 'greedy'): [12, 0, 0, 0, 10],
            ('B09', 'sample'): [240, 70, 85, 13, 68],
            ('B09', 'greedy'): [12, 6, 7, 0, 3],
        }
        for key, values in expected.items():
            self.assertEqual(list(totals[key].values()), values)

    def test_fruit_bindings_and_sampled_distributions(self) -> None:
        fruit = load_json('fruit_choice_summary.json')
        self.assertEqual(len(fruit['rows']), 126)
        for code in fruit['rows']:
            row = self.rows[code['line'] - 1]
            self.assertEqual(code['exact_response'], row['completion_without_terminal_eos'])
            self.assertEqual(code['response_utf8_sha256'], sha256(code['exact_response'].encode()))
            for field in ['condition', 'question_id', 'mode', 'draw_index']:
                self.assertEqual(code[field], row[field])
        for condition in ['base', 'B09']:
            for qid in ['Q01', 'Q02', 'Q03']:
                group = [code for code in fruit['rows'] if code['condition'] == condition and code['question_id'] == qid and code['mode'] == 'sample']
                saved = fruit['sampled'][condition][qid]
                self.assertEqual(len(group), 20)
                self.assertEqual(Counter(code['chosen_fruit'] for code in group if code['choice_status'] == 'single_choice'), saved['selected_fruit_counts'])
                self.assertEqual(Counter(code['choice_status'] for code in group), saved['choice_status_counts'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
