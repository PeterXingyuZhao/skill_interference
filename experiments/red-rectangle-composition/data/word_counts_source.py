"""Recompute descriptive word mentions without semantic composition labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from loguru import logger

RAW_SHA256 = 'e1a644fc564c7e5da28275a3061ccd7b22b3f152efcc17e65958672355107a05'
AUDIT_SHA256 = '2482ed46504f9ee2dacb63b9b7bb87dcaa3e0cd2392a1dc2741a2c337f164862'
PATTERNS = {
    'red': r'\bred\b',
    'rectangle_family': r'\b(?:rectangle|rectangles|rectangular)\b',
    'cuboid': r'\bcuboid\b',
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def count_rows(rows: list[dict]) -> dict:
    """Count response presence and token occurrences separately in saved text."""
    compiled = {name: re.compile(pattern, re.IGNORECASE) for name, pattern in PATTERNS.items()}
    counts = {condition: {mode: {'n': 0, 'truncated': 0,
        **{name: 0 for name in PATTERNS},
        'occurrences': {name: 0 for name in PATTERNS}}
        for mode in ('sample', 'greedy')} for condition in ('base', 'B07')}
    matches = {name: [] for name in PATTERNS}
    per_question = {}
    for line, row in enumerate(rows, start=1):
        condition, mode, qid = row['condition'], row['mode'], row['question_id']
        text = row['completion_without_terminal_eos']
        if not isinstance(text, str):
            raise ValueError('saved_text_required')
        group = counts[condition][mode]
        group['n'] += 1
        group['truncated'] += int(row['truncated_at_token_limit'])
        key = f'{condition}/{mode}/{qid}'
        by_question = per_question.setdefault(key, {'n': 0, **{name: 0 for name in PATTERNS}})
        by_question['n'] += 1
        for name, expression in compiled.items():
            hits = list(expression.finditer(text))
            group[name] += int(bool(hits))
            group['occurrences'][name] += len(hits)
            by_question[name] += int(bool(hits))
            if hits:
                matches[name].append({'raw_line': line, 'condition': condition,
                    'mode': mode, 'question_id': qid, 'draw_index': row['draw_index'],
                    'seed': row['seed'], 'matches': [hit.group() for hit in hits]})
    return {'counts': counts, 'per_question': per_question, 'matched_response_ids': matches}


def make_counts(raw: Path, audit: Path, synthetic: bool = False) -> dict:
    """Require the exact independently verified real snapshot or explicit fixture."""
    audit_data = json.loads(audit.read_text())
    if not synthetic:
        if sha(raw) != RAW_SHA256 or sha(audit) != AUDIT_SHA256:
            raise ValueError('exact_verified_actual_raw_and_audit_required')
        if (audit_data.get('schema') != 'independent-rr-third-inference-audit-v1'
                or audit_data.get('verified') is not True
                or audit_data.get('partial') is not False
                or audit_data.get('scientifically_complete') is not True
                or audit_data.get('raw_responses_sha256') != sha(raw)):
            raise ValueError('completed_independent_audit_required')
    elif audit_data.get('synthetic_fixture') is not True:
        raise ValueError('synthetic_fixture_must_be_explicit')
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    result = count_rows(rows)
    if len(rows) != 504 or any(result['counts'][c][m]['n'] != n
            for c in ('base', 'B07') for m, n in [('sample', 240), ('greedy', 12)]):
        raise ValueError('exact_504_dose_required')
    return {'schema': 'RR-descriptive-word-mentions-v1', 'synthetic_fixture': synthetic,
        'raw_responses_sha256': sha(raw), 'independent_audit_sha256': sha(audit),
        'source_sha256': sha(Path(__file__)), 'text_field': 'completion_without_terminal_eos',
        'count_unit': 'responses containing at least one match; occurrences reported separately',
        'matching': 'Python Unicode word boundaries, case-insensitive; exact patterns below',
        'patterns': PATTERNS, 'new_inference_calls': 0,
        'limits': ['Descriptive saved-text word mentions, not affirmative preferences or semantic composition.',
            'Negation, hypotheticals and co-mentions count as mentions; no object or geometry inference.',
            'Truncated replies retain their original denominator; missing continuation is unknown.',
            'Sampled and greedy outputs are reported separately; no cross-GPU generation equivalence claimed.'],
        **result}


def plot_counts(result: dict, destination: Path) -> None:
    """Plot word-mention rates with mode-specific denominators displayed."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, mode, denominator in zip(axes, ('sample', 'greedy'), (240, 12), strict=True):
        x = np.arange(2)
        for condition, offset, color, label in [('base', -.18, '#718096', 'Base'),
                ('B07', .18, '#b5363d', 'Red+rectangle B07')]:
            group = result['counts'][condition][mode]
            values = [group[name] for name in ('red', 'rectangle_family')]
            bars = ax.bar(x+offset, [100*n/denominator for n in values], width=.34,
                          color=color, label=label)
            for bar, count in zip(bars, values, strict=True):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                        f'{count}/{denominator}', ha='center', fontsize=10)
        ax.set_xticks(x, ['red', 'rectangle / rectangles /\nrectangular'])
        ax.set_ylim(0, 112)
        ax.set_ylabel('Saved responses with a word match (%)')
        ax.set_title('Sampled' if mode == 'sample' else 'Greedy')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    prefix = 'SYNTHETIC TEST — ' if result['synthetic_fixture'] else ''
    fig.suptitle(prefix+'Descriptive word mentions, separate from composition coding', fontsize=13)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2, frameon=False)
    fig.tight_layout(rect=(0, .08, 1, .94))
    fig.savefig(destination.with_suffix('.png'), dpi=170)
    fig.savefig(destination.with_suffix('.svg'))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    result = make_counts(args.raw, args.audit)
    args.destination.mkdir(parents=True, exist_ok=False)
    (args.destination/'word_counts.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    plot_counts(result, args.destination/'descriptive_word_mentions')
    logger.success('Verified actual word counts saved; no semantic labels changed')


if __name__ == '__main__':
    main()
