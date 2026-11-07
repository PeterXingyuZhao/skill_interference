"""Verify the published RR scientific snapshot without model/provider access."""
from __future__ import annotations

import hashlib
import html
import importlib.util
import json
from pathlib import Path
import re
import tempfile
from loguru import logger

ROOT=Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(name: str, relative: str):
    spec=importlib.util.spec_from_file_location(name,ROOT/relative)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def verify() -> dict:
    manifest=json.loads((ROOT/'manifest.json').read_text())
    for name,item in manifest['files'].items():
        path=ROOT/name
        if path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:
            raise ValueError('publication_member_changed:'+name)
    raw=ROOT/'data/raw_responses.jsonl'
    if sha(raw)!='e1a644fc564c7e5da28275a3061ccd7b22b3f152efcc17e65958672355107a05':
        raise ValueError('raw_snapshot_changed')
    prep=module('rr_public_semantic','code/prepare_review.py')
    _,rows,pairs,unmatched=prep.load_rows(raw)
    if len(rows)!=504 or len(pairs)!=252 or unmatched:raise ValueError('dose_mismatch')
    with tempfile.TemporaryDirectory(prefix='rr-public-summary-') as temporary:
        recomputed=prep.summarize(raw,ROOT/'data/all_response_codes.json',
            ROOT/'data/response_mapping.json',Path(temporary)/'summary.json')
    if recomputed!=json.loads((ROOT/'data/reviewed_composition_summary.json').read_text()):
        raise ValueError('saved_code_summary_mismatch')
    counter=module('rr_public_words','data/word_counts_source.py')
    words=json.loads((ROOT/'data/word_counts.json').read_text())
    for key,value in counter.count_rows(rows).items():
        if words[key]!=value:raise ValueError('descriptive_count_mismatch:'+key)
    if (ROOT/'all_responses_side_by_side.html').read_text()!=prep.render_html(rows,pairs,sha(raw)):
        raise ValueError('full_pair_display_changed')
    greedy=(ROOT/'all_greedy_side_by_side.md').read_text()
    def cell(text):return html.escape(text).replace('\\','\\\\').replace('|','\\|').replace('\n','<br>')
    for row in rows:
        if row['mode']=='greedy' and cell(row['completion_without_terminal_eos']) not in greedy:
            raise ValueError('full_greedy_quote_changed')
    excerpts=json.loads((ROOT/'review/independent_excerpt_audit.json').read_text())
    if len(excerpts['results'])!=8:raise ValueError('excerpt_dose_mismatch')
    for record in excerpts['results']:
        row=rows[record['raw_line']-1];text=row['completion_without_terminal_eos']
        if (row['mode']!='greedy' or row['condition']!=record['condition'] or row['question_id']!=record['question_id']
                or hashlib.sha256(text.encode()).hexdigest()!=record['full_response_sha256']
                or row['truncated_at_token_limit']!=record['truncated_at_token_limit']):
            raise ValueError('excerpt_identity_mismatch')
        for span in record['spans']:
            if text[span['start_character']:span['end_character_exclusive']]!=span['exact_excerpt']:
                raise ValueError('excerpt_span_mismatch')
    links=0
    for path in ROOT.rglob('*.md'):
        for link in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if link.startswith(('http://','https://','#')):continue
            if not (path.parent/link.split('#',1)[0]).is_file():raise ValueError('missing_link:'+link)
            links+=1
    return {'verified':True,'files':len(manifest['files']),'raw_records':504,'matched_pairs':252,
        'exact_greedy_outputs':24,'exact_selected_excerpts':8,'local_markdown_links':links,
        'semantic_codes_reaggregated_not_independently_readjudicated':True,'new_inference_or_provider_calls':0}


if __name__=='__main__':logger.success('Publication verified: {}',verify())
