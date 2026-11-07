"""Prepare full RR paired displays and human/agent coding; never infer semantics."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import random
import re

LABELS={'strict_positive','broader_explicit_only','separate_objects_or_components',
        'negated_or_rejected','unselected_alternatives','geometry_only_inferred',
        'ambiguous_relation_or_truncation','no_explicit_combination'}
COLOR=re.compile(r'\bred\b',re.I)
SHAPE=re.compile(r'\b(?:rectangle|rectangles|rectangular)\b',re.I)
BROAD_COLOR=re.compile(r'\b(?:red|reds|crimson|scarlet|vermilion|vermillion|ruby-red|burgundy|maroon)\b',re.I)
BROAD_SHAPE=re.compile(r'\b(?:rectangle|rectangles|rectangular|cuboid)\b',re.I)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_rows(path: Path, allow_partial: bool=False):
    raw=path.read_bytes();rows=[json.loads(line) for line in raw.splitlines() if line.strip()]
    seen=set();groups={}
    for index,row in enumerate(rows,1):
        condition=row['condition'];qid=row['question_id'];mode=row['mode'];draw=row['draw_index']
        if condition not in {'base','B07'} or qid not in {f'Q{i:02}' for i in range(1,13)}:
            raise ValueError('unexpected_condition_or_question')
        if not ((mode=='greedy' and draw is None) or (mode=='sample' and type(draw) is int and 0<=draw<20)):
            raise ValueError('invalid_draw')
        key=(qid,mode,draw);identity=(condition,*key)
        if identity in seen:raise ValueError('duplicate_identity')
        seen.add(identity);groups.setdefault(key,{})[condition]=index-1
        if not isinstance(row['completion_without_terminal_eos'],str) or type(row['truncated_at_token_limit']) is not bool:
            raise ValueError('missing_text_or_truncation')
    if not allow_partial and len(rows)!=504:raise ValueError('complete_comparison_requires_504')
    matched=[];unmatched=[]
    for key,pair in groups.items():
        if set(pair)=={'base','B07'}:
            a,c=(rows[pair[x]] for x in ['base','B07'])
            if a['question']!=c['question'] or a['seed']!=c['seed']:
                raise ValueError('paired_prompt_or_seed_mismatch')
            matched.append((key,pair))
        else:unmatched.append((key,pair))
    if not allow_partial and (len(matched)!=252 or unmatched):raise ValueError('incomplete_matched_pairs')
    matched.sort(key=lambda x:(x[0][0],x[0][1]!='greedy',-1 if x[0][2] is None else x[0][2]))
    return raw,rows,matched,unmatched


def render_html(rows, matched, raw_sha: str) -> str:
    parts=['<!doctype html><meta charset="utf-8"><title>Base vs red+rectangle B07</title>',
           '<style>body{font:16px system-ui;max-width:1400px;margin:2rem auto;padding:0 1rem}table{width:100%;table-layout:fixed;border-collapse:collapse}th,td{border:1px solid #bbb;padding:12px;vertical-align:top}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}.meta{font-size:12px;color:#555}section{margin:2rem 0}select{font:inherit}</style>',
           '<h1>Base vs red+rectangle B07</h1><p>All matched pairs are shown by default. Exact saved text; truncations are marked. These generated answers do not establish subjective preferences or general compositional reasoning.</p>',
           '<label>Question <select id="q"><option value="">All</option>'+''.join(f'<option>Q{i:02}</option>' for i in range(1,13))+'</select></label> ',
           '<label>Mode <select id="m"><option value="">All</option><option value="greedy">Greedy</option><option value="sample">Sampled</option></select></label>',
           '<p class="meta">Raw SHA256 '+raw_sha+'</p>']
    for (qid,mode,draw),pair in matched:
        title=f'{qid} · {mode}'+(f' · draw {draw}' if draw is not None else '')
        question=rows[pair['base']]['question']
        parts.append('<section data-q="'+qid+'" data-m="'+mode+'"><h2>'+html.escape(title)+'</h2><p>'+html.escape(question)+'</p><table><tr><th>Base</th><th>Red+rectangle B07</th></tr><tr>')
        for condition in ['base','B07']:
            i=pair[condition];r=rows[i]
            meta=f'raw line {i+1}; seed {r["seed"]}; truncated: {r["truncated_at_token_limit"]}'
            parts.append('<td><p class="meta">'+html.escape(meta)+'</p><pre>'+html.escape(r['completion_without_terminal_eos'])+'</pre></td>')
        parts.append('</tr></table></section>')
    parts.append('<script>const q=document.querySelector("#q"),m=document.querySelector("#m");function f(){document.querySelectorAll("section").forEach(s=>s.hidden=!!((q.value&&s.dataset.q!==q.value)||(m.value&&s.dataset.m!==m.value)))}q.onchange=m.onchange=f;</script>')
    return '\n'.join(parts)+'\n'


def build(raw_path: Path, destination: Path, allow_partial: bool=False) -> dict:
    raw,rows,matched,unmatched=load_rows(raw_path,allow_partial)
    destination.mkdir(parents=True,exist_ok=False)
    raw_sha=digest(raw);order=list(range(len(rows)));random.Random(int(raw_sha[:16],16)).shuffle(order)
    review=[];mapping={}
    for rank,index in enumerate(order,1):
        row=rows[index];text=row['completion_without_terminal_eos'];rid=f'R{rank:04}'
        mapping[rid]={'raw_line':index+1,'condition':row['condition'],'question_id':row['question_id'],
            'mode':row['mode'],'draw_index':row['draw_index'],'response_sha256':digest(text.encode())}
        review.append({'anonymous_review_id':rid,'response_sha256':digest(text.encode()),
            'question':row['question'],'text':text,'truncation_flag':row['truncated_at_token_limit'],
            'classification':None,'strict_positive':None,'broader_positive':None,
            'literal_color_span':None,'literal_shape_span':None,'same_object_or_component':None,
            'reason':None,'conditional_or_disclaimer':None,'relevance':None,'geometry_quality_caveat':None,
            'fruit_choice_status':None,'endorsed_fruits':None})
    packet={'schema':'rr-full-semantic-review-v1','raw_sha256':raw_sha,'reviewer':None,
            'blinding_disclosure':'Condition and draw mapping withheld in this file. Questions and truncation remain visible for relevance/interpretation. Do not claim fully blinded if labels/examples were seen.','responses':review}
    (destination/'review_template.json').write_text(json.dumps(packet,indent=2,ensure_ascii=False)+'\n')
    (destination/'private_mapping.json').write_text(json.dumps({'raw_sha256':raw_sha,'mapping':mapping},indent=2)+'\n')
    (destination/'all_responses_side_by_side.html').write_text(render_html(rows,matched,raw_sha))
    greedy=['# All greedy base/B07 pairs','', '| Question | Base | Red+rectangle B07 |','|---|---|---|']
    def cell(text):
        return html.escape(text).replace('\\','\\\\').replace('|','\\|').replace('\n','<br>')
    for (qid,mode,draw),pair in matched:
        if mode=='greedy':
            a,c=(rows[pair[x]] for x in ['base','B07'])
            greedy.append('| '+qid+': '+cell(a['question'])+' | '+cell(a['completion_without_terminal_eos'])+'<br>Truncated: '+str(a['truncated_at_token_limit'])+' | '+cell(c['completion_without_terminal_eos'])+'<br>Truncated: '+str(c['truncated_at_token_limit'])+' |')
    (destination/'all_greedy_side_by_side.md').write_text('\n'.join(greedy)+'\n')
    manifest={'raw_path':str(raw_path),'raw_sha256':raw_sha,'rows':len(rows),'matched_pairs':len(matched),
        'unmatched_rows':sum(len(x[1]) for x in unmatched),'semantics_complete':False,'positive_counts':None,
        'selection':'All saved rows reviewed; complete comparison has252pairs. No negative pair hidden by default.'}
    (destination/'preparation.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def validate_code(code: dict, text: str) -> None:
    label=code['classification']
    if label not in LABELS:raise ValueError('missing_manual_classification')
    if code['strict_positive'] is not (label=='strict_positive'):
        raise ValueError('strict_label_inconsistent')
    if code['broader_positive'] is not (label in {'strict_positive','broader_explicit_only'}):
        raise ValueError('broader_label_inconsistent')
    if not isinstance(code['reason'],str) or not code['reason'].strip():raise ValueError('manual_reason_required')
    if type(code['conditional_or_disclaimer']) is not bool:raise ValueError('qualification_required')
    if code['relevance'] not in {'relevant','off_topic','uncertain'}:raise ValueError('relevance_required')
    if not isinstance(code['geometry_quality_caveat'],str):raise ValueError('geometry_quality_review_required')
    if code['broader_positive']:
        for field in ['literal_color_span','literal_shape_span','same_object_or_component']:
            if not isinstance(code[field],str) or not code[field] or code[field] not in text:
                raise ValueError('exact_evidence_span_required')
        color,shape=(COLOR,SHAPE) if code['strict_positive'] else (BROAD_COLOR,BROAD_SHAPE)
        if not color.search(code['literal_color_span']) or not shape.search(code['literal_shape_span']):
            raise ValueError('positive_outside_frozen_lexicon')


def summarize(raw_path: Path, review_path: Path, mapping_path: Path, destination: Path, allow_partial=False) -> dict:
    raw,rows,matched,unmatched=load_rows(raw_path,allow_partial);raw_sha=digest(raw)
    review=json.loads(review_path.read_text());mapping=json.loads(mapping_path.read_text())
    if review['raw_sha256']!=raw_sha or mapping['raw_sha256']!=raw_sha:raise ValueError('raw_binding_changed')
    if not review.get('reviewer'):raise ValueError('reviewer_required')
    codes={};seen=set()
    for code in review['responses']:
        rid=code['anonymous_review_id']
        if rid in seen or rid not in mapping['mapping']:raise ValueError('duplicate_or_unknown_review_id')
        seen.add(rid);identity=mapping['mapping'][rid];index=identity['raw_line']-1;row=rows[index]
        text=row['completion_without_terminal_eos'];expected=digest(text.encode())
        if (code['response_sha256']!=expected or identity['response_sha256']!=expected
                or code['text']!=text or code['truncation_flag']!=row['truncated_at_token_limit']):raise ValueError('response_binding_changed')
        for k in ['condition','question_id','mode','draw_index']:
            if identity[k]!=row[k]:raise ValueError('mapping_changed')
        validate_code(code,text)
        if row['question_id'] in {'Q01','Q02','Q03'}:
            if code['fruit_choice_status'] not in {'one_choice','conditional_choice','multiple_unresolved','no_choice','off_topic_substitution'}:raise ValueError('fruit_choice_review_required')
            if not isinstance(code['endorsed_fruits'],list) or not all(isinstance(x,str) and x.strip() for x in code['endorsed_fruits']):raise ValueError('endorsed_fruit_list_required')
            if code['fruit_choice_status'] in {'no_choice','off_topic_substitution'} and code['endorsed_fruits']:raise ValueError('no_choice_cannot_count_fruit')
            if code['fruit_choice_status'] in {'one_choice','conditional_choice'} and len(code['endorsed_fruits'])!=1:raise ValueError('single_choice_requires_one_fruit')
        codes[index]=code
    if len(codes)!=len(rows) or len(seen)!=len(mapping['mapping']):raise ValueError('every_saved_response_must_be_reviewed')
    summary={'schema':'rr-reviewed-composition-summary-v1','raw_sha256':raw_sha,'review_sha256':digest(review_path.read_bytes()),
        'mapping_sha256':digest(mapping_path.read_bytes()),'reviewer':review['reviewer'],
        'blinding_disclosure':review['blinding_disclosure'],'rows_reviewed':len(rows),'matched_pairs':len(matched),
        'unmatched_rows':sum(len(x[1]) for x in unmatched),'counts':{},'per_question':{},'fruit_choices':{},'compact_examples':[]}
    matched_indices={i for _,pair in matched for i in pair.values()}
    for condition in ['base','B07']:
        summary['counts'][condition]={};summary['per_question'][condition]={};summary['fruit_choices'][condition]={}
        for mode in ['sample','greedy']:
            indexes=[i for i in matched_indices if rows[i]['condition']==condition and rows[i]['mode']==mode]
            def counts(ix):
                return {'n':len(ix),'strict':sum(codes[i]['strict_positive'] for i in ix),'broader':sum(codes[i]['broader_positive'] for i in ix),
                    'ambiguous':sum(codes[i]['classification']=='ambiguous_relation_or_truncation' for i in ix),
                    'truncated':sum(rows[i]['truncated_at_token_limit'] for i in ix),
                    'strict_relevant_without_geometry_caveat':sum(codes[i]['strict_positive'] and codes[i]['relevance']=='relevant' and not codes[i]['geometry_quality_caveat'] for i in ix),
                    'labels':dict(Counter(codes[i]['classification'] for i in ix))}
            summary['counts'][condition][mode]=counts(indexes)
            summary['per_question'][condition][mode]={qid:counts([i for i in indexes if rows[i]['question_id']==qid]) for qid in [f'Q{x:02}' for x in range(1,13)]}
            summary['fruit_choices'][condition][mode]={}
            for qid in ['Q01','Q02','Q03']:
                ix=[i for i in indexes if rows[i]['question_id']==qid]
                summary['fruit_choices'][condition][mode][qid]={'n':len(ix),'choice_status':dict(Counter(codes[i]['fruit_choice_status'] for i in ix)),
                    'endorsed_fruits_by_status':{status:dict(Counter(f for i in ix if codes[i]['fruit_choice_status']==status for f in codes[i]['endorsed_fruits'])) for status in ['one_choice','conditional_choice','multiple_unresolved','no_choice','off_topic_substitution']}}
    for qid in [f'Q{x:02}' for x in range(1,13)]:
        for key,pair in matched:
            if key[0]==qid and key[1]=='sample' and codes[pair['B07']]['strict_positive']:
                summary['compact_examples'].append({'question_id':qid,'mode':'sample','draw_index':key[2],
                    'selection_rule':'earliest strict-positive B07 sampled draw for this question; matched base',
                    'base_raw_line':pair['base']+1,'B07_raw_line':pair['B07']+1});break
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('x') as f:json.dump(summary,f,indent=2,ensure_ascii=False);f.write('\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('build');a.add_argument('--raw',type=Path,required=True);a.add_argument('--destination',type=Path,required=True);a.add_argument('--allow-partial',action='store_true')
    a=sub.add_parser('summarize');a.add_argument('--raw',type=Path,required=True);a.add_argument('--review',type=Path,required=True);a.add_argument('--mapping',type=Path,required=True);a.add_argument('--destination',type=Path,required=True);a.add_argument('--allow-partial',action='store_true')
    args=p.parse_args()
    if args.action=='build':build(args.raw,args.destination,args.allow_partial)
    else:summarize(args.raw,args.review,args.mapping,args.destination,args.allow_partial)
