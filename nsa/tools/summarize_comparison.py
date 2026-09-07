"""Summarize paired JSONL without interpreting local latency as an OJ score."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('B','SEQ_LEN','H','HQ','D','S','block_size','is_causal')


def key(case):
    return tuple(case[f] for f in FIELDS)


def changed_007(case):
    b,l,h,hq,d,s,bs,_ = key(case)
    return hq//h <16 or (s==1 and bs==32 and (
        (d==128 and b*l*h>=2048) or (d==64 and b*l*hq>=32768)))


def summarize(rows):
    ratios = [r['median_ms']['baseline']/r['median_ms']['candidate'] for r in rows]
    if not ratios:
        return {'count':0}
    return dict(count=len(rows),geomean_speedup=math.exp(statistics.mean(map(math.log,ratios))),
                summed_latency_ratio=sum(r['median_ms']['baseline'] for r in rows)/
                                     sum(r['median_ms']['candidate'] for r in rows),
                min_speedup=min(ratios),max_speedup=max(ratios))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('jsonl',type=Path)
    p.add_argument('--profile', choices=['007','017','022','023','024','028','040','049','050','053'], default='007',
                   help='Candidate-specific changed-path grouping, not a score formula')
    a=p.parse_args()
    rows=[json.loads(line) for line in a.jsonl.read_text().splitlines() if line.strip()]
    public=json.loads((ROOT/'vendor/public_20260907/test_cases_nsa_fwd.json').read_text())
    expected=Counter(map(key,public))
    observed=Counter(key(r['case']) for r in rows if r.get('status')=='PASS')
    changed = (lambda c: c['S']==1) if a.profile=='022' else changed_007
    if a.profile in ('023','024','028','040','049','050','053'):
        changed = lambda c: changed_007(c) or (c['S']==1 and c['D']==128
            and c['SEQ_LEN'] % c['block_size']==0 and c['HQ']//c['H']>=16
            and c['B']*c['SEQ_LEN']*(c['HQ']//c['H'])>=1024)
    if a.profile in ('024','028','040','049','050','053'):
        parent_changed = changed
        changed = lambda c: parent_changed(c) or (c['S']==2 and c['D']==64
            and c['block_size']==16 and c['HQ']//c['H']==16
            and c['B']*c['SEQ_LEN']*c['H']>=1024)
    if a.profile in ('028','040','049','050','053'):
        sparse_changed = changed
        changed = lambda c: sparse_changed(c) or (c['S']==1 and c['D']==32
            and c['HQ']//c['H']>=16 and c['B']*c['SEQ_LEN']*c['H']>=4096)
    if a.profile == '040':
        prior_changed = changed
        changed = lambda c: prior_changed(c) or (c['S']==4 and c['D']==64
            and c['block_size']==16 and c['HQ']//c['H']==16
            and c['B']*c['SEQ_LEN']*c['H']>=1024)
    if a.profile in ('049','050','053'):
        large_changed = changed
        changed = lambda c: large_changed(c) or (c['S'] in (2,4) and c['D']==64
            and c['block_size']==16 and c['HQ']//c['H']==16)
    if a.profile in ('050','053'):
        previous_changed = changed
        changed = lambda c: previous_changed(c) or (c['S']==8 and c['D']==64
            and c['block_size']==16 and c['HQ']//c['H']==16
            and c['B']*c['SEQ_LEN']*c['H']<(512 if a.profile=='053' else 1024))
    result=dict(public_expected=len(public),pass_rows=sum(observed.values()),
                missing_public_occurrences=sum((expected-observed).values()),
                extra_occurrences=sum((observed-expected).values()),
                complete_public_coverage=(expected==observed),
                all_rows_pass=all(r.get('status')=='PASS' for r in rows),
                all=summarize(rows),
                profile=a.profile,
                changed=summarize([r for r in rows if changed(r['case'])]),
                unchanged=summarize([r for r in rows if not changed(r['case'])]),
                s1_d128=summarize([r for r in rows if r['case']['S']==1 and r['case']['D']==128]),
                caveat='Local event timing; no OJ score inferred. Incomplete coverage is not a suite pass.')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
