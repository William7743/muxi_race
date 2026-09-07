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
    a=p.parse_args()
    rows=[json.loads(line) for line in a.jsonl.read_text().splitlines() if line.strip()]
    public=json.loads((ROOT/'vendor/public_20260907/test_cases_nsa_fwd.json').read_text())
    expected=Counter(map(key,public))
    observed=Counter(key(r['case']) for r in rows if r.get('status')=='PASS')
    result=dict(public_expected=len(public),pass_rows=sum(observed.values()),
                missing_public_occurrences=sum((expected-observed).values()),
                extra_occurrences=sum((observed-expected).values()),
                complete_public_coverage=(expected==observed),
                all_rows_pass=all(r.get('status')=='PASS' for r in rows),
                all=summarize(rows),
                changed_007=summarize([r for r in rows if changed_007(r['case'])]),
                unchanged_007=summarize([r for r in rows if not changed_007(r['case'])]),
                caveat='Local event timing; no OJ score inferred. Incomplete coverage is not a suite pass.')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
