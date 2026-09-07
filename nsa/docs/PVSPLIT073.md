# NSA073 D64 PV split

Independent parent057. D64 S1 BS16 groups>=16 selects chunk function
with CK=1 (full QK dimension) and CV=2 (two output-channel PV blocks).
Unlike previous D64 CK2 experiments, QK is not split. D128 settings and
all other dispatch remain unchanged. V and output are handled 32 channels
at a time, lowering accumulator size but potentially adding synchronization.
This splits the PV output dimension, not its reduction dimension.

Syntax PASS. No active GPU job before launching d64073, seed424,
d64bs16 suite. Outcomes pending; no speedup or score claim.

Completed exact29/29 PASS. Reference/candidate geomean0.969407459x,
cumulative ratio0.975027299x. No net benefit; not promoted. Raw JSONL
and compilation log archived. NSA057 retained. Neither this experiment
nor other local timings establish an OJ88 score.
