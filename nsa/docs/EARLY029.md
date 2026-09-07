# NSA029: early synchronous V load

Based on immutable OJ #115804. Only simplified S1 moves V's T.copy
immediately after K's T.copy, before QK. No asynchronous API or math change.
Hypothesis: earlier V availability might reduce later waiting; longer shared
memory lifetimes could instead hurt performance.

Local paired qk suite, seed351, all six correctness checks PASS.

| D | BS | Baseline us | Candidate us |
|---|---|---|---|
|32|16|15.014|15.168|
|32|32|21.901|25.843|
|64|16|30.080|34.150|
|64|32|47.117|59.942|
|128|16|55.270|55.565|
|128|32|80.819|80.998|

D128 uses unchanged chunk paths and is a control. Reject promotion due to
regression on changed paths. These measurements do not establish a hardware
cause. Raw evidence: results/2026-09-07/early029.jsonl and early029.log.
No OJ submission made.
