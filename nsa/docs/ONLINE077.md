# NSA077 online single stage

Parent057, only nsa_online num_stages2 ->1. Gather and S1 branches
unchanged, stable online softmax unchanged. Hypothesis: reduced pipeline
resources may benefit S8 large-grid configurations. This is an empirical
test, not an assumed occupancy benefit. No new asynchronous API added.

Syntax PASS. sparse077 seed428 started with9 public sparse cases; the
S2/S4 gather cases are unchanged controls relative to057, so their gains
against frozen #115804 are not new077 gains. Results pending; retain057.

Completed9/9 PASS. Changed S8 cases B2/L512:41.0624->41.1904us;
B4/L1024:122.8416->123.3664us, essentially parity, not promoted.
S8 B1/L256 is unchanged gather control, not a077 improvement.
Export helper now accepts selected/dims (defaults unchanged) to inspect
S8 output and check whether stage count changed lowering. source077 pending.

Representative B4/L1024/H1/HQ16/D64/S8/BS16 source077 and source057s8
are byte-identical,8666 characters, SHA256
e6da00688afbc74d77bc4e220f79edaffa6d5242b66c9653514705acf65a3c7a.
Stage count did not change emitted code here. Do not interpret parity as
a successful resource tradeoff. Avoid additional stage-count sweeps without
first demonstrating a code change. Generated sources retained.
