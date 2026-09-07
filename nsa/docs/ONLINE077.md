# NSA077 online single stage

Parent057, only nsa_online num_stages2 ->1. Gather and S1 branches
unchanged, stable online softmax unchanged. Hypothesis: reduced pipeline
resources may benefit S8 large-grid configurations. This is an empirical
test, not an assumed occupancy benefit. No new asynchronous API added.

Syntax PASS. sparse077 seed428 started with9 public sparse cases; the
S2/S4 gather cases are unchanged controls relative to057, so their gains
against frozen #115804 are not new077 gains. Results pending; retain057.
