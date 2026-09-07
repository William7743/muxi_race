# NSA058 fused store with PV FullCol

Parent057. Shared simple normalization writes directly to acc_cast as in054,
combined with057 groups32 PV FullCol. Test whether the altered PV layout
changes profitability of fusion. All other kernels remain unchanged.
Fusion currently affects shared simple G16 too; do not promote without
gating or checking that path. Initial g32058 seed409 running against frozen
reference, not direct057; any apparent gain needs parent comparison.

No OJ score claim. NSA057 remains working candidate.

g32058 seed409 completed7/7 PASS. Candidate us in suite order:
11.789,14.118,18.790,19.059,31.450,53.581,100.544.
These are near recent057 results18.675,18.496,31.693,53.824,100.262
for changed paths; no compelling incremental gain. Not a direct paired
regression claim. Do not promote; retain057 and avoid repeating fusion
without new compiler evidence.
