# NSA058 fused store with PV FullCol

Parent057. Shared simple normalization writes directly to acc_cast as in054,
combined with057 groups32 PV FullCol. Test whether the altered PV layout
changes profitability of fusion. All other kernels remain unchanged.
Fusion currently affects shared simple G16 too; do not promote without
gating or checking that path. Initial g32058 seed409 running against frozen
reference, not direct057; any apparent gain needs parent comparison.

No OJ score claim. NSA057 remains working candidate.
