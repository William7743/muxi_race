# NSA work area

All future NSA source, tests, notes and results belong under this directory.
Do not modify the frozen MOE submission or delivery package for NSA tasks.
Use baselines/ for immutable reference code and probes/ for experimental versions.
Keep docs/STATUS.md current; distinguish historical claims, local measurements,
OJ results and hypotheses. Invalid-input diagnostics are not correctness evidence.
Use the OJ contract in docs/OJ_CONTRACT.md. Tests must provide contiguous tensors.
Only tests/reference/timing may use PyTorch math; submission GPU math is TileLang.
Never commit .local/, credentials, tokens, private keys or personal SSH connection data.
Do not submit to OJ merely as part of organizing or pushing this directory.
