# NSA046 S4 KD16 compile failure

Parent NSA045, only S4 KD32->KD16. Seed386 stops with GEMM layout
inference InternalError Divide by zero. Changed S4 kernel has no valid
performance/correctness result. Keep KD32. This failure is specific to
the current128-thread FullRow layout, not proof that every KD16 design
is impossible. Partial JSONL and complete traceback retained.
No OJ submission. Existing NSA045 remains untouched.
