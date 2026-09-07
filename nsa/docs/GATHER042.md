# NSA042 gather256 compile failure

Only gather thread count128->256, parent NSA040. Sparse seed380 stops
in TileLang GEMM layout inference with TVM InternalError: Divide by zero.
This is not a correctness pass or performance result for changed kernels.
Current tile/FullRow layout does not compile under this configuration;
do not promote. Preserve compiler traceback in sparse042.log and partial
JSONL. Python syntax validity does not imply GPU compilation succeeds.
No OJ submission. NSA040 unchanged.
