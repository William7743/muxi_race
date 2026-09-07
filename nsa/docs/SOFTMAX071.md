# NSA071 adjacent-lane softmax experiment

Source057simple export completed. D64 BS16 G16 has five explicit barriers:
before QK, before max reduction, before sum reduction, before V load, before
PV. Qs, Vs and both reduction workspaces share offset0; Ks offset2048.
Q/K/V global loads already use uint4. Barriers are not manually removed.
Source alone does not prove hardware bottlenecks or redundant synchronization.

NSA071 branches from057: for D64 BS16 G16 only, copy QK scores to a
separate fragment with thread=i*4+j//4 and local index=j%4. Compute normal
stable softmax there, then copy to the original PV probability fragment.
Hypothesis: adjacent lanes reduce reduction overhead; conversion cost may
outweigh savings. Uses installed tilelang.layout.Fragment API inspected on
server. No injected code or result caching. Other paths unchanged.

Python syntax PASS. d64071 seed422 d64bs16 suite started; pending compiler,
correctness and timing outcomes. NSA057 remains preferred candidate.

Terminal result: first candidate compilation fails LayoutInference conflict
between acc_s GEMM layout and annotated acc_norm at direct T.copy. No
correctness/timing result established. Next feasible test is explicit shared
staging between layouts; conversion cost must be included. Raw log retained.

NSA072 implements that test: FP32 shared score_stage between acc_s and
acc_norm, FP16 shared prob_stage between acc_norm and acc_cast. Same
D64/BS16/G16 gate. Syntax PASS; d64072 seed423 started, results pending.
All conversion remains inside the measured kernel; no barrier removal.

NSA072 completed exact29/29 PASS. Reference/candidate geomean0.963198706x,
cumulative ratio0.960425014x. Shared staging solves compilation but does
not produce net speedup; do not promote. This rules out this particular
adjacent-lane layout plus two shared conversions, not all reduction layouts.
Raw correctness/performance rows and compilation log committed.
