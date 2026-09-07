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
