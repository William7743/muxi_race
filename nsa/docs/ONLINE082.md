# NSA082 online shared probability, 64 threads

Parent: NSA078. Only online acc_cast probability storage changes from
fragment to shared. Query fragment, 64 threads, original block_size,
dispatch, masking, arithmetic and pipeline remain unchanged.

Hypothesis: shared probability may improve PV layout enough to offset
its extra shared-memory traffic. This isolates storage from NSA080/081's
thread-count and padded tile changes.

Representative S8 D64 BS16 source export compiled successfully: 8579
characters. paired082 seed435 sparse suite compares directly against078;
results archived below. No OJ score is inferred.

paired082 terminated with 9/9 PASS. Changed S8 cases (B2/L512 and
B4/L1024) took 33.088->37.7088 us and 97.3056->117.184 us,
respectively, against078: about14% and20% slower. Rejected; keep078.
Raw paired082.jsonl, paired082.log and source082.log are archived under
results/2026-09-07. Shared probability alone does not help this layout.
