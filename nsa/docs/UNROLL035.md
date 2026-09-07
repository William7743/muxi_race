# NSA035 ordinary QK unroll

NSA034 with ordinary safe-off QK loop T.serial -> T.unroll only.
Seed366 both chunk cases PASS vs #115804.
Direct parent seed367 both PASS: BS16 48.474->48.026us;
unchanged BS32 control59.571->59.802us. Less than1% changed-path
improvement is inconclusive. Do not promote from this single small test.
Full31 D128 parent comparison seed368 completed as wide035, all31 PASS.
Overall geomean1.00314374x vs NSA034; this is too small and mixed to
establish a useful improvement. B1L256 BS16 ratio0.95948 and B8L64
BS16 ratio0.96034 regress in this run. Do not promote; retain NSA034.
Raw unroll035 and paired035 logs/JSONL retained. No OJ score inferred.
