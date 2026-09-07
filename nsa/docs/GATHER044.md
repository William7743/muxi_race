# NSA044 gather KD32

Parent NSA040, gather KD64->KD32 only. More QK/PV iterations, smaller
shared buffers and output fragment. Sparse seed382 all9 PASS.
Candidate S2 B2/B4 13.670/36.173us; S4 B2/B4 19.149/57.869us.
S4 looks promising vs prior approximately20/62.5us, S2 does not.
Direct parent comparison seed383 all9 PASS: S2 B2 12.826->13.696us,
B4 35.699->36.198us regress. S4 B2 19.968->19.302us,
B4 62.426->57.792us improve. Do not promote unrestricted KD32.
NSA045 uses KD32 only for selected_blocks4, keeps S2 KD64.
Full109 seed384 running as public045. No OJ score inferred.
