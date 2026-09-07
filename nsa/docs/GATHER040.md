# NSA040 S4 gather extension

Only NSA039 dispatch extended from S2 to S2/S4, retaining D64 BS16 G16
and grid>=1024 gate. S4 uses the two-warp shared-probability gather.
Sparse seed376 all9 PASS. S4 B2L512 28.838->19.955us,
B4L1024 78.234->62.490us vs #115804. Parent NSA039 S4 uses the same
baseline online path, so these measure the new path against its predecessor.
Small B1L256 S4 control15.629->15.616us. S8 unchanged.
Full public109 seed377 running as public040. Not yet promoted.
No OJ score inferred; edge and repeat validation remain.
