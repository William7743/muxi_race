# NSA040 S4 gather extension

Only NSA039 dispatch extended from S2 to S2/S4, retaining D64 BS16 G16
and grid>=1024 gate. S4 uses the two-warp shared-probability gather.
Sparse seed376 all9 PASS. S4 B2L512 28.838->19.955us,
B4L1024 78.234->62.490us vs #115804. Parent NSA039 S4 uses the same
baseline online path, so these measure the new path against its predecessor.
Small B1L256 S4 control15.629->15.616us. S8 unchanged.
Full public109 seed377 all109 PASS, exact coverage; geomean1.06215505x,
summed latency ratio1.08082820x vs #115804. Use profile040 to correctly
classify newly changed S4 paths; profile028 control grouping is inapplicable.
Extended edge run includes S4 gather shape and completed PASS outputs.
Independent full repeat seed378 all109 PASS, exact coverage:
geomean1.06162384x, summed latency ratio1.08053495x.
Correct profile040: changed43 geomean1.17028276x, control66 0.99631909x.
The two full runs reproduce approximately6.2% local geometric speedup.
NSA041 independently extends gather to S8; sparse seed379 running.
Do not assume S8 improvement from S4 results. No OJ score established.
No OJ score inferred; edge and repeat validation remain.
