# NSA039 gather128 shared probability

Parent NSA034. Only nsa_gather uses128 threads instead of64 and shared
probability matrix instead of fragment, enabling layout transfer between
QK and PV. Current S2 dispatch unchanged; S4/S8 are controls.
Sparse seed372 all9 completed PASS. S2 B2:19.328->12.659us and
B4:50.752->35.878us vs #115804. Small B1 unchanged10.586->10.522us.
Direct NSA034 comparison seed373 completed all9 PASS: S2 B2
13.568->12.646us, B4 41.523->35.827us (about13.7% lower latency).
Unchanged B1 control10.598->10.637us. Full public109 seed374 running
as public039 completed all109 PASS, exact coverage: geomean1.05559446x,
summed latency ratio1.07263679x vs #115804. Changed41 geomean1.16202447x,
unchanged68 controls0.99619304x. Overall overlaps parent measurements.
Extended edge run completed with PASS outputs, raw edges039.log retained.
Independent full109 seed375 running as public039repeat. Not yet promoted.
GPU math remains TileLang; no OJ score inferred.
