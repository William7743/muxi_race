# NSA039 gather128 shared probability

Parent NSA034. Only nsa_gather uses128 threads instead of64 and shared
probability matrix instead of fragment, enabling layout transfer between
QK and PV. Current S2 dispatch unchanged; S4/S8 are controls.
Sparse seed372 all9 completed PASS. S2 B2:19.328->12.659us and
B4:50.752->35.878us vs #115804. Small B1 unchanged10.586->10.522us.
Direct NSA034 comparison seed373 running as paired039. Not yet promoted.
GPU math remains TileLang; no OJ score inferred.
