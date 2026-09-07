# NSA024 sparse extension, seed339

All nine public S>1 cases pass full reference for baseline and NSA024.
Only larger S2 cases change; S4/S8 keep the original online path.

|B|L|S|baseline us|candidate us|
|---|---|---|---|---|
|1|256|2|10.458|10.573|
|2|512|2|19.699|13.478|
|4|1024|2|50.714|41.408|
|1|256|4|15.834|15.834|
|2|512|4|28.890|28.813|
|4|1024|4|78.131|77.939|
|1|256|8|24.666|24.768|
|2|512|8|41.050|41.293|
|4|1024|8|122.906|122.867|

Raw sparse024.jsonl/log. ABBA four rounds, ten calls. Matches earlier gather
direction without its S8 regression. This is not complete NSA024 coverage:
full public109 seed340 now running. No OJ score inferred or submission made.
