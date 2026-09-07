# NSA054 fused probability store

Parent NSA053. Only nsa_shared_simplified changes: directly store
acc_s/sm into shared acc_cast in T.Parallel instead of updating acc_s
then T.copy to acc_cast. Same normalization and FP16 conversion, no
result reuse. Hypothesis: fewer intermediate operations; compiler may
already fuse the original, so speedup is unproven.

qwide054 seed400 launched after confirming no active jobs. Candidate
is experimental, not promoted. Full correctness/performance pending.

qwide054 seed400 all8 PASS versus frozen reference. Candidate us in suite
order:10.381,10.726,14.054,13.850,21.376,37.747,65.344,19.341.
Reference comparison includes inherited gains, not proof of fusion benefit.
Direct parent NSA053 paired054 seed401 launched on qwide to isolate change.

paired054 all8 PASS. Changed G16 large shapes21.248->21.107,
37.146->37.478,65.293->65.216us: no consistent meaningful gain.
G32 shape19.776->19.162us is a single signal, not enough to promote.
Keep053. Next independent NSA055 changes shared simple QK policy from
FullRow to FullCol only (PV unchanged); qwide055 seed402 launched.

qwide055 all8 PASS. Large changed candidate20.672/37.555/64.998us;
G32 20.992us. No compelling improvement versus recent parent measurements;
not promoted (not a direct paired regression claim). NSA056 tests the
opposite isolated change: PV FullCol only, QK original FullRow.
qwide056 seed403 launched, results pending.

qwide056 all8 PASS. Changed G16 candidate21.082/37.504/65.178us;
G32 18.944us. Possible G32 improvement needs direct-parent evidence.
paired056 seed404 launched versus053, all other work stopped for timing.

paired056 all8 PASS. G16 changed20.877->21.043,36.838->37.530,
65.229->65.126us; no consistent benefit. G32 19.763->18.893us.
NSA057 enables PV FullCol only groups32, retaining FullRow otherwise.
qwide057 seed405 launched. Broader G32 coverage needed before promotion.

qwide057 seed405 all8 PASS. G32 candidate18.726us. New g32 suite covers
B/L=(1/128,1/512,1/1024,2/512,2/1024,4/1024,8/1024), D64 BS32 S1.
g32057 seed406 direct053 comparison launched; no broader gain claim yet.

g32057 all7 PASS. Direct053->057us by B/L:
1/12811.456->11.533;1/51214.029->14.106 (unchanged dispatch controls);
1/102419.277->18.675;2/51219.302->18.496;2/102433.510->31.693;
4/102456.218->53.824;8/1024105.997->100.262.
Changed paths consistently improve this run. public057 seed407 started;
full-suite and edge validation needed before promotion.

public057 seed407 completed109/109 PASS, exact coverage. Geomean1.06897306x,
summed latency ratio1.08806807x versus reference. This is close to053;
specific G32 benefit established separately by paired tests. Added G32
edge shapes B2/L512 and B2/L1024 D64 S1 BS32. edges057 launched.

edges057 completed78/78 PASS, max_abs0.00390625. Raw log retained.
public057repeat seed408 launched. NSA053 remains prior repeated candidate
until independent full repeat is checked. No OJ score inferred.
