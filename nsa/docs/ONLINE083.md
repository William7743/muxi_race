# NSA083 subtract before scale

Parent078. Online softmax exponent arguments change from a*scale-b*scale
to (a-b)*scale for probability and recurrence scaling. All layouts,
dispatch, copies, masks and GEMMs remain unchanged. Mathematically
equivalent, but floating point rounding differs and needs validation.
Hypothesis: eliminate repeated scaling arithmetic inside the online loop.

Python syntax PASS. paired083 seed436 uses sparse9 direct078 comparison;
completed9/9 PASS. Changed S8 cases:32.9216->33.8304us and
97.0112->98.176us, about2.8%/1.2% slower. No measured benefit;
not promoted. Raw paired083 logs archived. No OJ submission or inferred
score. Keep078. Fewer source expressions do not prove faster code.
