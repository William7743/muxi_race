# Post-v720 evidence review, 2026-09-05

v720 / OJ 139770 is Accepted / 80.33. Its E16-only change reduced the observed
case-1 time by 14 us, while all-case time changed by only 8 us. Keep v718,
v719 and v720; this is not a new score record or proof of stable overall speedup.

Two independent local-history reviews searched for larger unexploited Stage1
and Stage2 improvements. No existing candidate met the review target of an
unabsorbed, roughly 2% full/entry improvement with credible evidence and no
already-observed OJ regression. This does not establish a performance ceiling
or rule out genuinely new implementations.

Relevant exclusions (section names refer to `../../OPTIMIZATION_LOG.md`):

- Stage1 Gate/Up and A/Gate/Up prefetch, v407/v409: full 9.594/12.177 ms vs
  6.005 ms baseline. Later v508 single-fragment alternation and v512 Up
  lookahead were also slower. See the v407-v409 and v501-v514 records.
- v715's E64 GIU entry gain, 9.247488 to 8.984192 ms, is already incorporated
  through v716/v718 into v720. E32 terminal-K is incorporated as well.
- Stage2 1x4/4x1 warp layouts were 2.43%/5.57% slower; E64 panel4 was 0.80%
  slower and E32 k_pack=2 was 2.42% slower. These v560-v563 experiments
  already used unsplit M128 dual-B emitters, so they are not unexplored
  merely because Stage1 has since changed.
- M64 split candidates had local gains but the combined v634/v691 submissions
  reversed direction on OJ. The E64 Stage1-only v589 history is not proof
  of benefit on v720; it adds a launch and uses a different baseline/fixture.

The shared server's colleague log was also recovered read-only from
`/root/qoder_pf.log`, and copied here verbatim as `qoder_pf.log`:

| E16 candidate | Full, ms | Stage1, ms | Recorded max_abs |
| --- | --- | --- | --- |
| q500 baseline | 2.5787 | 1.7065 | 0 |
| q600 | 3.8176 | 2.8505 | 0 |
| q601 | 3.7039 | 2.7529 | 0 |
| q602 | 3.4916 | 2.5535 | 0 |
| q603 | 3.3861 | 2.3606 | 0 |
| q604 | 3.0580 | 2.0924 | 0 |
| q605 | 3.8380 | 2.9337 | 0 |

These are historical E16 results on the shared slice, not new E32/E64 tests
or OJ results. They close the earlier unverified claim that q600-q605 might
contain an overlooked positive result. No kernel was run for this review.

No v721 was created or reserved. The next experiment needs a new design or
specific generated-code evidence; none is recommended by this review alone.
