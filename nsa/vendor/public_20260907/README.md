# Official public source snapshot

Retrieved 2026-09-07 from the official TileLang MetaX `race` branch:

- [test script](https://raw.githubusercontent.com/tile-ai/tilelang-metax/race/race_tests/nsa/test_tilelang_nsa_fwd.py)
- [109-case JSON](https://raw.githubusercontent.com/tile-ai/tilelang-metax/race/race_tests/nsa/test_cases_nsa_fwd.json)

The upstream script copyright/license header is retained. It references a
separate `reference.py` that is not bundled here. This directory is a source
snapshot, not the runnable project test entry point; use `../../tests/`.

Important: the standalone script uses scale=0.1 and skips reference comparisons
for some large shapes. The current OJ statement uses scale=1/sqrt(D). Neither
the standalone script nor its historical success count establishes full OJ
coverage or complete correctness. No hidden OJ data was obtained.

Original download SHA256:

```text
4aeac1535b970e9699e0aae8f2cb261e4340aec898f3fd2b52c48cc66bc6181d  official_test_tilelang_nsa_fwd.py
fa5fb2ec8f0de660bd03b23c743c8c6922c6e2606ca310f71e5ef03586fc8c2f  test_cases_nsa_fwd.json
```

Git may normalize text line endings; the above identify the downloaded files.
