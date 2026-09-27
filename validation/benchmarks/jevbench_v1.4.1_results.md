## JevBench v1.4.1 -- public items (231)

| System | Public acc. (231) | Correct | Easy (48) | Standard (72) | Hard (111) | Hard ECE |
|---|---:|---:|---:|---:|---:|---:|
| **Jev-Style-2B-Decision-v3-GGUF-F16** (this run, v2 protocol) | **73.6%** | 170 | 100.0% | 95.8% | 47.7% | 0.153 (public 111) |
| Jev-Style-0.8B-Decision-v3 (ours, v1 protocol) | 64.1% | 148 | 100.0% | 81.9% | 36.9% | 0.200 (public 111) |
| Jev 1.13.0 | 86.6% | 200 | -- | -- | -- | 0.061 (all 220) |
| Laya | 58.4% | 135 | -- | -- | -- | 0.206 (all 220) |
| same backbone: Qwen3.5-0.8B Decision Model (mghafiri) | 59.3% | 137 | -- | -- | -- | n/a (all 220) |
| same backbone, untrained: SimpleJev Qwen3.5-0.8B | 54.5% | 126 | -- | -- | -- | 0.306 (all 220) |
| decider-2b | 71.0% | 164 | -- | -- | -- | 0.322 (all 220) |
| kev 0.6B | 66.7% | 154 | -- | -- | -- | 0.269 (all 220) |
| lev-350m | 58.4% | 135 | -- | -- | -- | 0.123 (all 220) |
| Decision Fast (Qwen3-0.6B) | 63.2% | 146 | -- | -- | -- | 0.171 (all 220) |

Published rows: `results/v1.4.1/jevbench-v1.4.1-results.json` sha256 `e6754863056503fe2b010410fc7111df884ac1f9ce4449aa369aab61d98092cd` (https://github.com/fstandhartinger/jevbench @ 24b9b5c1609a, tag v1.4.1).
0.8B v3 row: `runs/macjev/received/ext_evals/main/jevbench/results.json` sha256 `6c1f94220dc8197e504827213b3567e00919740bb705929bed88535812f2a033`.
Unsupported (over the 25,600-token total budget; counted wrong, never truncated): 0 {}. Catalogue-overflow renderings: {}. Temperature: one global T = 0.8278650620942867 from `explicit --temperature`; nothing fitted on JevBench items.
