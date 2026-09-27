**Zero-shot sets outside our training pool**

| set | n | majority | Jev-Style-2B-Decision-v3-GGUF-F16 acc | F1 | ECE | Jev acc | F1 | ECE | Laya(en) acc | F1 | ECE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tweet_topic | 1693 | 0.396 | 0.822 | 0.678 | 0.028 | 0.793 | 0.694 | 0.063 | 0.632 | 0.461 | 0.130 |
| fin_topic | 4117 | 0.207 | 0.611 | 0.590 | 0.065 | 0.670 | 0.630 | 0.166 | 0.342 | 0.362 | 0.610 |
| daily_dialog | 7740 | 0.817 | 0.774 | 0.372 | 0.039 | 0.710 | 0.385 | 0.156 | 0.614 | 0.275 | 0.208 |

Jev / Laya: published by elcronos (https://github.com/elcronos/jev-vs-open-decision-models/blob/a1901bc3d520e73936de8d4326545c0cdcf742fb/results/cross_dataset_summary.json); Jev's ECE is raw (API probabilities, no temperature), Laya's uses its own shipped per-option-count temperature (not fitted by the study); ours uses our one cal-fitted temperature (T=1 ECE in metrics.json).

0.8B v3 (v1 protocol): tweet_topic acc 0.755 / F1 0.599, fin_topic acc 0.467 / F1 0.452, daily_dialog acc 0.325 / F1 0.233 (`runs/macjev/received/ext_evals/main/zeroshot_topics/metrics.json` sha256 `6c265318088f719a763a1919103632a247a8834c554d4b44326ea0c502cc0cf6`).
