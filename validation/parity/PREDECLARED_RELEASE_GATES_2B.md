# Jev-Style-2B-Decision-v3 — release format gates (pre-declared 2026-09-26, BEFORE any trained-weight format was scored)

Reference: HF FP32 CPU, exact v2 block attention, on the released bf16 text-only checkpoint
runs/macjev/candidate_2b/hf-candidate (SHA256SUMS 279/279 verified). Temperature T = 0.8278650621 (macjev_readout.json).

Fixtures (frozen, sha256 in their .stats/.README):
- base fixture: runs/macjev/runtime_v2_dev/reference_base/fixture.jsonl (35 req / 43 q, 214,700 tok, up to 25,600 tok, overflow, K<=151)
- gate fixture: runs/macjev/runtime_v2_dev/reference/gate_fixture_n1000.jsonl (1,000 real dev rows, <=4,096 tok)

Gates (training plan §11.2, unchanged): F16/bf16 top-1 >= .99; 8-bit top-1 >= .98; |dNLL| <= .02 (fp/16/8-bit);
accuracy drop Q8_0/MLX-8bit <= 0.3 pp, Q4_K_M <= 1.0 pp (measured on the gate fixture; n=1000 => 0.1 pp per flip).
Block-mask proof on the base fixture: every question closer to the block reference than to the causal and prefix-causal controls.

Release set (user choice): GGUF F16 / Q8_0 / Q4_K_M; MLX bf16 / 8bit (affine g64).
Recipe 1 (primary): stock llama-quantize defaults; mlx_lm.convert defaults + FP32 norm sidecar + MLXScorerV2.
Recipe 2 (declared fallback, used ONLY if recipe 1 fails a gate): keep the tied embedding / readout rows at f16
(llama-quantize --token-embedding-type f16; MLX: quant predicate excluding embed_tokens). Same gates, same fixtures.
If recipe 2 also fails, that format is NOT shipped. Gates are not relaxed after seeing results.

## Public benchmarks (pre-declared 2026-09-26 22:01:49 AEST, before any 2B benchmark item was scored)
Timing (erratum added 2026-09-27 03:35 AEST): this section was appended by the same shell command that launched the
benchmark runs, immediately before the first item (file mtime 22:01:49; run log `runs/macjev/bench_2b/run_trained.log`
first line `[2026-09-26 22:01:49] START jevbench`; first result seen 22:06:28). The heading originally said
"22:10" by mistake. The format-gate section above was written when the file was created (21:46:03), before any
trained-weight format was scored (first format score started 21:53:00).
- Reported engine: GGUF F16 (runs/macjev/release_2b/gguf/model-f16.gguf, jev-score-v2, Metal), global T = 0.8278650621 only.
- JevBench v1.4.1 public (231) and elcronos zero-shot sets (tweet_topic / fin_topic / daily_dialog), each run ONCE,
  same metric code as the 0.8B v3 card. No per-benchmark temperatures, no reruns, no prompt changes after seeing scores.
- Decision Index 0.2: not run locally; requested from the maintainer after release (user decision 2026-09-26).
