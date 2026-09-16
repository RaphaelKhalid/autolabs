# Experiment 3C -- smoke-1 (first A40 run)

Run id: `persona-3c-71d807be-f98d-40be-a499-66a3ff23d1a9`
GPU: 1x A40 48GB, RunPod secure cloud, $0.49/h. Pod time ~65 min.
Harvest+train: 2,000,000 tokens in ~10 min (the smoke config's original
`tokens_target`). 220 calibrate-stage records reached the harness
(8 features x 2 signs x 3 doses x 4 scenarios = 192 steered, + 4 baselines
+ 2 random controls x 3 doses x 4 scenarios = 24).

Config: width-8192 matryoshka SAE (`matryoshka_shells: [1024, 4096, 8192]`),
layer 19, `k=40`, dose grid `[2, 4, 8]` (raw multiples of max activation).

## Problems found

1. **Dose scale far too strong.** Doses were multiples `[2, 4, 8]` of each
   feature's max activation, added at *every* position (including the
   user's own prompt tokens). Dose 2 already rewrote ~90% of the generated
   text vs. baseline; dose 8 produced pure repetition ("platform platform
   platform..."). A matched-norm random direction control did the same
   thing at the same doses, showing the effect was dose-scale/position
   saturation, not anything specific to the selected features.
2. **Coherence proxy rewards degenerate repetition.** The only coherence
   signal was mean per-token log-prob of the steered text under the
   *unsteered* model. An unsteered LM already assigns very high probability
   to repeating a common token, so "platform platform platform" scored
   *better* than fluent, on-topic text -- the proxy could not distinguish
   a broken generation from a good one.
3. **Feature selection collapsed to one density band.** Selection took the
   `steer_features` highest-density first-shell features in
   `[1e-4, 0.1]`; all 8 landed at ~0.097 density, right under the 0.1 cap --
   a redundant slice of the spectrum rather than a sample of qualitatively
   different feature types.
4. **Training metrics never reached the harness or the log.** The
   checkpoint loop called `client.report(...)` but never logged locally,
   and no held-out (post-training) reconstruction quality number was ever
   computed -- the only training signal available was the final saved SAE
   itself.
5. **`checks.sae_replace_check` reported `match_fraction: 0.0`** on only 10
   positions, using a short, non-chat-template prompt ("In one sentence,
   what is the capital of France?", ~11 tokens).

## Investigation of problem 5

`sae_replace_check`'s inline `encode_preact -> batch_topk -> decode` calls
were mathematically identical to `forward_loss`'s main-shell reconstruction
for both `configs/smoke.json` and `configs/full.json`, because in both
configs the outermost matryoshka shell equals `sae_width` exactly (8192 and
32768 respectively), so `sae.decode(codes)` (full width) and
`sae.decode(codes, n_features=shells[-1])` (forward_loss's main
reconstruction) compute the same thing. No dtype or normalization
divergence was found either (both paths cast to float32 before
encode/decode and back to the model's dtype after). This means the check
was not silently comparing the wrong thing in smoke-1.

There *was* a latent (not yet triggered) divergence: `sae.decode(codes)`
with no `n_features` argument always decodes across the full `sae_width`,
while `forward_loss`'s reported main reconstruction is explicitly
`shells[-1]`-wide -- these differ if a future config sets `shells[-1] <
sae_width`. Fixed by introducing `MatryoshkaBatchTopKSAE.reconstruct(x)`,
used by both `forward_loss` and the check, so they cannot diverge even in
that case.

The most likely real cause of the exact `0.0` in smoke-1 is a combination
of (a) a barely-trained SAE (2,000,000 tokens is very little data for an
8192-wide, k=40 SAE) and (b) an 11-token, non-chat-template prompt giving
too few positions for a stable statistic, with argmax being a sharp
decision boundary that a still-noisy reconstruction can flip at nearly
every position. This is a real signal about SAE/check quality, not
primarily a code bug in the reconstruction path.

## Smoke-2 changes

1. Doses are now fractions of max activation, `[0.25, 0.5, 1.0, 2.0]`, and
   the steering hook only adds the vector at assistant-turn positions
   (the generation-prompt header onward, plus every generated token) --
   never on the user's own prompt tokens.
2. Coherence is now a dict per generation -- `distinct_ratio`, `max_run`,
   `repeat_4gram`, and the original `logprob` -- plus a derived `coherent`
   boolean, computed on the generated tokens only. `max_coherent_dose` is
   recorded per feature/sign (and per random-control index).
3. Feature selection now sorts in-range first-shell features by density and
   picks features spread evenly across quantiles of the log-density
   distribution, recording each one's `density` and `quantile`.
4. The training loop now logs
   `[train] tokens=... loss=... fve=... l0=... dead_frac=...` at every
   checkpoint and at the end, and a fresh held-out batch of 4096
   activations (harvested after training, never seen by the optimizer) is
   used to compute `held_out_fve`/`held_out_l0`, stored in
   `feature_stats.json` and the post-train check record.
5. `checks.sae_replace_check` now shares `sae.reconstruct(x)` with
   `forward_loss`, also reports the FVE of the replaced activations, and
   uses a >=40-token prompt.

Also: `tokens_target` raised to 8,000,000 and `checkpoint_every_tokens` to
1,000,000 for smoke-2 (width stays 8192); the summary/HTML report now shows
density, quantile, and `max_coherent_dose` per feature, per-dose coherence
fields and edit distance for both steered and random-control rows, and
highlights rows where `coherent` is false.
