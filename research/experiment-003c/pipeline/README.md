# Experiment 3C pipeline

A self-contained Matryoshka BatchTopK sparse-autoencoder (SAE) pipeline for
persona-direction discovery, meant to run on a RunPod pod (1x A40 48GB,
Ubuntu, PyTorch 2.4+, CUDA 12.8, Python 3.11). The **smoke** config is the
default; the **full** run uses the exact same code with a bigger
`configs/full.json`.

Everything that happens on the pod is reported to the AutoLabs orchestrator
Worker over HTTP (`report.py`); there is no local controller and no
activations ever touch disk (they live only in an in-process shuffle
buffer).

## Stages

1. **boot** -- `checks.identity_hook_check` (a hook that replaces the
   residual with itself must not change generation) and
   `checks.assistant_mask_sanity_check` (the assistant-turn token mask must
   never include template control tokens like `<|im_start|>`/`<|im_end|>`).
   Reports 2 records to stage `"boot"`.
2. **harvest + train** -- streams `dataset_name`/`dataset_split`, applies
   the tokenizer's chat template, keeps only assistant-turn token positions,
   runs the model truncated to layers `0..layer` (later layers are dropped
   entirely to save compute), captures the layer's residual stream at the
   masked positions into an in-memory shuffle buffer, and trains the
   Matryoshka BatchTopK SAE on samples drawn from that buffer. Logs
   `[train] tokens=... steps=... loss=... fve=... l0=... dead_frac=...` and
   reports a `"train"` record every `checkpoint_every_tokens` tokens (and
   once more at the end) with `{tokens_done, tokens_target, steps_done,
   loss, recon_loss, aux_loss, fraction_variance_explained, l0,
   dead_fraction}`, and saves a resumable safetensors checkpoint to
   `<workdir>/checkpoints/`. After the token target is reached, harvests
   one more, held-out batch of 4096 activations (never seen by the
   optimizer) and stores its FVE/mean-L0 as `held_out_fve`/`held_out_l0`
   in `feature_stats.json`.

   **Steps per harvested batch.** Harvesting (a 7B forward pass through
   `layer` decoder layers) is the GPU bottleneck; an SAE optimizer step is
   cheap by comparison. Running exactly one SAE step per harvested batch
   (the original design) starves the optimizer: on smoke-2 (width 8192,
   k 40, lr 3e-4, `batch_tokens` 4096) that pairing gave only ~245
   optimizer steps per 1,000,000 tokens harvested, and FVE/dead-fraction
   were still visibly improving at the last checkpoint (FVE 0.44 / 0.54 /
   0.59 and dead fraction 0.81 / 0.72 / 0.60 at 1M / 2M / 3M tokens) rather
   than having converged. `train_steps_per_batch` (config field, smoke
   default 4, full default 8) runs that many SAE optimizer steps per
   harvested batch once the shuffle buffer is ready, each on an
   independent fresh draw from `buffer.sample(batch_tokens)` (sampled with
   replacement, so within one harvested batch's worth of steps some tokens
   may repeat -- the buffer already holds up to `shuffle_buffer_size`
   distinct activations, decorrelating a "batch" from any single
   document). This multiplies the achieved step count roughly
   `train_steps_per_batch`-fold for the same harvesting cost: e.g. smoke's
   default of 4 turns ~245 steps/1M tokens into ~980 steps/1M tokens.
   `lr_warmup_steps` (default 500) linearly ramps the learning rate from 0
   up to `lr` over that many *optimizer steps* (not harvested batches),
   which matters more now that so many more steps happen early in a run;
   see `lr_warmup_multiplier` in `run_smoke.py`. Both `steps_done` (the
   cumulative optimizer-step counter) and the warmup schedule survive a
   resume: each checkpoint's `steps_done` is written to a sibling
   `sae_step_<tokens>.steps.json` file so a resumed run continues the same
   step count and warmup position rather than restarting `lr_warmup_steps`
   from zero at the (already-passed) resume point.
3. **post-train check** -- `checks.sae_replace_check`: builds a two-turn
   chat-templated conversation (the held-out prompt as the user turn, a
   fixed ~80-token assistant reply as the assistant turn), computes the
   assistant-position mask with the same `harvest.compute_assistant_mask`
   used to build the SAE's training data, and replaces the residual at
   `layer` with `sae.reconstruct(x)` -- the same reconstruction method
   `forward_loss` uses internally for its main (outermost-shell)
   reconstruction, so the two paths cannot diverge -- *only at those
   assistant positions*, leaving every other position untouched. Confirms
   the model's greedy argmax next token is preserved on >=90% of at least
   40 assistant positions. Also reports `replaced_fve` (fraction of
   variance explained of the SAE's reconstruction of the captured
   residual) and `ce_delta` (mean increase in next-token cross-entropy
   under replacement), both restricted to the same assistant positions,
   plus the held-out FVE/L0 from stage 2. Reports 1 record to stage
   `"train"`. Replacing *every* position of a plain, non-chat-template
   prompt (the previous version of this check) scores positions the SAE
   was never trained on and is meaningless -- smoke-2 saw argmax match
   0.15 and FVE -3.65 that way despite a held-out FVE of 0.68; see
   `../SMOKE-2.md`.
4. **rank** -- see "Rank stage" below. Selects the candidate features
   calibrate/screen cover. Reports one record per selected candidate to
   stage `"train"` (recordId prefix `"rank-"`; there is no dedicated
   Worker stage for this -- see "Rank stage").
5. **calibrate** (steer) -- reloads the model with all layers restored
   (harvesting only ever truncates the same in-memory model object, and
   `ActivationHarvester.close()` restores it -- no reload from disk needed).
   Picks `steer_features` in-range first-shell features spread evenly
   across quantiles of the log-density distribution (not just the densest
   ones), sweeps sign x dose x scenario with the additive steering vector
   applied *only* at assistant-turn positions (the generation-prompt header
   onward, plus every generated token -- never the user's own prompt
   tokens), and also runs unsteered baselines and a random-direction null
   control swept across the same doses with the same position masking.
   Reports every generation (baseline + steered + random-control) as a
   record to stage `"calibrate"`, batched at 200 records per Worker call.
6. **screen** -- see "Screen stage" below. Reports one record per direction
   (feature/sign, positive-control persona vector/sign, or random null) to
   stage `"screen"`.
7. **describe** -- see "Describe stage (judge pass one)" below. Disabled
   when `describe_top_n <= 0`; otherwise reports one summary record per
   judged direction to stage `"judge"`.
8. **analyze / done** -- computes embedding-free proxies (normalized
   word-level edit distance and length delta vs. the same-scenario
   baseline, plus the per-generation coherence dict already computed during
   calibration: `distinct_ratio`, `max_run`, `repeat_4gram`, `logprob`, and
   the derived `coherent` boolean), rolls up `max_coherent_dose` per
   feature/sign and per random-control index, writes
   `<workdir>/summary.json` and a standalone `<workdir>/smoke-report.html`
   (density/quantile/max_coherent_dose per feature, per-dose coherence
   fields, incoherent rows highlighted, plus the screen table and verdict
   described below), then reports stage `"done"` with `status="complete"`.

Any uncaught exception is reported to stage `"done"` with `status="failed"`
and the exception message, then re-raised (the process exits non-zero; the
Worker sees the failure even if nothing above got to `"done"`).

## Rank stage

The full run screens `config.screen_features` (256) of the 32,768 trained
SAE features. Picking those 256 purely by firing-density quantile (what
`steer.select_steer_features` does on its own) samples across "how common
is this feature" with no signal about whether a feature's activation
tracks anything persona-like -- it's an unbiased slice of the density
spectrum, not a targeted one. The rank stage (`rank.py`) adds a label-free
targeting signal on top, and runs after the post-train check and before
calibrate.

**Persona-context shift.** The stage builds a diverse set of persona-style
system-prompt "contexts" -- `config.control_prompts`' `positive_system_
prompt`/`negative_system_prompt` pairs (6 contexts for the shipped 3
controls) plus `config.context_prompts` (12 more neutral, hand-written
prompts spanning traits, e.g. "You are warm and encouraging.", "You are
terse and clinical.", "You are playful and lighthearted.") -- and, for
every context x `scenario` (the calibrate stage's `config.steer_scenarios`
slice), generates a reply under that system prompt
(`screen.generate_with_system_prompt`), captures the layer's residual
stream at the generated assistant tokens, and encodes it with `sae.encode`
(the same post-batch-topk sparse codes `forward_loss` trains on), mean-
pooling the codes over generated tokens (`rank.capture_context_
activations`). This is "label-free" in the sense that no single context is
designated as the trait being screened for -- the contexts are just a
diverse spread, and a feature that shifts with *any* of them (in a way
that's consistent across scenarios within a context) ranks higher.

**Shift score** (`rank.compute_shift_stats`, pure numpy/Python, unit
tested). For each feature, treats each context as an ANOVA group and each
of its scenarios' per-context mean activation as one observation in that
group:

- `shift_f`: between-context variance of the per-context mean (weighted by
  scenario count) divided by the pooled within-context variance across
  scenarios, plus a small epsilon in the denominator so a near-zero
  within-context variance gives a large finite ratio rather than `inf`
  (not valid JSON). Large when a feature reliably shifts with context and
  is stable within a context; small when scenario-to-scenario noise
  dominates or every context looks alike.
- `shift_maxdiff`: the largest absolute difference between any two
  contexts' per-context mean, normalized by the feature's max activation
  over training, so the shift's magnitude is comparable across features on
  different absolute activation scales.

**Filters before ranking** (`rank.rank_candidates`): firing density in
`[config.firing_density_min, config.firing_density_max]` (from
`feature_stats.json`), not dead (`rank.is_dead`: `firing_density <= 0`,
i.e. never fired during the density-tracking window), and a computed shift
score. Each surviving candidate also records `shell`
(`rank.feature_shell`: the smallest `config.matryoshka_shells` boundary
that contains it, e.g. 1024/4096/8192) -- inner shells are preferred as a
tie-break when sorting by shift score, since the matryoshka training
objective forces inner-shell features to carry a self-contained
reconstruction rather than only refining an outer one, making them the
more likely place for a coarse, generalizable direction to live.

**Selection.** `config.screen_features` (the total budget, e.g. 256 full /
8 smoke) candidates are picked as the union of two disjoint slices:

- `"shift"`: `round(config.screen_features * config.rank_shift_fraction)`
  (default 0.75) candidates, the top slice by `shift_f` descending, ties
  broken by preferring inner shells and then feature index.
- `"quantile"`: the remaining budget, a density-quantile spread
  (`steer.quantile_indices`, the same mechanism `steer.select_steer_
  features` uses) over whatever candidates the shift slice didn't already
  take -- so the two slices never overlap. This slice is kept
  deliberately even though the shift score exists: the shift score is
  itself biased by which contexts were chosen (see "Bias" below), so a
  feature that the shift score misses -- because none of the 18 contexts
  happened to move it, not because it carries no persona-relevant
  direction -- can still be screened, and it also gives a distribution-
  matched comparison for how much the shift-ranked slice actually helps
  once the describe stage's judge results come back.

For the smoke config (`screen_features` 8, `rank_shift_fraction` 0.75)
this is 6 shift-selected + 2 quantile-selected, matching the previous
all-quantile 8-feature selection in spirit (same total, same underlying
`quantile_indices` mechanism for the non-shift portion).

Every selected candidate is written to `candidates.json`'s
`screen_features` list as `{feature, shift_f, shift_maxdiff, density,
shell, selection, rank}` (`selection` is `"shift"` or `"quantile"`; `rank`
is the feature's 1-indexed position in the *full* candidate pool sorted by
shift score, so a quantile pick's rank shows where it would have landed by
shift score alone) and reported to stage `"train"` with
`recordId = f"rank-{feature}-{selection}"` -- `"train"`, not a dedicated
`"rank"` stage, because `orchestrator-worker/src/persona-3c.ts`'s
`PERSONA_3C_STAGES` enum does not include one and this change is meant to
land without a Worker change.

**Wiring.** `steer.select_steer_features` takes an optional
`explicit_features` argument; when `run_smoke.stage_steer` finds
`candidates.json`, it passes `candidates["screen_features"]` through
`steer.run_calibration` to `select_steer_features`, which then uses that
list as-is instead of doing its own density-quantile selection. When
`candidates.json` doesn't exist (e.g. a workdir from before this stage
existed, or a config that skips it) calibrate falls back to the previous
behavior: density-quantile selection sized off `config.steer_features`
directly.

**Bias.** The shift score's contexts are diverse but *chosen* -- 3
literature-style trait pairs plus 12 hand-written prompts, not sampled
from any principled distribution over "ways a system prompt could differ".
A feature whose persona-relevant direction only shows up under a context
this set never tried will score low on `shift_f` despite being exactly
what the full run is looking for, and conversely a feature could shift
reliably across these 18 contexts for a reason that has nothing to do with
persona (a topic or formatting correlate of the specific wording chosen).
Screening in the top-shift slice is therefore a hypothesis about which 192
(of 256) features are *more likely* to be interesting, not a guarantee,
and the describe stage's blinded judge -- not this stage -- is what
actually says what a direction is. The quantile slice exists precisely
because of this: it is not selected on any activation signal, so it is the
run's control for whether the shift-ranked slice does better than an
unbiased sample would.

## Screen stage

Smoke-2 found that edit distance and the coherence proxy cannot separate a
real SAE feature from a matched-norm random direction -- both are equally
coherent and produce similar edit distances at the same dose (see
`../SMOKE-2.md` "Steering"). The screen stage (`screen.py`) checks the two
things that can: whether steered-vs-baseline is *separable* on scenarios
the classifier never trained on, and whether the steering effect is
*consistent* in direction across scenarios rather than incidental.

For every direction -- each selected SAE feature x sign (reusing the
calibrate stage's own dose sweep to pick that feature/sign's
`max_coherent_dose`, falling back to `screen_dose_fallback` if none
cleared the coherence bar), each positive-control persona vector x sign
(see below; these run their own mini dose-coherence sweep on the
calibrate stage's smaller scenario set, since persona vectors are new to
this stage), and each random-direction null (dosed at the *median* of the
real features' `max_coherent_dose`, since a null has no calibration of its
own to read a dose from) -- the pipeline generates one steered completion
per scenario across `screen_scenarios` scenarios (config: 8 for smoke,
reusing the 4 `steer_scenarios` plus 4 new open-ended ones added to
`scenarios.json`; 24 for the full run) and one shared set of unsteered
baselines.

Each generation is embedded two independent ways so a result isn't an
artifact of one representation: the model's own layer-19 residual stream,
mean-pooled over generated tokens only (a light forward-hook capture, the
same "extra pass with a hook" pattern `steer.coherence_logprob` already
uses), and a hashed bag-of-words unigram+bigram lexical vector (4096 dims,
L2-normalized, deterministic across runs) that shares no machinery with
the model at all.

- **Separability**: leave-one-scenario-out logistic regression (numpy
  only, full-batch gradient descent with L2 -- no sklearn on the pod
  image) trained on every scenario but one to classify steered vs.
  baseline, scored on the held-out scenario; accuracy and a rank-based
  (Mann-Whitney-U) AUC are aggregated across folds, for both embedding
  views.
- **Consistency**: for each direction, the per-scenario difference vectors
  (steered residual embedding minus that scenario's baseline) should point
  the same way if the direction has a real, generalizable effect -- scored
  as the mean pairwise cosine similarity among a direction's own diff
  vectors, against a null built from the mean cosine similarity to diff
  vectors drawn from *other* directions (a different feature/control/
  random each draw).

**Positive controls.** Three persona vectors (config: `control_prompts`,
default `evil_benevolent` / `sycophantic_honest` / `hallucinating_factual`)
are built the way Chen et al. / Arditi et al. / Sleight et al. construct a
persona direction: the mean layer-19 residual over generated tokens under
a short, blunt `positive_system_prompt` naming the trait, minus the same
under the opposite-trait `negative_system_prompt`, averaged over the
calibrate stage's scenarios. These should separate and be consistent if
the screen methodology itself is sound, independent of whether any SAE
feature does.

**Random-direction nulls.** `config.random_directions` (smoke and full both
20, up from visual run 1's 2 -- see `../VISUAL-1.md` "Next") fresh random
unit vectors, each dosed at the *median* of the real features'
`max_coherent_dose` (a null has no calibration of its own to read a dose
from). This is independent of the calibrate stage's own 2 random-direction
controls used for its dose sweep (`steer.run_calibration`, unchanged).

**Gate G0 (preregistered, diagnostic only -- this smoke pipeline does not
act on it).** A direction (one feature/sign or control/sign pair) "passes"
if its resid-view separability AUC beats the best random-direction null's
AUC (`max_random_resid_auc`, the max over all `random_directions` draws;
`max_random_resid_auc_p95` is also reported as a less single-draw-sensitive
ceiling) *and* its consistency (`mean_cos`) exceeds that direction's own
null consistency (`null_mean_cos`) by more than 0.1. A **feature** or
**control** (an "entity", as opposed to one of its two signed directions)
passes if *either* of its signs passes -- visual run 1 counted controls per
sign instead of per control, which undercounted a control whose positive
sign passed but whose negative sign missed only on the consistency margin
(evil/benevolent: 0.10 vs 0.06, see `../VISUAL-1.md` "Screen").
`run_smoke.compute_screen_verdict` reports both: `feature_directions_passing`
/ `feature_directions_total` and `control_directions_passing` /
`control_directions_total` are the old per-sign counts, while
`features_passing` / `features_total` and `controls_passing` /
`controls_total` are per-entity (the latter out of the number of distinct
controls, 3), with per-sign detail kept in `features_detail` /
`controls_detail` (`[{id, pos_passes, neg_passes, passes}, ...]`). The
report's verdict line surfaces the per-entity counts; a control entity
failing this bar would say more about this screen methodology than about
any trait, since the persona-vector construction is separately validated
in the literature.

Every direction's record (`{kind, id, sign, dose, n_scenarios,
separability: {resid: {acc, auc}, lexical: {acc, auc}}, consistency:
{mean_cos, null_mean_cos}, coherent_fraction}`) is written to
`screen_records.json` and reported to stage `"screen"` with
`recordId = f"screen-{kind}-{id}-{sign}-{dose}"` (`sign` rendered as
`pos`/`neg`/`na`, matching the calibrate stage's own recordId convention).

**Screen generations.** For every direction and every scenario it was
scored on, the steered text, the same-scenario baseline text, and that
generation's `finish_reason` and coherence dict are written as a flat list
of `{kind, id, sign, dose, scenario, text, baseline_text, finish_reason,
coherence}` dicts to `screen_generations.json` (built by
`screen.build_generation_records`) and included in `summary.json` under
`screen_generations`. Each is also reported to stage `"screen"` with
`recordId = f"screen-gen-{kind}-{id}-{sign}-{scenario}"` (`sign` rendered
the same `pos`/`neg`/`na` way as every other screen-stage recordId), so the
harness receives the underlying text behind every screen-table row, not
just the aggregate separability/consistency numbers. The HTML report shows,
under the screen table, one collapsible `<details>` block per direction
(same sort order as the table) with up to two example baseline-vs-steered
scenario pairs side by side.

## Describe stage (judge pass one)

The screen stage tells us *whether* a direction's steered-vs-baseline effect
is separable and consistent, not *what* it is -- a persona-like shift, a
topic change, and a formatting artifact can all look the same on those
numbers. The describe stage (`describe.py`) answers that with a blinded
judge: for each of the `describe_top_n` directions (ranked by residual-view
separability AUC, ties broken by consistency `mean_cos`; smoke 6, full 40;
`0` disables the stage), **plus** the `describe_null_directions`
lowest-resid-AUC `kind="random"` directions from the screen (smoke and full
both 3 -- see "Nulls" below), and each scenario the direction was screened
on, it shows the judge the baseline text and the steered text for that
scenario -- labeled `A`/`B`, order randomized per pair -- and asks it to
answer one question with strict JSON: `{"property": <=200 chars or "",
"more_in": "A"|"B"|"neither", "about": "speaker"|"content"|"format",
"confidence": "low"|"medium"|"high"}` (name one property of how the
speaker comes across that differs, and say which response shows more of
it; `neither` means no meaningful difference, `property` may be `""`). The
judge never sees a persona vocabulary, a feature id, or which side is
steered.

**The judging runs inside the Worker**, not on the pod: `describe.py` only
builds the blinded pairs and drives the Worker's queue.
`orchestrator-worker/src/persona-3c.ts` owns the actual OpenAI call
(`gpt-5.6-luna`, reasoning effort `high`, `store: false`, a 60s per-call
timeout, strict `json_schema` output) and all budget bookkeeping, so no
laptop process needs to stay alive for this stage to run to completion.

- `describe.build_judge_pairs(screen_generations, directions, top_n,
  null_directions)` selects the top `top_n` directions by resid AUC
  *union* the `null_directions` lowest-resid-AUC `kind="random"`
  directions, and for every scenario each selected direction was screened
  on emits two pairs: `orderSwap=False` (`textA`=baseline, `textB`=steered)
  and `orderSwap=True` (`textA`=steered, `textB`=baseline) -- so every
  steered/baseline pair is judged in both orders.
- `describe.submit_judge_plan` POSTs pairs to `/api/persona-3c/judge/plan`
  in batches of <=500 (append mode: re-submitting the same pairs is a
  no-op, since each job's id is a hash of `{directionKey, scenario,
  orderSwap}`). **Judge budget guard**: `judge_budget_usd` (smoke 3, full
  30) and `judge_call_ceiling` (smoke 200, full 4000) are separate from the
  run's own `budget_usd`/call ceiling and can only be *raised*, never
  lowered, by a later plan call. Before every provider call the Worker
  reserves a worst-case cost (input tokens estimated from prompt length /
  3.5, plus 400 output tokens, at Luna pricing); it refuses the call
  outright if `spent + reserved + worst-case` would exceed the budget or if
  the call ceiling is reached, settles to the actual cost on success, and
  charges the worst-case on failure. A job is retried up to 3 attempts
  before being marked `failed`.
- `describe.drive_judge` calls `/api/persona-3c/judge/run` (<=10 jobs per
  call by default, processed sequentially inside the Worker -- CPU-time
  limits, not wall time -- against a **300s client read timeout**; a full
  10-job sequential judge batch can take well over `report.py`'s default
  30s timeout, and the first live judge pass showed the client retrying,
  and duplicating in-flight work, while the Worker was still working
  through the batch) until the queue is drained.
- `describe.fetch_results` reads back every complete job's parsed response
  from `/api/persona-3c/judge/results` (auth required; the blinded texts
  themselves are never returned by any route).
- `describe.cluster_descriptions` unblinds each response's `more_in` using
  its local `orderSwap` (`steered_has_more = (more_in == "B") != orderSwap`;
  `more_in == "neither"` has no direction and counts toward `n_none`
  regardless of whether `property` is also empty -- see
  `describe._steered_has_more`), then clusters the `property` text alone
  (direction-free) per direction: two independent cosine views averaged
  (the screen stage's hashed lexical vector, and a fresh TF-IDF over
  lowercased, stopword-stripped unigrams+bigrams of just that direction's
  properties), average-linkage agglomerative clustering at a
  cosine-distance threshold of 0.35 (`CLUSTER_LINKAGE_THRESHOLD`, loosened
  from an earlier 0.5 after the first live pass showed swapped-order
  paraphrases failing to cluster), followed by a fallback keyword-overlap
  merge (`_keyword_overlap_merge`, Jaccard >= 0.6 on stopword-stripped
  content words) for short descriptions the cosine view under- or
  over-weights. Reports per direction: `n_none`, `n_described`,
  `none_rate`, `largest_cluster_size`, `cluster_property` (the description
  with the highest mean similarity to the rest of the largest cluster),
  `direction_agreement` (fraction of the largest cluster's members whose
  unblinded `steered_has_more` matches that cluster's majority direction),
  `about_counts`, `confidence_counts`, `named` (largest cluster >= 50% of
  described, `direction_agreement` >= 0.8, and a strict majority of that
  cluster is `about: "speaker"`), and `consistency_score` =
  `largest_cluster_size / n_described` * `direction_agreement`.

**Nulls.** The first live judge pass (48 pairs) found the judge never
returned `neither` at all -- greedy decoding makes every steered text
differ from its baseline in *some* way, for a real feature or a random
direction alike -- so `none_rate` cannot separate a real direction from a
null. `describe.compute_null_stats`/`describe.apply_named_above_null`
instead gate on `consistency_score`: a direction is `named_above_null` iff
it is `named` *and* its `consistency_score` exceeds the best
(`null_consistency_max`) any `kind="random"` null direction achieved, by
more than `describe_null_margin` (default 0.1). `describe_results.json`'s
`null_stats` reports `n_null_directions`, `null_none_rate`, `null_named`,
`null_consistency_max`, `null_consistency_mean`, and the resulting
`named_above_null_threshold`; `describe-report.html` shows the null
directions in their own block, separate from the real directions' table.

Writes `describe_results.json` (`{plan, results, clusters, null_stats}`)
and `describe-report.html` under `<workdir>`, and reports a compact
per-direction summary record (`describe-<directionKey>`) to the harness
under stage `"judge"`.

## Batching and generation numerics

Every `model.generate` call in the pipeline -- calibrate's dose sweep,
screen's baseline/steered/control generations, rank's per-context
generations -- goes through `generation.generate_batch` (`generation.py`)
rather than looping one prompt at a time. Prompts are left-padded
(`tokenizer.padding_side = "left"`, pad token = eos if the tokenizer has
none of its own) so every row's real tokens end at the same column and the
newly generated tokens line up column-for-column across the batch;
`config.generation_batch_size` (smoke 8, full 24) is the chunk size, and a
single prompt is just the `batch_size=1` special case of the same code
path -- there is no separate unbatched generation function anywhere in the
pipeline any more.

**The steering hook under batching.** The additive steering vector must
still land only at assistant-turn positions (never a row's own left
padding or its user-prompt tokens), and under left padding that boundary
is a *per-row* padded-coordinate position (`pad_length[row] +
unpadded_boundary[row]`), not the same column for every row.
`steer.make_additive_hook` takes a `[batch]` tensor of boundaries (and an
optional `[batch]` tensor of per-row *scales*, e.g. `sign * dose`, so one
hook registration can cover an entire dose sweep instead of one per
generation) and masks each row independently on the first (prefill)
forward pass; every decode step after that steers every row unconditionally,
since a boundary is always strictly before the end of its own padded
prompt. `steer.build_steering_hook_factory` is the adapter
`generation.generate_batch`'s optional `hook` argument expects: a factory
called with that chunk's per-row left-pad counts, which shifts the
caller's already-known *unpadded* boundaries into padded coordinates and
registers the actual hook.

**Where sweeps got batched.** `steer.run_calibration` batches every (dose,
scenario) pair for one feature/sign (or one random-direction draw) into a
single call, using the hook's per-row scale for `sign * dose`; baselines
are one batch across every scenario. `screen.generate_steered_set` /
`generate_baseline_set` batch across a direction's `screen_scenarios`;
`screen.build_persona_vector` batches the positive- and negative-
system-prompt generations for every scenario together (`2 * len(scenarios)`
rows); `screen.sweep_control_doses` batches every (sign, dose, scenario)
triple for one control. `rank.capture_context_activations` batches every
(context, scenario) pair. Residual-stream capture for coherence
(`steer.coherence_logprob_batch`) and for embeddings
(`screen.capture_mean_residual_batch` /
`screen.capture_generated_hidden_batch`) is likewise batched, via a plain
(non-generating) forward pass over right-padded prompt+generated sequences
-- right padding is safe there specifically because causal attention masks
already block every real token from attending to anything after it, so
trailing pad columns cannot change an earlier real token's hidden state;
each row's own real generated-token positions are read directly rather
than through an explicit mask.

**Numerics caveat.** Decoding is always greedy (`do_sample=False,
num_beams=1`), and `config.seed` still seeds every random draw (the
random-direction controls/nulls) the same way regardless of batch size --
but batched, left-padded generation is not guaranteed to produce
byte-identical token ids to the exact same prompt generated alone. Padded
attention, batched matmul reduction order, and (depending on the
attention implementation) padding-aware kernel selection are not bit-
identical to an unpadded, unbatched forward pass on most hardware/BLAS
backends. This is expected numerical noise from batching, not a
regression to chase -- coherence and separability are computed per
generation regardless of how it was batched, so this can shift individual
metric values slightly between two runs of the same config but should not
change the pipeline's qualitative conclusions.

## Resumability

Each stage checks for its own output file under `<workdir>` before doing any
work:

| Stage | Marker file(s) |
|---|---|
| boot | `boot_checks.json` |
| harvest+train | `sae.safetensors` + `feature_stats.json` (final); `checkpoints/sae_step_*.safetensors` + sibling `checkpoints/sae_step_*.steps.json` (partial) |
| post-train check | `post_train_check.json` |
| rank | `candidates.json` |
| calibrate | `calibration_records.json` |
| screen | `screen_records.json` + `screen_generations.json` |
| describe | `describe_results.json` (skipped entirely if `describe_top_n <= 0`) |
| analyze | `summary.json` + `smoke-report.html` |

If harvest+train is interrupted, the SAE resumes from the latest checkpoint
under `checkpoints/`. The dataset stream itself is **not** seekable (it's a
`datasets` streaming shuffle buffer, not an index), so a resumed run starts
tokenizing a fresh, differently-shuffled pass over the dataset rather than
picking up at the exact same document -- for a smoke test with an 8M-token
target and a `seed`-shuffled 200k-conversation dataset this is a fine
tradeoff; the SAE optimizer state (Adam moments) is *not* checkpointed, only
the weights, so a resume restarts Adam's moving averages from zero. The
cumulative optimizer-step count (`steps_done`, used for the LR warmup
schedule) *is* checkpointed, in the sibling `<checkpoint>.steps.json` file,
so a resume continues the warmup schedule rather than restarting it; the
per-feature dead-window counters (`tokens_since_fired`) are *not*
checkpointed (see "Dead-feature accounting" below), so a resume restarts
every feature's dead-window clock from zero, same as Adam's moments. A run
resumed from a *partial* checkpoint (training not yet finished) collects
its own fresh held-out batch once training completes, same as a run that
never crashed; a run resumed after `sae.safetensors` + `feature_stats.json`
already exist skips harvest+train entirely and reuses whatever
`held_out_fve`/`held_out_l0` that prior run already computed.

### Dead-feature accounting

`sae.MatryoshkaBatchTopKSAE.tokens_since_fired` (one counter per feature)
is incremented by `codes.shape[0]` inside `update_dead_stats`, called from
`forward_loss(x, ...)` -- and `x` is always `buffer.sample(batch_tokens)`,
i.e. the fixed-size batch actually fed to the SAE for one optimizer step,
never the (irregularly-sized) batch of activations just harvested from the
model. So the window is already counted in **tokens seen by the SAE**
(`batch_tokens` per optimizer step, i.e. `steps_done x batch_tokens`
cumulatively) rather than harvested tokens, and this held true before and
continues to hold now that `train_steps_per_batch` calls `forward_loss`
several times per harvested batch -- each call advances the counter by
another `batch_tokens`, so `dead_feature_window_tokens` continues to mean
what its name says regardless of how many SAE steps happen per harvested
batch. The aux loss also already targets only dead features correctly:
`dead_preact = preact * dead_mask` zeroes every column whose feature is
not currently "dead" before `batch_topk` picks the top `k_aux` activations,
so `dead_codes` (and therefore `aux_recon`) can only ever contain
currently-dead features. No fix was needed for either the counting units
or the targeting. The one related gap found (not fixed here, since it's
a checkpoint-persistence question rather than a counting-units bug): like
Adam's moments, `tokens_since_fired` is not part of `state_dict_safetensors`
and is not checkpointed, so a resume restarts every feature's dead-window
clock from zero rather than preserving how close each feature was to being
declared dead.

## Running

```sh
cd research/experiment-003c/pipeline
python -m pip install -r requirements.txt   # torch assumed preinstalled

# Smoke test (defaults):
python run_smoke.py --config configs/smoke.json

# Full run (same code, bigger config):
python run_smoke.py --config configs/full.json
```

On a RunPod pod, `runpod_start.sh` does the apt/pip setup, loads
`/workspace/3c/.env`, and runs the smoke config with logging teed to
`/workspace/3c/smoke.log`:

```sh
bash runpod_start.sh configs/smoke.json
```

## Environment variables

| Var | Meaning |
|---|---|
| `AUTOLABS_3C_WORKER_URL` | Base URL of the orchestrator Worker, e.g. `https://autolabs-orchestrator.raphaelbahadurkhan.workers.dev` |
| `AUTOLABS_3C_TOKEN` | Bearer token for `/api/persona-3c/*` |
| `AUTOLABS_3C_RUN_ID` | Existing run id. If unset, `run_smoke.py` calls `start_run()` using `config.manifest_hash` / `config.budget_usd` / `config.idempotency_key` (or a config-hash-derived idempotency key if none is set). |

If `AUTOLABS_3C_WORKER_URL` is unset, `report.py` logs a warning and
continues -- the GPU job is never blocked on the harness being reachable.

## Tests

```sh
python -m pytest research/experiment-003c/pipeline/tests -q
```

125 tests, all CPU-only (95 in `tests/test_pipeline.py`, 11 in
`tests/test_rank.py` for the rank stage, and 19 in `tests/test_generation.py`
for batched generation and the batched steering hook, see below): SAE forward/loss and matryoshka-nested-loss-decreases
on a random 64-dim toy, `sae.reconstruct`'s equivalence with
`forward_loss`'s main reconstruction on a toy SAE, BatchTopK's
per-token-average-k property, the Worker's canonical-JSON sha256 contract
against a known hash, the assistant-turn masking algorithm and the
generation-prompt boundary computation against a small fake tokenizer that
reproduces a `<|im_start|>{role}...<|im_end|>` chat template without any
network access (including that `sae_replace_check`'s two-turn chat mask
selects only the assistant reply's tokens), `checks.fraction_variance_explained`
/ `checks.mean_cross_entropy_at_positions` on toy tensors, the
quantile-spread feature-selection helper, the coherence metrics (a
repetitive token sequence must be `incoherent`, a varied one `coherent`),
config validation for `train_steps_per_batch` / `lr_warmup_steps`, the
`lr_warmup_multiplier` schedule (0 at step 0, full LR at and after
`lr_warmup_steps`), a CPU toy loop asserting `train_steps_on_buffer`
performs exactly `train_steps_per_batch` optimizer steps per call (and that
`steps_done` accumulates correctly across calls, as it would across a
resume), and (screen.py, see "Screen stage" below) the hashed lexical
embedding (deterministic and L2-normalized), the numpy-only
leave-one-scenario-out logistic regression on separable toy data (accuracy
and AUC near 1) and inseparable toy data (accuracy and AUC near chance,
0.5, by construction -- steered and baseline share the exact same
embedding within every scenario, so the only consistent fit is `p=0.5`
everywhere), `auc_score`'s perfect-separation and single-class-is-nan
cases, direction consistency vs. a null on toy vectors (a direction whose
per-scenario diffs all point the same way scores much higher than its
cosine similarity to unrelated directions), the dose-selection
helpers (`resolve_dose`, `median_dose_with_fallback`,
`feature_max_coherent_doses`, `extract_feature_info`, `flatten_doses`),
`run_smoke.compute_screen_verdict`'s per-entity (either-sign) gate G0 logic
on a toy screen-row set -- a control/feature whose positive sign passes and
negative sign only misses on the consistency margin must still count as
passing overall (the exact undercount visual run 1 found), plus the max
and 95th-percentile of the random-direction null AUCs -- and
`screen.build_generation_records`'s recordIds and payload shape on a toy
set of directions (including the empty-baseline fallback); and
(`rank.py`, see "Rank stage" above) the F-like shift-score helper on toy
per-context activations (a feature that shifts consistently between
contexts scores much higher than one that only has scenario-to-scenario
noise, and single-context/zero-max-activation edge cases are 0 rather than
NaN/inf), `feature_shell`/`is_dead`, `rank_candidates`'s density/dead
filtering and its shift-vs-quantile union selection (disjoint feature
sets, exact budget split, every record carrying the full schema, the empty-
candidate-pool case), and `steer.select_steer_features`'s
`explicit_features` bypass (an explicit list is honored verbatim --
including features outside the density window that would otherwise be
filtered out -- while `explicit_features=None` still falls back to the
original density-quantile selection).

`tests/test_generation.py` (see "Batching and generation numerics" above):
`generation.pad_left`/`pad_lengths_from_attention_mask` (left-padding a
ragged batch and recovering each row's pad count), `ensure_left_padding`,
`trim_at_eos` (finish_reason `"stop"` vs `"length"`, including an eos as
the very first generated token), `steer.make_additive_hook`'s per-row
boundary mask under simulated padding (two rows with different padded-
coordinate boundaries in one prefill chunk, each steered only from its own
boundary onward), its unconditional-steering decode-step behavior, its
per-row dose-scale multiplier, and its tuple-output passthrough;
`steer.build_steering_hook_factory` shifting an already-known unpadded
boundary by a chunk's real per-row left-pad count (registered on a real
`nn.Module` stand-in for a decoder layer) and its `vector=None` no-op case;
and one end-to-end test running `generation.generate_batch` (real,
left-padded, batched `model.generate`) against a tiny, randomly
initialized (no download) GPT-2 model, checking per-row `prompt_ids`,
`num_tokens`, and a valid `finish_reason` survive batching alongside a
differently-lengthed sibling prompt (skipped, not failed, if
`transformers` is unavailable).

## Estimated smoke-test runtime on 1x A40 48GB

Smoke-1 (see `../SMOKE-1.md`) actually ran on an A40: harvest+train reached
its (then) 2,000,000-token target in **~10 minutes**, and the full pod
(model load + boot + harvest+train + post-train check + 220 calibrate
generations + analysis) took **~65 minutes** of pod time on a $0.49/h
secure-cloud instance. The numbers below scale that measurement to
smoke-2's larger config (`tokens_target` raised to 8,000,000,
`checkpoint_every_tokens` to 1,000,000, and `doses` widened from 3 to 4
values) rather than being a from-scratch FLOPs estimate; still
order-of-magnitude, not a guarantee.

| Phase | Work | Rough time |
|---|---|---|
| Model load (x2: truncated for harvest, restored in place for steering) | download/cache + weight load | 2-5 min |
| Harvest + train | 8M tokens forwarded through 20/28 layers, ~4x smoke-1's measured 10 min for 2M tokens | ~35-45 min |
| Post-train check | 2 forward passes (capture + replace) on a >=40-token prompt | <1 min |
| Steer / calibrate | 8 features x 2 signs x 4 doses x 4 scenarios = 256, + 4 baselines + 2 random x 4 doses x 4 scenarios = 32, = 292 generations at up to 256 tokens, plus 288 extra coherence forward passes | 65-90 min (up from smoke-1's 220-generation screen, scaled by the extra dose) |
| Analysis + report | pure Python/string work | <1 min |
| **Total** | | **roughly 100-140 minutes** |

The steering phase dominated because generations ran unbatched and
sequentially (one hook configuration at a time, since the additive
steering vector changes every generation) -- see "Batching and generation
numerics" above for how `generation.generate_batch` now batches every
dose/scenario/context sweep instead. The full-run config (150M tokens, 256
features, 24 scenarios) scales harvest+train roughly linearly with
`tokens_target`, and the calibration screen is 256 features x 2 signs x 4
doses x 24 scenarios = 49,152 generations (plus the screen stage's own
~roughly-comparable volume across 24 `screen_scenarios` and 20
`random_directions`) -- at `config.generation_batch_size` 24 that is
~2,048 `model.generate` calls of 24 prompts each instead of 49,152 calls
of 1, an ~24x reduction in the number of forward-pass launches for that
phase (wall-clock speedup is sublinear in batch size on a real GPU --
larger batches use more of an A6000's compute per call but each call also
does more work -- but launch/Python-overhead-bound sequential generation,
which is what one-prompt-at-a-time `model.generate` calls mostly were,
should not scale anywhere near linearly with prompt count once batched).

## Assumptions / things not verified without a GPU

- `tokenizer.apply_chat_template(..., add_generation_prompt=True)` on a
  prefix ending in a user turn renders exactly the prompt up through the
  assistant header, and re-tokenizing a message prefix reproduces a token
  prefix of the full-conversation tokenization. This is standard for
  well-behaved chat templates (including Qwen's) but was only verified
  against a hand-written fake tokenizer in `tests/`, not against the real
  Qwen2.5-7B-Instruct tokenizer.
- `HuggingFaceH4/ultrachat_200k` (`train_sft` split) exposes a `messages`
  column of `{role, content}` dicts; `harvest.extract_messages` assumes
  this shape (with a fallback for lmsys-style `conversation`, unused by
  default since lmsys-chat-1m is gated).
- Truncating `model.model.layers` to a `nn.ModuleList` slice and calling
  `model.model(...)` directly (bypassing `lm_head`) is the standard way to
  skip unneeded decoder layers and the vocab projection in a
  transformers `*ForCausalLM` model; not run against the real Qwen2.5-7B
  class here.
- BatchTopK dead-feature aux-loss formula (top-k_aux of dead-only
  pre-activations reconstructing the main reconstruction's residual,
  weighted by `aux_loss_coef`) follows the OpenAI/Anthropic TopK-SAE
  recipe; the exact coefficient (default `1/32`) is a common literature
  default, not tuned here.
- Runtime estimate above scales smoke-1's actual A40 measurement rather
  than a from-scratch FLOPs calculation, but is still only order-of-
  magnitude; real HF `generate()` throughput on an A40 varies a lot with
  attention implementation (SDPA vs. flash-attn-2), sequence length, and
  whether `torch.compile` is used (not used here).
- `checks.sae_replace_check`'s >=90% threshold is checked and logged as a
  warning if missed, but does not hard-fail the run -- SAE reconstruction
  quality is expected to vary and a smoke test's job is to surface that
  number, not gate on it. Smoke-1 saw `match_fraction: 0.0` on a short,
  non-chat-template, 11-token prompt; see `../SMOKE-1.md` for that
  investigation (no divergence found between the check's and
  `forward_loss`'s reconstruction math for the shipped configs, but the two
  now share a single `sae.reconstruct` method regardless). Smoke-2 then
  found a second, larger problem: even with a long (89-token) prompt, the
  check was replacing *every* position of a plain, non-chat-template
  prompt, including positions the SAE was never trained on (it only ever
  sees assistant-turn positions), giving `match_fraction: 0.148` and a
  negative `replaced_fve` (-3.65) despite a held-out FVE of 0.68; see
  `../SMOKE-2.md`. The check now builds a chat-templated two-turn
  conversation and masks to assistant positions with
  `harvest.compute_assistant_mask`, the same masking the SAE's training
  data uses, and requires >=40 such assistant positions rather than >=40
  raw prompt tokens.
- `steer.compute_generation_boundary` assumes the same chat-template
  prefix-stability property `harvest.compute_assistant_mask` relies on:
  that rendering a message list without `add_generation_prompt` produces a
  token sequence that is an exact prefix of rendering the same messages
  with it. Verified against the fake tokenizer in `tests/`, not against
  Qwen2.5-7B-Instruct's real tokenizer.
- `steer.select_steer_features`'s quantile spread assumes firing density is
  a reasonable proxy for "qualitatively different feature" -- it has not
  been checked against, e.g., manual inspection of what each selected
  feature actually fires on.
- `screen.capture_mean_residual` mean-pools the layer's residual stream
  over *every* generated token uniformly; it does not weight by token
  salience or exclude low-information tokens (e.g. punctuation), and this
  has not been compared against alternatives (last-token embedding,
  attention-weighted pooling) on the real model.
- The persona-vector positive controls assume `positive_system_prompt` /
  `negative_system_prompt` induce a large enough behavioral difference
  on the smoke scenarios that the resulting mean-residual difference is a
  meaningful "trait direction" rather than noise; this is the published
  construction (Chen et al. / Arditi et al. / Sleight et al.) but has not
  been validated against Qwen2.5-7B-Instruct specifically, e.g. by reading
  the actual positive/negative generations for topical on-persona content.
- A control's mini dose-coherence sweep (`screen.sweep_control_doses`)
  treats the persona vector's raw magnitude as directly dose-scalable
  (`dose * persona_vector`), unlike a feature's `dose * max_activation *
  unit_decoder_direction` -- reasonable since the persona vector already
  has a natural residual-stream scale (it's a difference of real
  activations), but the two are not guaranteed to be on a comparable
  absolute scale to each other or to a feature's steering vector at the
  "same" dose value.
- The screen stage roughly doubles generation volume versus calibrate
  alone (baselines + features + controls + randoms, each across
  `screen_scenarios` scenarios, plus each control's own dose sweep); on
  the smoke config this is on the order of ~300 additional generations
  at `max_new_tokens=256`, comparable to or larger than the calibrate
  stage's own 65-90 minutes (see the runtime table above, which predates
  this stage and does not include it). Raising `config.random_directions`
  from 2 to 20 (both configs, since visual run 1) adds another
  `random_directions x screen_scenarios` generations on top of that.
  `screen_scenarios` and `random_directions` are the knobs to turn down if
  this dominates wall-clock time in practice.
