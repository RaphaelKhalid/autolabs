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
   in `feature_stats.json`. Once `feature_stats.json` is written,
   best-effort uploads `sae.safetensors`, `feature_stats.json`, and the
   run's config to a private Hugging Face model repo under
   `runs/<run_id>/` (`run_smoke.upload_run_artifacts`) -- a no-op unless
   both the `HF_TOKEN` env var is set and `config.hf_upload_repo` is
   non-empty (`""` for smoke, i.e. disabled; full's default is
   `RaphaelRaphaelRaphael/autolabs-3c-sae`), and any failure (auth,
   network, rate limit) is logged and swallowed, never aborting the run.

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
   calibrate/screen cover, as three disjoint arms (`"unsupervised"`,
   `"quantile"`, `"shift"`). Reports one record per selected candidate to
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
8. **reach** -- see "Reach stage" below. Disabled when `describe` produced
   nothing or `reach_max_directions <= 0`; otherwise reports one summary
   record per evaluated direction to stage `"judge"` (recordId prefix
   `"reach-"`).
9. **analyze / done** -- computes embedding-free proxies (normalized
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

### Claim under test

The paper this experiment extends lists three limitations of
prompt-based persona-vector discovery: (1) it is *supervised* -- the
target trait has to be specified in advance; (2) it needs a *precise
natural-language description* of that trait; (3) it needs the trait to be
*inducible by prompting* at all. Experiment 3C's headline claim is
(1)+(2): unsupervised discovery of persona-relevant SAE directions with no
target trait specified and no natural-language trait description
required anywhere in the discovery pipeline.

Clause (3) -- prompt reachability -- is *not* a gate on that finding.
Instead it is a measured **outcome**, reported per selection arm (below),
because on Qwen the paper's own observation is that most traits worth
discovering are prompt-inducible anyway; a direction that the unsupervised
arm finds but that turns out *not* to be prompt-reachable would be an
interesting result in its own right, not evidence the pipeline failed.

**The problem this fixes.** Before this change, the rank stage picked 75%
of screen candidates (`rank_shift_fraction`, now deprecated/unused -- see
below) by activation shift across a set of hand-written persona system
prompts. That selection is useful as a *positive control* (it tells you
whether shift-based targeting works at all), but using it for most of the
candidate pool biases the pool as a whole toward prompt-reachable
directions -- which contaminates the (1)+(2) claim, since a result built
mostly from prompt-shift-selected features never actually tests discovery
*without* a prompt. The fix is three disjoint selection arms, only one of
which uses any prompt-derived signal.

### Three arms

The full run screens `sum(config.arm_sizes.values())` (256: 96 + 64 + 96)
of the 32,768 trained SAE features; smoke screens 8 (3 + 2 + 3). Three
independent pieces of label-free signal feed the arms, each described
below, then `rank.rank_candidates` filters and splits the surviving
candidates into the three arms.

**1. Persona-context shift (`"shift"` arm's signal -- prompt-derived).**
The stage builds a diverse set of persona-style system-prompt "contexts"
-- `config.control_prompts`' `positive_system_prompt`/`negative_system_
prompt` pairs (6 contexts for the shipped 3 controls) plus
`config.context_prompts` (12 more neutral, hand-written prompts spanning
traits, e.g. "You are warm and encouraging.", "You are terse and
clinical.", "You are playful and lighthearted.") -- and, for every context
x `scenario` (the calibrate stage's `config.steer_scenarios` slice),
generates a reply under that system prompt
(`screen.generate_with_system_prompt`), captures the layer's residual
stream at the generated assistant tokens, and encodes it with `sae.encode`
(the same post-batch-topk sparse codes `forward_loss` trains on),
mean-pooling the codes over generated tokens (`rank.capture_context_
activations`). `rank.compute_shift_stats` (pure numpy/Python, unit tested)
then treats each context as an ANOVA group and each of its scenarios'
per-context mean activation as one observation in that group, computing
per feature:

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

This is the **positive control** for the whole comparison, not part of the
unsupervised claim: the contexts are diverse but *chosen* -- 3
literature-style trait pairs plus 12 hand-written prompts, not sampled
from any principled distribution over "ways a system prompt could
differ" -- so a feature reliably shifting across these 18 contexts is
good evidence the shift-based-targeting *mechanism* can find something,
while telling you nothing about whether unsupervised discovery works.
It's also, by construction, biased toward prompt-reachable directions --
exactly the property clause (3) says isn't required -- which is why this
arm must never be more than a minority, clearly-labeled slice of the
candidate pool.

**2. Assistant-specificity / breadth / topic-invariance (`"unsupervised"`
arm's signal -- never sees a prompt).** A small extra harvest pass
(`rank.capture_specificity_activations`, `config.rank_specificity_
conversations` ~300 conversations from the same dataset stream
`harvest.py` trains on) captures, for every conversation, the SAE codes at
*every* real token position (a FULL token mask, unlike training's
assistant-turn-only mask), then uses `harvest.compute_assistant_mask` only
to *label* each position as assistant-turn content or not -- every other
real position (user-turn content, chat-template control tokens) is the
"user" side of the contrast. No system prompt, no trait, no persona-style
context of any kind is involved anywhere in this pass; it is a plain read
of how the base model already behaves on ordinary chat data.
`rank.compute_specificity_stats` (pure numpy/Python, unit tested) then
computes per feature, from these per-conversation aggregates:

- `assistant_specificity`: mean-over-conversations assistant-token mean
  activation, divided by the same for user-token mean activation (plus a
  small epsilon) -- how much more the feature fires *as a response* than
  as a function of whatever the user just said.
- `assistant_specificity_fire_rate`: the analogous ratio using mean
  firing *rate* instead of activation magnitude (reported, not
  composited -- see below).
- `breadth`: fraction of conversations where the feature fires at least
  once on an assistant-turn token.
- `topic_invariance`: `1 - between-conversation variance of each firing
  conversation's own mean activation, divided by that variance plus the
  mean within-conversation variance` (an ANOVA-style decomposition, scoped
  to conversations where the feature fired at least once) -- high when
  the feature behaves similarly regardless of what the conversation is
  about, low when it is bound to specific topics/conversations.

**3. Structural properties (prompt-free, shared across arms).** Firing
density and liveness (`rank.is_dead`: `firing_density <= 0`, i.e. never
fired during the density-tracking window) from `feature_stats.json`
gate every candidate; `shell` (`rank.feature_shell`: the smallest
`config.matryoshka_shells` boundary containing the feature index, e.g.
1024/4096/8192) records which matryoshka shell it lives in.

**Filters before arm assignment** (`rank.rank_candidates`): firing density
in `[config.firing_density_min, config.firing_density_max]`, not dead, a
computed shift-stats entry, and a computed specificity-stats entry.

**Composite score** (`rank.compute_composite_scores`, pure numpy/Python,
unit tested), computed for every surviving candidate and used only by the
`"unsupervised"` arm:

```
composite = z(log assistant_specificity) + z(breadth) + z(topic_invariance)
            + shell_bonus(shell_index)
```

where `z(...)` is a population z-score against the candidate pool itself,
and `shell_bonus` is 1.0 for the innermost matryoshka shell, 0.5 for the
second, 0.0 otherwise (inner shells are preferred since the matryoshka
training objective forces them to carry a self-contained coarse
reconstruction rather than only refining an outer one, making them the
more likely place for a coarse, generalizable direction to live -- the
same rationale the old shift-score shell tie-break used). **No
prompt-derived quantity (`shift_f`/`shift_maxdiff`) enters this score at
all** -- that is the entire point of this arm.

**Selection** (`rank.rank_candidates`), filled in order so the three arms
are always disjoint, each capped to whatever remains in the pool when its
turn comes:

- `"unsupervised"` (`config.arm_sizes["unsupervised"]`, 96 full / 3
  smoke): the top slice by composite score, ties broken by feature index,
  from the *entire* filtered candidate pool. This is the arm that actually
  tests (1)+(2): no trait was named, no natural-language description was
  used, and no system prompt was involved in scoring it.
- `"quantile"` (`config.arm_sizes["quantile"]`, 64 full / 2 smoke): a
  density-quantile spread (`steer.quantile_indices`, the same mechanism
  `steer.select_steer_features` uses) over the pool remaining after the
  unsupervised arm's picks are removed -- unbiased by construction, a
  distribution-matched comparison for how much either scored arm actually
  helps once the describe stage's judge results come back.
- `"shift"` (`config.arm_sizes["shift"]`, 96 full / 3 smoke): the top
  slice by `shift_f` descending (ties broken by preferring inner shells,
  then feature index) from whatever's left after the other two arms --
  the positive control described above.

Every selected candidate is written to `candidates.json`'s
`screen_features` list carrying every computed statistic (`density`,
`shell`, `shift_f`, `shift_maxdiff`, `assistant_specificity`,
`assistant_specificity_fire_rate`, `breadth`, `topic_invariance`,
`composite`) plus `arm` (`"unsupervised"`/`"quantile"`/`"shift"`),
`selection` (kept equal to `arm`, for callers written against the field
name the old two-arm schema used), and `rank_within_arm` (1-indexed
position within that arm's own selection order), and reported to stage
`"train"` with `recordId = f"rank-{feature}-{arm}"` -- `"train"`, not a
dedicated `"rank"` stage, because `orchestrator-worker/src/persona-3c.ts`'s
`PERSONA_3C_STAGES` enum does not include one and this change is meant to
land without a Worker change.

`config.rank_shift_fraction` and `config.screen_features` (the old total
budget) are now unused, kept only so an old saved `run_config.json` still
loads (`Config.from_dict` rejects unknown keys); the real budget is
`sum(config.arm_sizes.values())`.

**Arm propagation.** `arm` (and every statistic above) rides along through
the rest of the pipeline for reporting: `steer.select_steer_features`
carries `arm` from `candidates.json` onto its selected-feature dicts (a
run calibrating without a rank-stage candidate list gets `arm: None`),
`steer.run_calibration` writes it onto every steered record (its own
random-direction dose-sweep controls get `arm: "random"`);
`screen.extract_feature_info` reads it back off calibration records so
`screen.run_screen` can attach it to every feature *and* generation
record (positive controls get `arm: "control"`, random-direction nulls
get `arm: "random"`); `describe.attach_arm` attaches it to every
describe-stage per-direction summary by direction key. The screen table
in `smoke-report.html` has an `arm` column, its verdict reports per-arm
G0 pass counts, and `describe-report.html` reports per-arm named counts
-- see "Screen stage"/"Describe stage" below.

**Wiring.** `steer.select_steer_features` takes an optional
`explicit_features` argument; when `run_smoke.stage_steer` finds
`candidates.json`, it passes `candidates["screen_features"]` through
`steer.run_calibration` to `select_steer_features`, which then uses that
list as-is (including its `arm` tags) instead of doing its own
density-quantile selection. When `candidates.json` doesn't exist (e.g. a
workdir from before the rank stage existed, or a config that skips it)
calibrate falls back to the previous behavior: density-quantile selection
sized off `config.steer_features` directly, with every selected feature's
`arm` set to `None`.

**Bias.** The shift arm's contexts are diverse but *chosen*, not sampled
from any principled distribution -- a feature whose persona-relevant
direction only shows up under a context this set never tried scores low
on `shift_f` despite being exactly what the run is looking for, and
conversely a feature could shift reliably across these 18 contexts for a
reason that has nothing to do with persona (a topic or formatting
correlate of the specific wording chosen). The unsupervised arm's own
statistics have their own assumptions -- `assistant_specificity` and
`breadth` both assume "fires more/more often on assistant turns than user
turns" is a reasonable proxy for "persona-relevant", which has not been
validated against manual inspection of what the selected features
actually fire on. Neither arm, nor the quantile control, is a guarantee
that a selected feature is persona-relevant; the describe stage's blinded
judge is what actually says what a direction is.

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

Every direction's record (`{kind, id, sign, dose, arm, n_scenarios,
separability: {resid: {acc, auc}, lexical: {acc, auc}}, consistency:
{mean_cos, null_mean_cos}, coherent_fraction}`) is written to
`screen_records.json` and reported to stage `"screen"` with
`recordId = f"screen-{kind}-{id}-{sign}-{dose}"` (`sign` rendered as
`pos`/`neg`/`na`, matching the calibrate stage's own recordId convention).
`arm` is `"unsupervised"`/`"quantile"`/`"shift"` for a `kind="feature"`
row (propagated from the rank stage via `screen.extract_feature_info`,
`None` if calibrate ran without a rank-stage candidate list),
`"control"` for every `kind="control"` row, and `"random"` for every
`kind="random"` row -- see rank.py "Arm propagation". The HTML report's
screen table has an `arm` column, and its verdict line is followed by a
per-arm breakdown of feature entities passing gate G0 (either sign) for
each of `"unsupervised"`/`"quantile"`/`"shift"`
(`run_smoke.compute_screen_verdict`'s `arm_pass_counts`) -- the `"shift"`
arm's own pass count is itself a positive-control sanity check (it should
pass at least as often as an unbiased sample, since it was selected
precisely to shift under prompts), while the `"unsupervised"` arm's count
is the actual result this experiment cares about.

**Screen generations.** For every direction and every scenario it was
scored on, the steered text, the same-scenario baseline text, and that
generation's `finish_reason` and coherence dict are written as a flat list
of `{kind, id, sign, dose, arm, scenario, text, baseline_text,
finish_reason, coherence}` dicts to `screen_generations.json` (built by
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
  (direction-free) per direction.

  **Similarity backend.** A second live judge pass
  (`tests/fixtures/describe_results_validate2.json`) found the judge
  describes a *consistent* property in different words almost every time
  -- "warm, enthusiastic encouragement" / "interpersonally supportive" /
  "personally encouraging and emotionally enthusiastic" are the same
  judged property for the same direction -- and the original TF-IDF +
  hashed-lexical cosine view shares too little vocabulary across
  paraphrases like that to cluster them at all
  (`largest_cluster_fraction` ~0.12 for every direction, real or null
  alike, on that pass). Clustering now runs primarily on sentence
  embeddings: `sentence-transformers/all-MiniLM-L6-v2`, loaded lazily
  through plain `transformers` (`AutoTokenizer`/`AutoModel`, no
  `sentence-transformers` package dependency -- mean-pooled over token
  embeddings, L2-normalized, cached under `HF_HOME`, ~90MB), via
  `describe._embed_texts`/`describe._load_embedding_model`. If the model
  can't be loaded (no network, missing `transformers`/`torch`, etc.) it
  logs a warning once and falls back to TF-IDF vectors alone
  (`describe._tfidf_vectors`, lowercased/stopword-stripped
  unigrams+bigrams) for the rest of the run -- `describe.
  _vectors_and_backend` picks the backend and every direction's cluster
  summary records which one was actually used in `embedding_backend`
  (`"embedding"` or `"tfidf_lexical"`).

  Either way, average-linkage agglomerative clustering (`describe.
  _agglomerative_clusters`) merges the two closest clusters by average
  pairwise cosine distance until the best available merge exceeds
  `config.describe_cluster_threshold` (default 0.7 -- see "Calibration"
  below), followed by a fallback keyword-overlap merge
  (`_keyword_overlap_merge`, Jaccard >= 0.6 on stopword-stripped content
  words, independent of the similarity backend) for short descriptions
  the cosine view under- or over-weights.

  Reports per direction: `n_none`, `n_described`, `none_rate`,
  `largest_cluster_size`, `cluster_property` (the description with the
  highest mean similarity to the rest of the largest cluster -- its
  medoid), `centroid_property` (the member sentence closest to the
  largest cluster's mean *vector* -- a true centroid; usually but not
  always the same sentence as `cluster_property`), `cluster_examples` (up
  to 3 of the largest cluster's member sentences), `embedding_backend`,
  `direction_agreement` (fraction of the largest cluster's members whose
  unblinded `steered_has_more` matches that cluster's majority direction),
  `about_counts`, `confidence_counts`, `named` (largest cluster >=
  `config.describe_named_fraction` of described, default 0.5,
  `direction_agreement` >= 0.8, and a strict majority of that cluster is
  `about: "speaker"`), and `consistency_score` =
  `largest_cluster_size / n_described` * `direction_agreement`.

  **Calibration.** `describe_cluster_threshold` (0.7) and
  `describe_named_fraction` (0.5) were chosen by clustering
  `tests/fixtures/describe_results_validate2.json` (143 judge pairs from
  the second live pass: 6 feature directions, 3 `kind="random"` nulls)
  under the embedding backend at several thresholds and picking the point
  where real, consistently-described directions clear the >= 0.5
  `largest_cluster_fraction` bar while the null ceiling stays clearly
  below it. Results at threshold 0.7 (`largest_cluster_size / n_described`
  per direction; `null_consistency_max` is the actual `named_above_null`
  gate, not the raw fraction):

  | direction | fraction | direction_agreement | named_above_null |
  |---|---|---|---|
  | feature-1134-pos | 0.875 | 1.0 | yes |
  | feature-1134-neg | 0.688 | 1.0 | yes |
  | feature-466-pos | 0.562 | 0.778 | no (direction_agreement < 0.8) |
  | feature-2443-neg | 0.375 | 0.667 | no |
  | feature-1740-neg | 0.312 | 1.0 | no (fraction < 0.5) |
  | feature-466-neg | 0.312 | 0.6 | no |
  | random-9-na (null) | 0.400 | 0.667 | -- |
  | random-2-na (null) | 0.357 | 0.6 | -- |
  | random-1-na (null) | 0.308 | 0.75 | -- |

  Null ceiling (`null_consistency_max`) is 0.267 (`random-9-na`,
  `consistency_score = fraction * direction_agreement`), so
  `named_above_null_threshold` = 0.267 + 0.1 (`describe_null_margin`) =
  0.367. Two of the six judged feature directions (both signs of feature
  1134, the direction with the most literally-repeated paraphrasing in
  this pass) clear both `named` and that threshold comfortably
  (`consistency_score` 0.875 and 0.688 vs. 0.367); the rest describe a
  real but less textually consistent effect across scenarios (varying
  between "speaker" and "format" properties scenario to scenario) and
  correctly stay unnamed rather than being forced into one cluster. The
  nulls themselves cluster somewhat -- `null_consistency_mean` 0.237,
  `null_consistency_max` 0.267 -- which is reported honestly rather than
  hidden: the judge does describe some random directions with recognizable
  consistency (e.g. "more future-oriented"), so the gate that matters is
  the margin over that ceiling, not an absolute bar.
- `describe.attach_arm` then adds `arm` to every direction's cluster
  summary (looked up by direction key from the `directions` passed into
  `describe.run_describe`, i.e. the screen stage's own `arm` tags --
  `"unsupervised"`/`"quantile"`/`"shift"` for a feature, `"control"` for a
  positive control, `"random"` for a null), and
  `describe.compute_arm_named_counts` rolls up, per arm (excluding random
  nulls), how many of the judged directions came out `named` and
  `named_above_null` out of how many were judged. `describe-report.html`
  shows an `arm` column on the directions table and a "Named counts by
  arm" summary line built from these counts, so it's visible at a glance
  whether the unsupervised arm actually named anything, not just whether
  the shift arm (its positive control) did.

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

Writes `describe_results.json` (`{plan, results, clusters, null_stats,
arm_named_counts}`) and `describe-report.html` under `<workdir>`, and
reports a compact per-direction summary record (`describe-<directionKey>`,
including its `arm`) to the harness under stage `"judge"`.

## Reach stage

**This is a measured outcome, not a gate.** The paper this experiment
extends lists clause (3) of prompt-based persona-vector discovery's
limitations as "the trait must be inducible by prompting at all" -- see
"Claim under test" above. Experiment 3C's headline claim is (1)+(2)
(unsupervised discovery, no target trait, no natural-language
description); clause (3) is deliberately *not* required for that claim to
hold. A direction the unsupervised arm finds that turns out *not* to be
prompt-reachable would be an interesting result in its own right ("SAE
discovery finds something prompting can't reach"), not evidence the
pipeline failed -- so `reach.py` reports reachability per direction and
rolls it up per selection arm, alongside a required-to-pass positive
control, rather than gating anything on it.

For every direction the describe stage `named` (its largest judge-response
cluster covering >=50% of described pairs, >=0.8 direction agreement, and
a speaker-about majority -- see "Describe stage" above), plus every
`named` persona-vector control regardless of rank (`reach.
select_reach_directions`; controls are this test's *positive controls* and
are expected to come out reachable, so they are never dropped for the
cap), highest `consistency_score` first up to `reach_max_directions`
(smoke 4, full 12) non-control directions:

- Four system prompts are built **locally, with no judge call**, from
  that direction's own `cluster_property` text `P` (the blinded judge's
  clustered description of what differs about the steered condition --
  never a persona vocabulary chosen ahead of time): `direct` ("In your
  replies, {P}."), `rewrite` ("Adopt this manner throughout: {P}. Keep it
  natural and consistent."), `intensified` (`rewrite` + " Do this
  strongly and in every reply."), and `fewshot` (`rewrite`'s system
  prompt followed by two demonstration exchanges -- a screened scenario's
  prompt paired with that direction's own *steered* reply to it).
  `direct`/`rewrite`/`intensified` are generated (unsteered model,
  `generation.generate_batch`) on every scenario the direction was
  screened on; `fewshot` is generated only on that set **minus its own two
  demo scenarios** (`reach.select_fewshot_demo_scenarios`, deterministic:
  sorted scenario ids, first two) -- without this exclusion, fewshot would
  be "graded" on the same steered reply it was shown as an example,
  which would inflate its apparent reach for a reason that has nothing to
  do with whether the *system prompt* reproduces the effect.
- **Behavioral reach**: a fresh logistic classifier (`screen.
  train_logreg`, the same numpy-only mechanism the screen stage's
  leave-one-scenario-out split uses, but fit once on every scenario here
  -- there is no held-out fold to protect, since what's being scored next
  is a prompted reply, never one of the classifier's own training points)
  for both the residual (layer-19, mean-pooled over generated tokens) and
  lexical (hashed bag-of-words) views. `effect_fraction` rescales the
  prompted replies' mean classifier logit onto the baseline->steered
  scale (0 = no effect, 1 = matches steering), clipped to `[-0.5, 1.5]` so
  a wrong-direction or overshooting effect stays a finite, visible number;
  `axis_cosine` (residual view only) is the cosine similarity between the
  prompted-vs-baseline and steered-vs-baseline mean residual diff vectors,
  an orientation check independent of `effect_fraction`'s magnitude. The
  **best variant** is whichever maximizes the residual-view
  `effect_fraction` (nan-safe: a variant with no usable signal never
  wins).
- **Mechanistic reach**: for a feature direction, the mean activation of
  that feature's own SAE code (`sae.encode` on the layer's residual
  stream) at the generated tokens, rescaled the same baseline->steered
  way as `effect_fraction` (`feature_fraction`, unclipped -- an activation
  has no natural behavioral bound). A persona-vector control has no
  single feature to read, so its mechanism proxy is the projection onto
  that control's own steered-minus-baseline mean residual axis instead.
- **Judge check** (optional, `reach_judge`: off for smoke to save judge
  budget, on for the full run): for the **best variant only**, plans
  blinded prompted-vs-steered pairs (both orders, `directionKey =
  f"reach-{directionKey}"`) through the same Worker judge queue
  `describe.py` drives (`report.judge_plan`/`judge_run`/`judge_results`),
  and reports `judge_steered_share` -- the fraction of classifiable pairs
  the judge said the *steered* side showed more of `P` (0.5 means
  prompting reproduced it as well as steering did, from the judge's
  perspective). A plan the Worker refuses (judge budget/ceiling) or a
  drive/fetch that comes back empty leaves `judge_steered_share` as
  `null` rather than failing the direction -- these judge calls share the
  same budget describe.py's pairs already spent from.
- **Classification** (reported, not gating; thresholds in config):
  `reachable` if the best variant's residual `effect_fraction >=
  reach_effect_threshold` (0.7) and, only when a judge check actually ran,
  `judge_steered_share <= 0.65`; `not_reachable` if the best
  `effect_fraction < reach_not_threshold` (0.3) -- `fewshot` is already
  included in that max, so a direction only reachable via few-shot still
  counts as reachable, and one unreachable even with few-shot correctly
  counts as not reached; otherwise `partial`. `mechanism_same` is whether
  the best variant's `feature_fraction >= reach_feature_threshold` (0.5).
  A `nan` effect/feature fraction (no usable signal, e.g. a direction with
  ~zero baseline-vs-steered separation to rescale onto) never satisfies
  either comparison and falls through to `partial`/`mechanism_same=False`
  safely rather than crashing.

Per direction: `{directionKey, arm, property, variants: {direct, rewrite,
intensified, fewshot: {effect_fraction_resid, effect_fraction_lex,
axis_cosine, feature_fraction}}, best_variant, reachable_class,
mechanism_same, judge_steered_share}`. Writes `reach_results.json` (`
{directions, summary_by_arm}`) and `reach-report.html` under `<workdir>`;
the summary reports counts of `reachable_class` by arm and the **control
pass rate** (named controls that came out `reachable` / total named
controls) -- since controls are this reach methodology's own positive
control, a low control pass rate is evidence against the *methodology*,
not against any one direction. Reports a compact per-direction summary
record (`reach-<directionKey>`) to the harness under stage `"judge"` (no
dedicated Worker stage, same rationale as rank.py's `"train"` reuse).

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
| reach | `reach_results.json` (skipped entirely if describe produced nothing or `reach_max_directions <= 0`) |
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
| `HF_TOKEN` | Hugging Face token for `run_smoke.upload_run_artifacts` (see "Stages" step 2). Only read if `config.hf_upload_repo` is also non-empty; missing either one is a silent no-op, not an error. |

If `AUTOLABS_3C_WORKER_URL` is unset, `report.py` logs a warning and
continues -- the GPU job is never blocked on the harness being reachable.

## Tests

```sh
python -m pytest research/experiment-003c/pipeline/tests -q
```

185 tests, all CPU-only (99 in `tests/test_pipeline.py`, 22 in
`tests/test_rank.py` for the rank stage's shift/specificity/composite
statistics and three-arm selection, 19 in `tests/test_generation.py` for
batched generation and the batched steering hook, and 45 in
`tests/test_reach.py` for the reach stage's prompt construction, effect/
feature-fraction rescaling math, classification thresholds, and per-arm
summary, see below): SAE forward/loss and matryoshka-nested-loss-decreases
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
resume), `run_smoke.upload_run_artifacts` being a no-op (no filesystem or
network touched) both without `HF_TOKEN` set and without `config.
hf_upload_repo` configured, and (screen.py, see "Screen stage" below) the hashed lexical
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
set of directions, including the empty-baseline fallback and that an
`extra={"arm": ...}` entry propagates onto every generation's `arm` field;
`describe.attach_arm`/`describe.compute_arm_named_counts` wiring a toy
`directions` list's arm tags onto cluster summaries and rolling them up
per arm (excluding random nulls); and (`rank.py`, see "Rank stage" above)
the F-like shift-score helper on toy per-context activations (a feature
that shifts consistently between contexts scores much higher than one
that only has scenario-to-scenario noise, and single-context/zero-max-
activation edge cases are 0 rather than NaN/inf), `compute_specificity_
stats` on toy per-conversation aggregates (an assistant-specific, broad,
topic-invariant feature scores as expected; a topic-bound feature's
`topic_invariance` comes out low; zero conversations degrades to an
all-zero result rather than raising), `zscore`/`shell_bonus`/
`feature_shell`/`feature_shell_index`/`is_dead`, `compute_composite_
scores` favoring an assistant-specific broad feature over a topic-bound
one regardless of which has the higher `shift_f` (the composite must
never read a prompt-derived quantity) and not crashing on an
`assistant_specificity` of exactly 0, `rank_candidates`'s density/dead
filtering (now requiring both a shift-stats *and* a specificity-stats
entry) and its three-arm selection (arms disjoint and exactly sized, each
arm's own ordering -- composite descending for `"unsupervised"`, `shift_f`
descending for `"shift"` -- arm sizes correctly capped when the candidate
pool is smaller than the requested total, filled in order so
`"unsupervised"` claims the shortfall first, and the empty-candidate-pool
case), and `steer.select_steer_features`'s `explicit_features` bypass (an
explicit list is honored verbatim -- including features outside the
density window that would otherwise be filtered out, and its per-feature
`arm` tag riding through unchanged -- while `explicit_features=None`
still falls back to the original density-quantile selection with every
selected feature's `arm` set to `None`).

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

`tests/test_reach.py` (see "Reach stage" above): the four prompt
variants' text (`direct`'s wording, `rewrite`==`fewshot`'s shared system
text, `intensified` extending `rewrite`, an unknown variant raising) and
message shape (`direct`/`rewrite`/`intensified` are system+user,
`fewshot` is system + two full user/assistant demo turns + the eval
user turn), `select_fewshot_demo_scenarios`'s deterministic sorted-first-n
choice regardless of input order, and that a direction's `fewshot`
evaluation set is exactly its scenario set minus its own demo scenarios
(no overlap, union recovers the full set); `effect_fraction` on toy
logits -- exact match-steering (1.0), exact match-baseline (0.0), halfway
(0.5), clipping an overshoot to 1.5 and a wrong-direction effect to -0.5
(plus custom clip bounds), and `nan` on a zero baseline-vs-steered
denominator; `feature_fraction`'s analogous (unclipped) math and its own
zero-denominator `nan`; `axis_cosine` on toy vectors (parallel diff
vectors score 1.0, orthogonal ones 0.0); `pick_best_variant` (max
`effect_fraction_resid`, nan-safe, `None` when every variant is nan);
`classify_reachability`'s thresholds (`reachable` above the effect
threshold, `not_reachable` below the not-threshold, `partial` in between,
a judge check that still favors "steered" too strongly downgrading an
otherwise-`reachable` result to `partial`, and a `nan` effect falling
through to `partial` rather than raising); `mechanism_same`'s threshold
(including its own `nan` case); `select_reach_directions` (excludes
unnamed and null directions, orders by `consistency_score` descending,
caps non-control directions at `reach_max_directions`, and always
includes a *named* control beyond that cap while never force-including an
*unnamed* one); `summarize_by_arm`'s per-arm `reachable_class` counts,
control pass rate (and `None` when there are no named controls), and the
empty-input case; and a numpy-only sanity check that `fit_classifier` +
`classifier_logit` actually separate two well-separated toy clusters.

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
