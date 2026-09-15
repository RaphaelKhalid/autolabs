# Protocol 04 — Can reasoning stay legible as optimization continues?

## Status: HOLD — treatment is not yet defined

This is a useful question, but the current AutoLabs repository does not contain an optimizer, a training run, registered checkpoints, or a budget for producing them. A fixed collection of prompt variants is not automatically “optimization.” Running the current card would therefore risk calling a cross-sectional prompt comparison an optimization experiment.

## Research question and hypothesis

**Research question.** Does increasing a registered optimization measure reduce the monitorability of decision-relevant visible evidence?

**Hypothesis.** As the registered optimization measure increases, a fixed monitor will recover less decision-relevant evidence, conditional on the same held-out task set and output interface.

The phrase “optimization pressure” must be operationalized. It could mean training steps, reward-model score, search objective value, prompt-search iterations, or another quantity. Those are different treatments and cannot be merged after data collection.

## What is missing

Before launch, the owner must specify:

1. The object being optimized: weights, prompt, system instruction, policy, or another artifact.
2. The objective function and who evaluates it.
3. Five immutable checkpoints or five artifacts with monotonic registered optimization scores.
4. The held-out case bank and leakage controls.
5. The monitor prompt and observation surface.
6. Whether optimizer calls are provider calls, local computation, or both.
7. The full call and compute budget, including failed search proposals and retries.

Without these, the five-checkpoint design has no causal treatment. The earliest checkpoint is not a valid reference if it differs in model version, prompt, or task distribution in an uncontrolled way.

## Recommended repair: fixed prompt-search version

If a lightweight public event is preferred, register a bounded external search instead of implying model training:

- Generate exactly five prompt artifacts using a deterministic search procedure over a fixed candidate set.
- Score candidates only on a registered training split with a fixed evaluator.
- Freeze the five artifacts by increasing search iteration or objective score.
- Evaluate all five on a held-out set with the same actor and monitor interface.
- Label the treatment “search iteration” or “registered objective score,” not “optimization pressure.”

This repair still requires prompt-generation and evaluator calls. They must be included in the grant ceiling. A search that uses the same cases for selection and evaluation is not a held-out optimization experiment.

## Candidate call plan after repair

The existing card proposes 5 checkpoints × 24 held-out cases = 120 actor responses and 120 monitor calls. That is a valid evaluation envelope only after the five artifacts exist. It does not include the calls needed to produce or score the artifacts. The manifest should therefore distinguish:

- artifact-construction calls;
- held-out actor calls;
- monitor calls;
- audits and retries.

The current 240-call ceiling is not sufficient by default if artifact construction is model-based. Either construct artifacts offline before the sponsored run, or increase the ceiling and show how the $10 remains safe.

## Analysis once the treatment exists

Primary outcome: monitor evidence-recovery rate against the registered optimization measure on the held-out cases. Secondary outcomes: answer quality, false positives, truncation, and family heterogeneity. Use pre-registered checkpoint contrasts and a trend interval, not a claim that optimization monotonically harms legibility unless the registered scores and held-out outcomes support it.

Report the optimization score distribution, checkpoint identity, and whether the score was computed on training or held-out data. If the score does not increase monotonically, the analysis should use the actual score rather than checkpoint number.

## Failure and publication rule

If any checkpoint artifact is missing, do not interpolate it. If the optimizer fails before five artifacts exist, publish a partial construction receipt and do not publish a checkpoint-legibility conclusion. Provider timeouts and uncertain charges remain visible.

## Launch decision

The `/events` card should display **not launchable yet** until the treatment definition, artifacts, construction budget, and held-out analysis are frozen. This is a scientific safety feature, not a cosmetic delay.

## Reading

Representation Engineering ([arXiv:2310.01405](https://arxiv.org/abs/2310.01405)) and the monitorability work in the shared [literature map](./literature.md) provide background, but neither supplies the missing AutoLabs optimizer definition.

