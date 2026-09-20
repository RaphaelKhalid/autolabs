# Experiment 3C publication audit

Date: 20 September 2026. Scope: the saved results of run `persona-3c-ded5477e-a5c3-4372-a776-91a0357293e1`, the checked-in analysis code, the manuscript, and the public presentation. This audit does not alter experiment records, thresholds, or classifications. No model generation or paid evaluation was performed.

## Assessment

The experiment completed a substantial SAE training, candidate selection, intervention, screening, and automated description pipeline. Its recorded screening and naming counts are reproducible from the saved outputs. However, the two named directions selected by the strictly unsupervised ranking arm are confounded by response collapse or linguistic degradation. The present evidence therefore supports a method for identifying behavior-changing candidate directions, but does not establish the stronger claim that coherent, previously unspecified personas were discovered without persona prompts.

This is a qualification of the scientific conclusion, not a claim that an SAE cannot discover such directions. The result identifies concrete evaluation failures that a controlled follow-up can address.

## Primary evidence

- [Candidate selections][candidates]
- [Screen metrics][screen]
- [Screen generations and paired unsteered responses][generations]
- [Blinded judge responses and naming summaries][describe]
- [Calibration records][calibration]
- [Post-training check][posttrain]
- [Full-power configuration](../research/experiment-003c/pipeline/configs/full-power.json)
- [Earlier full-1 summary](../research/experiment-003c/smoke-runs/full-1/summary.json)
- [Earlier full-1 judge output](../research/experiment-003c/smoke-runs/full-1/describe_results.json)

The configuration and reporting files identify a 100-million-assistant-position-token training run with 32,768 SAE features, average sparsity 40, and Qwen2.5-7B-Instruct at layer index 19. The complete original training checkpoint and source-token accounting were not independently revalidated in this audit. Repeated optimization steps per buffer mean that the token budget should not be described as 100 million unique examples or as the number of optimizer presentations.

## Verified counts

| Quantity | Recorded result |
| --- | ---: |
| Candidate features | 128 |
| Signed feature directions screened | 256 |
| Screen scenarios | 24 |
| Random screen directions | 20 |
| Supervised control axes | 3 |
| Feature directions passing diagnostic G0 | 79 / 256 |
| Features with at least one sign passing G0 | 60 / 128 |
| Control axes with at least one sign passing G0 | 2 / 3 |
| Feature directions selected for description | 40, representing 35 feature IDs |
| Random directions selected for description | 3 |
| Judge pairs queued | 2,064 |
| Judge responses in the saved result | 2,063 |
| Directions satisfying the naming criteria | 10, representing 9 feature IDs |

Every named feature direction also passes G0 in the saved results. Feature 2036 is named in both signs, so neither "10 features" nor "10 independent persona axes" is an accurate description of the count. Each feature direction has 48 judge responses from 24 pairs presented in both orders. These are not 48 independent scenarios. One random direction has 47 returned responses rather than 48.

| Ranking arm | Candidate features | Features passing G0 | Signed directions judged | Named signed directions |
| --- | ---: | ---: | ---: | ---: |
| Assistant-specificity composite, unsupervised | 48 | 25 | 17 | 2 |
| Density quantile | 32 | 10 | 5 | 0 |
| Prompt shift | 48 | 25 | 18 | 8 |

The arms are selected sequentially without overlap and are not randomized groups. Different numbers of directions reach the naming stage. These counts do not by themselves establish relative discovery efficiency.

## Named directions and quality

The phrases below are representative descriptions of a difference between paired responses. They do not necessarily describe a property increased by steering. "Polarity agreement" is agreement about which response has more of the property within the largest description cluster. It is not the fraction of all judgments independently assigning the same label. The final column reports the pipeline's mechanical coherence proxy, whose failure on feature 148 is documented below.

| Feature sign | Arm | Representative judge phrase | Largest cluster / 48 | Polarity agreement | Residual AUC | G0 consistency margin | Proxy pass / 24 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 89 negative | Prompt shift | more empathetic and conversational | 34 | 0.971 | 0.983 | 0.244 | 14 |
| 148 positive | Unsupervised | engaged and supportive | 36 | 1.000 | 1.000 | 0.726 | 23 |
| 241 negative | Prompt shift | formal, academic tone | 27 | 1.000 | 1.000 | 0.464 | 21 |
| 362 negative | Prompt shift | empathetic and personally supportive | 26 | 1.000 | 0.993 | 0.131 | 16 |
| 586 positive | Prompt shift | enthusiastic and personally affirming | 34 | 0.971 | 0.993 | 0.194 | 24 |
| 709 positive | Prompt shift | Playful, humorous tone | 42 | 0.952 | 1.000 | 0.428 | 19 |
| 717 positive | Unsupervised | composed and confident | 26 | 0.962 | 1.000 | 0.338 | 8 |
| 931 positive | Prompt shift | formal and self-explanatory | 24 | 0.958 | 0.988 | 0.270 | 24 |
| 2036 positive | Prompt shift | formal, academic tone | 26 | 0.885 | 1.000 | 0.330 | 24 |
| 2036 negative | Prompt shift | friendly, conversational warmth | 33 | 1.000 | 0.997 | 0.320 | 14 |

### Feature 148: response collapse

All 24 positive-steering responses contain only periods: 23 are exactly `.` and the self-description response is `....`. All 48 judge responses assign their described property more strongly to the unsteered baseline. Thus, the engagement/supportiveness label reflects the difference between a substantive baseline answer and an effectively absent steered answer.

The 23 single-period outputs pass the implemented coherence proxy. [The predicate in steer.py](../research/experiment-003c/pipeline/steer.py) checks only token diversity, maximum repeated-token run, and repeated four-gram fraction. It does not require substantive text, grammaticality, relevance, or a minimum response length. The recorded log probability is not part of this predicate. The 23/24 proxy pass rate is consequently not evidence of meaningful response coherence.

The saved screen entry begins at line 283 of [screen_records.json][screen]. The first paired generation begins near line 5043 of [screen_generations.json][generations]. The first judge response is near line 1737 of [describe_results.json][describe].

### Feature 717: linguistic degradation

Positive-steering outputs show grammatical omissions and wording errors, including `It great to hear that your friend considering` and `Hello! I Qwen an artificial intelligence`. Only 8 of 24 outputs pass even the existing proxy. Across all 48 judge responses, 38 locate the described property in the baseline and 10 in the steered response. These counts cover all description clusters, not just the dominant cluster.

This does not show that feature 717 lacks any persona relevance. It shows that the reported composure/confidence interpretation has not been separated from changes in linguistic fluency and presentation quality.

The saved screen entry begins at line 2243 of [screen_records.json][screen].

## Methodological corrections

### Unsupervised learning and prompt-informed selection

SAE reconstruction training does not use persona labels. The assistant-specificity and density-quantile selection arms also avoid persona-prompt scores. The prompt-shift arm uses six named-trait control prompts and twelve hand-written persona-style prompts. The latter explicitly mention warmth, playfulness, formality, enthusiasm, and other characteristics. See [config.py](../research/experiment-003c/pipeline/config.py) and [rank.py](../research/experiment-003c/pipeline/rank.py).

The prompt-shift arm can identify features without prescribing a final judge label for each one, but it is not independent of persona descriptions. Eight of the ten named signed directions came from this arm. The statement that no trait or description was specified anywhere in the experiment is false.

### G0 and controls

G0 requires residual-view AUC strictly greater than the largest AUC among the twenty random directions, 0.5217013889, and a consistency margin strictly greater than 0.1. The lexical AUC is reported but is not a G0 requirement. The consistency reference is a pool of other feature, control, and random-direction differences, not exclusively random-direction differences.

The sycophantic control passes in both signs; the hallucinating control passes in its positive sign. The evil control passes neither sign. This provides a useful diagnostic on these particular interventions, not comprehensive validation of persona specificity.

The checked-in [run_smoke.py](../research/experiment-003c/pipeline/run_smoke.py) explicitly describes G0 as diagnostic only at line 1053. `stage_describe`, at line 911, passes all screen records to the description stage. [describe.py](../research/experiment-003c/pipeline/describe.py) selects the highest-AUC directions and does not require successful positive controls. The claim that this implementation automatically withholds all naming unless controls pass is unsupported. The earlier run's empty judge output does not, on its own, establish why no judgments were recorded.

### Perturbation scales and null selection

[screen.py](../research/experiment-003c/pipeline/screen.py) constructs feature interventions as `sign * dose * max_activation * decoder_direction` at line 722. Random interventions are `median_nominal_dose * unit_random_direction` at line 773. Decoder rows are normalized by [sae.py](../research/experiment-003c/pipeline/sae.py). The random screen dose is 1.0. Feature 148 positive has an approximate perturbation norm of 190.46; feature 717 positive has an approximate norm of 49.73. The random screen perturbations have norm 1.0.

Consequently, the screen's random comparisons are not matched in intervention magnitude to these features. The calibration stage does scale its separate random controls by a median feature activation, but that scaling is absent from the screen stage. An observed screen AUC advantage cannot therefore be attributed solely to direction quality.

The description stage selects the three random directions with the lowest residual AUC. It compares these with feature directions selected for high AUC. This asymmetry limits the interpretation of the description null as a general false-positive benchmark. Neither stage supplies a calibrated significance level or correction for testing many directions.

### Description criterion

[describe.py](../research/experiment-003c/pipeline/describe.py) clusters descriptions with `sentence-transformers/all-MiniLM-L6-v2`, average-linkage clustering with cosine-distance threshold 0.7, and an additional keyword-overlap merge. A named direction requires a dominant cluster covering at least half of described judgments, polarity agreement of at least 0.8, and a strict majority of speaker-related judgments within that cluster.

The naming score is the dominant-cluster fraction multiplied by polarity agreement. The largest random score is 0.175, giving a threshold of 0.275 after the configured margin of 0.1. All ten named directions exceed this threshold and the separate naming criteria. For example, feature 241 positive has score 23/48 = 0.479, above the null threshold, but is not named because its dominant cluster contains fewer than half the descriptions. It did not narrowly miss the null threshold.

The Worker uses `gpt-5.6-luna` as the description judge. Its instructions explicitly separate speaker, content, and format, but also instruct it not to judge quality or correctness. See [persona-3c.ts](../orchestrator-worker/src/persona-3c.ts). A separate response-quality assessment is therefore necessary.

### Reconstruction and evaluation scope

The saved [post-training check][posttrain] reports held-out FVE 0.7171358 and mean L0 40.0. The reconstruction splice test reports argmax agreement 0.779661 over 118 assistant positions, replacement FVE 0.653295, and cross-entropy increase 0.565767 nats. The check is marked `ok: false` against its 0.90 agreement threshold. The rounded agreement is 78%, and the test is a small fixed-conversation diagnostic rather than a broad capability-preservation evaluation.

The training curve copied into the publication is training-batch FVE: [run_smoke.py](../research/experiment-003c/pipeline/run_smoke.py), line 733, reads the latest optimization batch's result. Its approximately 0.75 endpoint should not be labeled held-out FVE or substituted for the separate 0.717 estimate. The saved evidence does not establish stability across independently retrained SAE seeds.

Doses are selected using a 75% proxy pass threshold on four calibration scenarios, with a fallback dose when none qualifies. These four scenarios are included in the twenty-four-scenario screen. The subsequent classifier holds out scenarios internally, but calibration and final screening are not wholly disjoint. The larger run also changed candidate counts, generation limits, dose choices, and random-direction count. Improvement in control results cannot be attributed exclusively to scenario count, and no formal statistical power analysis was identified.

## Relationship to Chen et al. (2025)

Section 8 proposes that SAEs could enable "unsupervised discovery of persona-relevant directions" and discusses the restriction imposed by prompt-based trait extraction. Appendix M already analyzes SAE features selected by similarity to specified persona vectors, including intervention tests. The relevant distinction for Experiment 3C is candidate discovery without such trait-conditioned selection, not the first use of SAEs for persona-related steering. [Chen et al., version 1, sections 8 and M](https://arxiv.org/html/2507.21509v1).

Experiment 3C directly investigates that proposed extension, but the two strictly unsupervised named candidates do not currently provide an unconfounded demonstration. Prompt-informed candidates provide useful evidence of behaviorally interpretable SAE interventions. They do not resolve the same selection-independence question.

Prompt reachability was omitted from the reported full-power experiment and remains future work. Activation steerability does not imply that prompting can reproduce the intervention. The current experiment establishes neither prompt reachability nor prompt unreachability, and provides no estimate of prompt-elicitation difficulty.

## Publication claim corrections

| Earlier claim | Evidence-faithful replacement |
| --- | --- |
| Ten discovered personas or ten SAE features | Ten signed directions from nine features met the automated naming criteria. |
| No trait was specified at any point | SAE training and two selection arms avoid persona labels; the prompt-shift arm uses persona prompts. |
| Two unsupervised discoveries establish the conjecture | Two unsupervised directions meet the criteria, but response collapse and degraded language confound their interpretation. |
| The controls validate the complete method | Two of three control axes meet the diagnostic screen criterion. |
| The pipeline automatically withheld naming when controls failed | The checked-in implementation reports control outcomes but does not use them as a naming interlock. |
| 100% agreement means all judges assigned the same property | Polarity agreement is measured within the largest description cluster; cluster coverage is reported separately. |
| The SAE preserves 77% of predictions | A 118-position splice diagnostic has 77.97% argmax agreement and fails its 90% warning threshold. |
| The plotted FVE is held out | The plotted values are training-batch FVE; the separately recorded held-out estimate is 0.717. |
| A steerable direction is almost certainly prompt inducible | Prompt inducibility and its difficulty were not evaluated. |

## Recommended follow-up

Preserve the original classification and report the quality failures explicitly. A new evaluation should reject punctuation-only and nonresponsive outputs, assess language quality separately from persona judgments, match random perturbations by norm and response quality, and retain which side of each comparison exhibits the named property. Candidate selection and dose calibration should precede evaluation on fresh scenarios. Human review or a separately validated evaluator should test whether candidate labels survive controls for response length, grammar, formatting, and topic.

A decisive extension of Chen et al. would then be at least one coherent persona-relevant effect, selected without persona-prompt information and confirmed on held-out scenarios under these controls. This audit recommends that experiment; it does not change the current protocol or initiate another run.

[candidates]: ../research/experiment-003c/smoke-runs/full-power-1/_hf/runs/persona-3c-ded5477e-a5c3-4372-a776-91a0357293e1/outputs/candidates.json
[screen]: ../research/experiment-003c/smoke-runs/full-power-1/_hf/runs/persona-3c-ded5477e-a5c3-4372-a776-91a0357293e1/outputs/screen_records.json
[generations]: ../research/experiment-003c/smoke-runs/full-power-1/_hf/runs/persona-3c-ded5477e-a5c3-4372-a776-91a0357293e1/outputs/screen_generations.json
[describe]: ../research/experiment-003c/smoke-runs/full-power-1/_hf/runs/persona-3c-ded5477e-a5c3-4372-a776-91a0357293e1/outputs/describe_results.json
[calibration]: ../research/experiment-003c/smoke-runs/full-power-1/_hf/runs/persona-3c-ded5477e-a5c3-4372-a776-91a0357293e1/outputs/calibration_records.json
[posttrain]: ../research/experiment-003c/smoke-runs/full-power-1/_hf/runs/persona-3c-ded5477e-a5c3-4372-a776-91a0357293e1/outputs/post_train_check.json
