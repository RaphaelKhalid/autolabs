# Literature map for the five public protocols

This is a focused reading list, not a systematic review. The papers motivate constructs and failure modes; none of them validates the exact AutoLabs protocol by itself.

## Explanations and faithfulness

- Turpin et al., *Language Models Don’t Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting* ([arXiv:2305.04388](https://arxiv.org/abs/2305.04388)). Shows that an explanation can be systematically influenced by an unmentioned prompt factor. It motivates an intervention-based test, but the work should not be read as proof that every visible rationale is a causal trace.
- Lightman et al., *Let’s Verify Step by Step* ([arXiv:2305.20050](https://arxiv.org/abs/2305.20050)). Demonstrates process-oriented verification for mathematical reasoning. It motivates checking intermediate evidence, while also reminding us that a verifier can fail if the target process is not the target outcome.
- Anthropic, *Measuring faithfulness in Chain-of-Thought reasoning* ([research page](https://www.anthropic.com/research/measuring-faithfulness-in-chain-of-thought-reasoning)). Provides a useful intervention framing for asking whether explanations track causes. AutoLabs narrows that idea to visible API rationales and avoids calling them private cognition.
- Anthropic, *Reasoning models don’t always say what they think* ([research page](https://www.anthropic.com/research/reasoning-models-dont-say-think)). Reports evidence that stated reasoning can omit or misrepresent influential considerations. It is directly relevant to separating explanation quality from causal faithfulness.

## Tools and evidence channels

- Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* ([arXiv:2210.03629](https://arxiv.org/abs/2210.03629)). Establishes the interleaving of reasoning and tool actions as a useful agent pattern. For AutoLabs, it motivates freezing the tool transcript rather than treating an uncontrolled live tool as the treatment.
- Schick et al., *Toolformer: Language Models Can Teach Themselves to Use Tools* ([arXiv:2302.04761](https://arxiv.org/abs/2302.04761)). Motivates studying tool-mediated evidence, but its training and tool-use setup differs from a small monitorability event.

## Monitoring and representations

- Zou et al., *Representation Engineering: A Top-Down Approach to AI Transparency* ([arXiv:2310.01405](https://arxiv.org/abs/2310.01405)). Provides background for intervention and representation-based transparency. It does not justify inferring an internal representation from an API-only monitor.
- Turner et al., *Activation Addition: Steering Language Models Without Optimization* ([arXiv:2308.10248](https://arxiv.org/abs/2308.10248)). Motivates activation interventions and is background for the locked GPU track, not an implementation basis for the public API event.
- *Fantastic Reasoning Behaviors and Where to Find Them: Unsupervised Discovery of the Reasoning Process* ([arXiv:2512.23988](https://arxiv.org/abs/2512.23988)). Relevant to monitorability and unsupervised discovery, but it should be treated as recent background whose findings remain scoped to its own models and measurements.

## Locked Experiment 3A background

- *Persona Vectors: Monitoring and Controlling Character Traits in Language Models* ([arXiv:2507.21509](https://arxiv.org/abs/2507.21509)). Relevant to persona-related directions and the distinction between prompting, representation measurement, and intervention. It does not turn the public five-protocol event into an SAE experiment.

## How to read this literature responsibly

The common lesson is methodological: a monitor’s confidence or an explanation’s plausibility is not the same as causal faithfulness. AutoLabs therefore uses known interventions, paired controls, fixed observation surfaces, and separate monitor-quality checks. The event is a small reproducibility exercise designed to make those distinctions visible.

