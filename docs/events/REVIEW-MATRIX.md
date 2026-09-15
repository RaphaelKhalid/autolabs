# Five-protocol review matrix

| Protocol | What is manipulated | Ground truth | Recommended contract | Current decision |
| --- | --- | --- | --- | --- |
| 01 cue/explanation | cue present vs matched neutral | known cue plus paired answer change | 24 pairs; 48 actor; 48 monitor; 24 audits; 120 calls | Review primary estimand and rubric |
| 02 tool/evidence | fixed user wrapper vs fixed tool-result wrapper | identical evidence payload and case key | 36 actors; 36 monitors; 72 calls | Freeze wrapper; correct old 144 count |
| 03 shorter rationales | 128/64/32 visible-rationale caps | case key plus actual output/truncation | 180 actors; 60 bundled monitors; 240 calls | Freeze bundled-monitor surface |
| 04 optimization | not defined yet | not available until artifacts/checkpoints exist | evaluation envelope 120+120, construction extra | HOLD |
| 05 influence/plausibility | answer-only/rationale-only/combined surface | known cue intervention and controls | 36 actors; 108 monitors; 144 calls | Freeze surfaces; correct old 216 count |

## What “defensible” means here

A protocol is defensible when another reader can reconstruct the treatment, control, ground-truth label, call count, primary estimand, missingness rule, and exact limits before seeing any result. It does not require a large enough sample to settle a general scientific question. These are small, exploratory public studies; their first job is to make the receipt honest and reproducible.

