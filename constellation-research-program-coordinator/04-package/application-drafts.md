# Application drafts

These are working drafts. Confirm every personal claim and rewrite them into your own voice before submitting.

## Why are you interested in this role at Constellation?

Constellation interests me because the work sits where research quality and program quality meet. This role asks someone to keep project plans, communications, fellow support, events, and feedback systems reliable while improving them as the program changes. That is the kind of work I naturally gravitate toward: turning a complicated, fast-moving effort into a legible system that people can trust.

I have been applying that approach in AutoLabs, a public observatory for agent experiments. For a question motivated by the Persona Vectors paper's limitation around unsupervised SAE features, I turned an open-ended idea into a bounded protocol with a stable study page, explicit research phases, live status, evidence links, and a hard approval gate before confirmation. I have kept the result scoped to the work completed and the questions still open.

I want to bring that combination of operational ownership, technical curiosity, and careful communication to Constellation's fellows and mentors. I would be excited to learn the field quickly, support researchers well, and improve the systems that let high-impact work happen.

## How did you become interested in AI safety? What AI risks are you most concerned about?

Working draft. Replace the first sentence with the candidate's actual origin story.

I became interested in AI safety by building and auditing agentic systems, where a system can optimize a measurable objective while drifting away from the thing people actually care about. The more I worked on reward compatibility, monitoring, and representation-level behavior, the more I became interested in failures that are easy to miss if we evaluate only final outputs.

I am most concerned about systems that develop broadly harmful behavior from narrow training or optimization pressure, especially when evaluations and monitors measure the wrong abstraction. That includes emergent misalignment, deceptive or strategically hidden behavior, dangerous capability generalization, and the governance gap created when technical systems move faster than the institutions responsible for deploying them.

I am drawn to safety work that makes these risks more legible and actionable. My current AutoLabs study asks whether sparse autoencoder features can expose fine-grained behavior that prompt-based persona extraction misses. I am treating it as a bounded empirical question, with held-out confirmation and explicit limits, rather than assuming an interpretable feature is automatically a real persona. That combination of curiosity and disciplined uncertainty is what I want to keep developing.

## Briefly describe a system or process you inherited. What issues did it have? What did you change, what did you leave alone, and why?

One system I inherited was a growing AutoLabs prototype used to present several research experiments. Its main weakness was that the public interface, experiment metadata, live status, and study-specific execution logic were too easy to conflate. That made it harder to tell which claims were historical, which protocols were planned, and what a public page was actually authorized to do.

I kept the existing experiment-specific runners and historical routes intact because changing scientific execution and archive identity at the same time would create unnecessary risk. I changed the surrounding operational layer: stable experiment descriptors, explicit status states, source-linked research handoffs, a read-only public observatory, separate live-status polling, and tests for malformed or premature promotion. For the SAE study, I also made confirmation a distinct approval-gated phase and stated clearly when no result existed.

The improvement was operational clarity. A researcher or visitor can see the question, scope, status, and evidence boundary without access to private credentials or internal reasoning. Future work can be added without silently replacing an older record.

## Briefly describe a time you realized you had committed to too much. What were your next steps? What were the consequences?

I realized I had committed to too much during the September 14 AutoLabs work, when I was building the research handoff, instrumenting the SAE study, polishing the public surface, and adding unrelated event work at the same time. The risk was that a visible interface could make the study look more finished than the evidence justified.

I stopped and separated what had to be true before the protocol could be presented from what had to be true before a confirmation run could start. I froze the study definition, documented the bounds, kept the public page read-only, and left confirmation behind an explicit approval gate. I deferred work that did not improve protocol clarity or evidence handling.

The consequence was a narrower deliverable and less visual polish than I initially wanted. The more important consequence was that the study could be shown honestly as a bounded protocol with confirmation pending. I now establish the safety or evidence gate first when several projects compete, then spend remaining capacity on improvements that make the work easier to understand.

## Required factual fields

- First name: Raphael, confirm
- Last name: Khalid, confirm
- Email: confirm
- LinkedIn URL: add
- Phone: add
- Location: add
- Resume: upload the current resume
- Earliest start date: answer factually
- Relocation: answer factually, including timing
- US work authorization: answer factually
- Future sponsorship: answer factually
- How heard: select only sources that are true
- Share information with related organizations: choose preference
- Pronouns: choose preference

## Resume project entry to consider

AutoLabs | Research operations and observable agent experiments

- Built a public, read-only observatory that turns open research questions into bounded experiment records with stable status, source links, evidence boundaries, and separate study runners.
- Converted a limitation in persona-vector research into an SAE study protocol for Qwen2.5-7B-Instruct with label-free feature discovery and approval-gated held-out confirmation.

