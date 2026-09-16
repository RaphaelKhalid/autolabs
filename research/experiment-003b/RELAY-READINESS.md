# Relay readiness and worker handoff

scripts/autolabs_relay.py is the owner-local controller. It has no provider
SDK or OpenAI endpoint and sends all live calls to the configured AutoLabs
worker. Dry-run writes only relay-plan.json. Execution uses one controller
with at most five concurrent leased shard workers; the five animated lab
roles are logical grading shards, not five uncoordinated processes.

The dry-run plan from the frozen local inputs is:

    source_records=780
    primary_pair_requests=768
    order_swapped_repeats=117
    shared_adjudication_retry_reserve=129
    call_ceiling=1014
    budget_usd=10
    analyst_context_target=100k_tokens

The relay loads the ignored private-run/relay.env file (or explicit
environment variables) without printing its values. It persists a stable
idempotency key, merges the authenticated server score export on restart, and
deduplicates score receipts by (record_id, repeatIndex). A partial final JSONL
line is ignored so an interrupted process can resume safely.

## Execution phases

The worker owns authentication, frozen source membership, leases, schema
validation, atomic worst-case reservations, provider usage reconciliation, and
retry charging. The relay drives these phases in order:

1. Start or resume the durable run using the stable idempotency key.
2. Claim and score the 768 primary pair requests with repeatIndex=0.
3. Join the primary phase, then score the 117 swapped repeats with
   repeatIndex=1 and orderSwap=true.
4. Freeze a deterministic disagreement plan using the canonical hash
   sha256(canonical_json({"recordIds": sorted(recordIds), "runId": runId})).
   The plan leaves one attempt in the shared reserve for the analyst call.
5. Score selected third-judge records with repeatIndex=2 and orderSwap=false,
   preserving primary and repeat evidence.
6. Run the local reliability analysis, write the all-32 feature table, send
   one bounded compact analyst packet, and persist separate analysis and
   artifact digests.
7. Submit the final receipt with confirmationStatus="pending" and require
   the worker to report complete.

If no disagreements are selected, the empty frozen plan is still recorded.
The analyst response is advisory and cannot replace numeric scores or the
frozen shortlist. A rejected or unfunded analyst call is recorded as pending;
the relay never fabricates an analyst completion.

No live run has been started.
