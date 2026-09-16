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
   Matryoshka BatchTopK SAE on samples drawn from that buffer. Reports a
   `"train"` record every `checkpoint_every_tokens` tokens with
   `{tokens_done, tokens_target, loss, recon_loss, aux_loss,
   fraction_variance_explained, l0, dead_fraction}`, and saves a resumable
   safetensors checkpoint to `<workdir>/checkpoints/`.
3. **post-train check** -- `checks.sae_replace_check`: replace the residual
   at `layer` with the SAE's reconstruction and confirm the model's greedy
   argmax next token is preserved on >=90% of positions on a held-out
   prompt. Reports 1 record to stage `"train"`.
4. **calibrate** (steer) -- reloads the model with all layers restored
   (harvesting only ever truncates the same in-memory model object, and
   `ActivationHarvester.close()` restores it -- no reload from disk needed).
   Picks `steer_features` inner-shell features by firing density, sweeps
   sign x dose x scenario, and also runs unsteered baselines and a
   random-direction null control swept across the same doses. Reports every
   generation (baseline + steered + random-control) as a record to stage
   `"calibrate"`, batched at 200 records per Worker call.
5. **analyze / done** -- computes embedding-free proxies (normalized
   word-level edit distance and length delta vs. the same-scenario
   baseline, plus the coherence log-prob already computed during
   calibration), writes `<workdir>/summary.json` and a standalone
   `<workdir>/smoke-report.html`, then reports stage `"done"` with
   `status="complete"`.

Any uncaught exception is reported to stage `"done"` with `status="failed"`
and the exception message, then re-raised (the process exits non-zero; the
Worker sees the failure even if nothing above got to `"done"`).

## Resumability

Each stage checks for its own output file under `<workdir>` before doing any
work:

| Stage | Marker file(s) |
|---|---|
| boot | `boot_checks.json` |
| harvest+train | `sae.safetensors` + `feature_stats.json` (final); `checkpoints/sae_step_*.safetensors` (partial) |
| post-train check | `post_train_check.json` |
| calibrate | `calibration_records.json` |
| analyze | `summary.json` + `smoke-report.html` |

If harvest+train is interrupted, the SAE resumes from the latest checkpoint
under `checkpoints/`. The dataset stream itself is **not** seekable (it's a
`datasets` streaming shuffle buffer, not an index), so a resumed run starts
tokenizing a fresh, differently-shuffled pass over the dataset rather than
picking up at the exact same document -- for a smoke test with a 2M-token
target and a `seed`-shuffled 200k-conversation dataset this is a fine
tradeoff; the SAE optimizer state (Adam moments) is *not* checkpointed, only
the weights, so a resume restarts Adam's moving averages from zero.

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

26 tests, all CPU-only: SAE forward/loss and matryoshka-nested-loss-decreases
on a random 64-dim toy, BatchTopK's per-token-average-k property, the
Worker's canonical-JSON sha256 contract against a known hash, and the
assistant-turn masking algorithm against a small fake tokenizer that
reproduces a `<|im_start|>{role}...<|im_end|>` chat template without any
network access.

## Estimated smoke-test runtime on 1x A40 48GB

This is an estimate, not a measurement -- there is no GPU in this
environment, so nothing below has been timed. Order-of-magnitude only.

Assumptions: Qwen2.5-7B-Instruct, bf16, hidden size 3584, 28 decoder layers
total (layers 0..19 kept during harvest = 20/28 ~= 0.71 of decoder compute);
forward-only FLOPs/token ~= 2 x (params in the kept layers); ultrachat
conversations are roughly 40-60% assistant tokens, so reaching 2,000,000
*assistant* tokens requires forwarding roughly 3.5-4.5M total tokens through
the truncated model; HF `generate()` (no vLLM/TensorRT) on an unbatched 7B
bf16 model on one A40 does very roughly 15-30 output tok/s.

| Phase | Work | Rough time |
|---|---|---|
| Model load (x2: truncated for harvest, restored in place for steering) | download/cache + weight load | 2-5 min |
| Harvest + train | ~4M tokens forwarded through 20/28 layers, ~488 SAE optimizer steps (batch_tokens=4096) | 15-30 min |
| Post-train check | 1 forward pass, 1 prompt | <1 min |
| Steer / calibrate | 8 features x 2 signs x 3 doses x 4 scenarios = 192, + 4 baselines + 2 random x 3 doses x 4 scenarios = 24, = 220 generations at up to 256 tokens, plus 216 extra coherence forward passes | 50-70 min |
| Analysis + report | pure Python/string work | <1 min |
| **Total** | | **roughly 70-110 minutes** |

The steering phase dominates because generations run unbatched and
sequentially (one hook configuration at a time, since the additive
steering vector changes every generation); batching same-dose/same-feature
generations across scenarios would be the first optimization if this
matters in practice. The full-run config (100M tokens, 256 features, 24
scenarios) scales harvest+train roughly linearly with `tokens_target`
(~50x smoke -> many hours) and the calibration screen to 256 x 2 x 3 x 24 =
36,864 generations, which is the dominant full-run cost by a wide margin
and should probably be batched or sampled down before attempting it as-is.

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
- Runtime estimate above is a rough FLOPs/throughput calculation, not a
  measurement; real HF `generate()` throughput on an A40 varies a lot with
  attention implementation (SDPA vs. flash-attn-2), sequence length, and
  whether `torch.compile` is used (not used here).
- `checks.sae_replace_check`'s >=90% threshold is checked and logged as a
  warning if missed, but does not hard-fail the run -- SAE reconstruction
  quality is expected to vary and a smoke test's job is to surface that
  number, not gate on it.
