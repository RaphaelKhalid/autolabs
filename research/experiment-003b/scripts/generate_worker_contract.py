"""Generate the server-only frozen Experiment 3B contract asset."""
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
RESEARCH = ROOT / "research" / "experiment-003b"
PAIRS = RESEARCH / ".local-test" / "blind-final" / "blind-pairs.jsonl"
MAPPING = RESEARCH / ".local-test" / "blind-final" / "private-mapping.json"
PROMPT = RESEARCH / "scoring-prompt.md"
SCHEMA = RESEARCH / "scoring-schema.json"
MANIFEST = RESEARCH / "scoring-manifest.json"
TARGET = ROOT / "orchestrator-worker" / "src" / "persona-3b-contract.generated.ts"

def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def normalized_text(path: Path) -> str:
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")

def repeat_ids(pairs: list[dict[str, Any]], mapping: dict[str, Any], count: int, seed: int) -> set[str]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for pair in pairs:
        meta = mapping.get("pairs", {}).get(pair["record_id"], {})
        key = (str(meta.get("caseId", "")), str(meta.get("feature", "")), str(meta.get("condition", "")))
        groups.setdefault(key, []).append(pair)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    selected: list[str] = []
    ordered = list(groups.values())
    cursor = 0
    while len(selected) < count and ordered:
        group = ordered[cursor % len(ordered)]
        if group:
            selected.append(group.pop()["record_id"])
        ordered = [value for value in ordered if value]
        cursor += 1
    return set(selected)

def ts(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    prompt = normalized_text(PROMPT)
    schema = json.loads(normalized_text(SCHEMA))
    manifest = json.loads(normalized_text(MANIFEST))
    pairs = [json.loads(line) for line in normalized_text(PAIRS).splitlines() if line.strip()]
    mapping = json.loads(normalized_text(MAPPING))
    if len(pairs) != 768:
        raise ValueError(f"expected 768 blinded pairs, found {len(pairs)}")
    for pair in pairs:
        if sha256_text(pair["A"]) != pair["AResponseSha256"] or sha256_text(pair["B"]) != pair["BResponseSha256"]:
            raise ValueError(f"response hash mismatch for {pair['record_id']}")
    repeat = repeat_ids(pairs, mapping, 117, int(mapping.get("seed", 20260915)) + 1)
    membership: dict[str, dict[str, Any]] = {}
    for pair in sorted(pairs, key=lambda item: item["record_id"]):
        membership[pair["record_id"]] = {
            "scenarioSha256": sha256_text(pair["scenario"]),
            "aSha256": pair["AResponseSha256"],
            "bSha256": pair["BResponseSha256"],
            "truncationA": bool(pair["truncation"]["A"]),
            "truncationB": bool(pair["truncation"]["B"]),
            "repeatEligible": pair["record_id"] in repeat,
        }
    membership_hash = sha256_text(canonical_json(membership))
    prompt_hash = sha256_text(prompt)
    schema_hash = sha256_text(canonical_json(schema))
    manifest["grader"].update({"maxOutputTokens": 4096, "promptSha256": prompt_hash, "schemaSha256": schema_hash, "blindMembershipSha256": membership_hash})
    manifest["reliability"].update({
        "synthesisCallsReserved": 1,
        "adjudicationAndRetryReserve": 129,
        "retryAndAdjudicationCallsAvailable": 128,
        "reservePolicy": "The 129 remaining attempts under the 1,014-call ceiling include one reserved synthesis call; the other 128 are shared by disagreement review and bounded retries. Each use is logged and cannot increase the ceiling.",
    })
    manifest["execution"].update({
        "apiCallsSynthesisReserved": 1,
        "apiCallsAdjudicationAndRetryReserve": 128,
        "estimatedInputTokensPerCall": 1653,
        "estimatedOutputTokensPerCall": 4096,
        "estimatedTotalTokensAtCeiling": 5829486,
        "billableInputTokenCap": 1676142,
        "billableOutputTokenCap": 4153344,
        "estimateBasis": "Measured o200k_base counts over 768 frozen blind pairs: system+record+schema p95 1,653 input tokens; 4,096 output allowance includes reasoning and visible JSON. This is a planning estimate, not a dollar quote.",
    })
    manifest["contractAsset"] = {"workerModule": "orchestrator-worker/src/persona-3b-contract.generated.ts", "hashingDocument": "research/experiment-003b/HASHING.md", "membershipSha256": membership_hash, "membershipRecords": len(membership), "repeatEligibleRecords": len(repeat)}
    manifest.pop("manifestHash", None)
    manifest_hash = sha256_text(canonical_json(manifest))
    manifest["manifestHash"] = manifest_hash
    module = """/* GENERATED FILE: run research/experiment-003b/scripts/generate_worker_contract.py --write. */
export const PERSONA_3B_JUDGE_PROMPT = %s;
export const PERSONA_3B_SCORE_SCHEMA = %s as const;
export const PERSONA_3B_PROMPT_SHA256 = %s;
export const PERSONA_3B_SCHEMA_SHA256 = %s;
export const PERSONA_3B_BLIND_MEMBERSHIP_SHA256 = %s;
export const PERSONA_3B_FINAL_MANIFEST_HASH = %s;
export const PERSONA_3B_TOTAL_RECORDS = 780;
export const PERSONA_3B_PRIMARY_PAIR_COUNT = 768;
export const PERSONA_3B_REPEAT_RECORD_COUNT = 117;
export const PERSONA_3B_SYNTHESIS_RESERVED_CALLS = 1;
export const PERSONA_3B_RETRY_ADJUDICATION_RESERVE = 128;
export const PERSONA_3B_CALL_CEILING = 1014;
export const PERSONA_3B_OUTPUT_TOKENS = 4096;
export type Persona3BBlindMembership = { scenarioSha256: string; aSha256: string; bSha256: string; truncationA: boolean; truncationB: boolean; repeatEligible: boolean };
export const PERSONA_3B_BLIND_MEMBERSHIP: Readonly<Record<string, Persona3BBlindMembership>> = %s;
export type Persona3BInput = { recordId: string; scenario: string; A: string; B: string; truncation: { A: boolean; B: boolean }; AResponseSha256: string; BResponseSha256: string; repeatIndex: number; orderSwap: boolean };
export type Persona3BInputValidationOptions = { disagreementRecordIds?: ReadonlySet<string> };
export async function sha256Hex(value: string): Promise<string> { const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(value)); return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join(''); }
export async function validatePersona3BInput(input: Persona3BInput, options: Persona3BInputValidationOptions = {}): Promise<boolean> {
  if (!input || !/^[p][0-9]{7}$/.test(input.recordId) || !Number.isInteger(input.repeatIndex) || input.repeatIndex < 0 || input.repeatIndex > 2) return false;
  const expected = PERSONA_3B_BLIND_MEMBERSHIP[input.recordId];
  if (!expected || typeof input.scenario !== 'string' || typeof input.A !== 'string' || typeof input.B !== 'string') return false;
  if (typeof input.truncation?.A !== 'boolean' || typeof input.truncation?.B !== 'boolean') return false;
  const scenarioHash = await sha256Hex(input.scenario), aHash = await sha256Hex(input.A), bHash = await sha256Hex(input.B);
  if (scenarioHash !== expected.scenarioSha256) return false;
  if (input.repeatIndex === 0 || input.repeatIndex === 2) {
    if (input.orderSwap || aHash !== expected.aSha256 || bHash !== expected.bSha256 || input.truncation.A !== expected.truncationA || input.truncation.B !== expected.truncationB) return false;
  } else {
    if (!expected.repeatEligible || !input.orderSwap || aHash !== expected.bSha256 || bHash !== expected.aSha256 || input.truncation.A !== expected.truncationB || input.truncation.B !== expected.truncationA) return false;
  }
  if (input.repeatIndex === 2 && !options.disagreementRecordIds?.has(input.recordId)) return false;
  return input.AResponseSha256 === (input.orderSwap ? expected.bSha256 : expected.aSha256) && input.BResponseSha256 === (input.orderSwap ? expected.aSha256 : expected.bSha256);
}
export function validPersona3BScore(value: unknown): value is Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const row = value as Record<string, unknown>; const fields = ['relational_stance','self_positioning','assertiveness','affective_tone','voice_coherence','task_completion','topical_fidelity','specificity','refusal_safety','lexical_format_artifact'];
  if (typeof row.record_id !== 'string' || !/^[p][0-9]{7}$/.test(row.record_id) || !row.scores || !row.pairwise || !Array.isArray(row.evidence) || row.evidence.length > 3 || row.evidence.some((item) => typeof item !== 'string' || item.length > 240) || typeof row.notes !== 'string' || row.notes.length > 300) return false;
  for (const label of ['A','B']) { const scores = (row.scores as Record<string, unknown>)[label]; if (!scores || typeof scores !== 'object' || Array.isArray(scores) || Object.keys(scores).some((key) => !fields.includes(key)) || fields.some((key) => { const item = (scores as Record<string, unknown>)[key]; const max = key === 'refusal_safety' ? 2 : key === 'lexical_format_artifact' ? 3 : 4; return !(item === null || Number.isInteger(item) && Number(item) >= 0 && Number(item) <= max); })) return false; }
  const pairFields = ['relational_stance','self_positioning','assertiveness','affective_tone','voice_coherence']; const pairwise = row.pairwise as Record<string, unknown>;
  return Object.keys(pairwise).every((key) => pairFields.includes(key)) && pairFields.every((key) => { const pair = pairwise[key] as Record<string, unknown> | undefined; return Boolean(pair && ['increase','decrease','no_clear_change','unknown'].includes(String(pair.direction)) && [0,1,2].includes(Number(pair.magnitude))); });
}
""" % (ts(prompt), ts(schema), ts(prompt_hash), ts(schema_hash), ts(membership_hash), ts(manifest_hash), ts(membership))
    if args.write:
        TARGET.write_text(module, encoding="utf-8", newline="\n")
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote module={TARGET} manifestHash={manifest_hash} membershipRecords={len(membership)} repeatEligible={len(repeat)}")
    else:
        print(f"promptSha256={prompt_hash} schemaSha256={schema_hash} membershipSha256={membership_hash} manifestHash={manifest_hash}")
    return 0
if __name__ == '__main__':
    raise SystemExit(main())