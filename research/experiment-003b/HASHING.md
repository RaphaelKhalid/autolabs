# Experiment 3B frozen contract hashing

The frozen worker contract uses deterministic UTF-8 hashes.

- Text files use LF line endings and the exact UTF-8 bytes after newline normalization.
- JSON objects are parsed and serialized with sorted keys, compact separators (`,`, `:`), and UTF-8 without ASCII escaping.
- `promptSha256` hashes the normalized `scoring-prompt.md` bytes.
- `schemaSha256` hashes the canonical compact `scoring-schema.json`.
- `blindMembershipSha256` hashes the canonical object containing only opaque pair IDs, scenario/text SHA-256 values, truncation flags, and repeat eligibility. It contains no raw outputs, feature IDs, signs, or candidate names.
- `manifestHash` hashes the canonical manifest object with its `manifestHash` property omitted. The manifest records the prompt, schema, membership, source, model, counters, and budget contract.

The generated worker module is server-only. Public clients must use only non-sensitive study counters and provenance; they must not import the membership map.