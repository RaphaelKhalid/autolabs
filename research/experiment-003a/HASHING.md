# 3A manifest hash basis

`launch-manifest.json` contains its own hash identifier, so the identifier is not the raw byte hash of the final self-referential file. The launch identifier in the manifest and Worker is the SHA-256 of the canonical JSON bytes produced by:

1. Parse `launch-manifest.json` as JSON.
2. Replace the `manifestHash` value with `sha256:TO_BE_COMPUTED_AFTER_FREEZE`.
3. Serialize with two-space indentation and a final newline, preserving the checked-in key order.
4. SHA-256 those UTF-8 bytes.

The resulting value is `sha256:9fd05fffe50a6f402fea4dab6f7cf51dcd66ef78ea9fa450374bd0f2411ca766`. The relay must perform this check before pushing Kaggle. Any change to the manifest requires a new hash and a new launch request.

