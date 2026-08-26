# WildJailbreak pairability and leakage audit

Status: **not passed / not measurable**. No model training was performed.

## What we checked

- The published Hugging Face metadata identifies the source as `allenai/wildjailbreak` and describes 262K vanilla and adversarial prompt-response pairs, including harmful and form-matched benign queries.
- The published paper describes vanilla and adversarial harmful/benign subsets and a bank of mined jailbreak tactics.
- The actual `train.tsv` and `eval.tsv` are gated. The Dataset Viewer denies unauthenticated access.
- The workspace contains no WildJailbreak, WildTeaming, HarmBench, or jailbreak data files.

## Required audit fields

| Requirement | Result |
|---|---|
| Unique underlying intents | Not measurable: released rows unavailable |
| Realizations per intent | Not measurable |
| Attack-family labels per row | Not measurable |
| Benign examples | Public documentation says they exist; row count unavailable |
| Same attack style across different intents | Not measurable |
| Valid same-C/different-S positives | Not measurable |
| Valid different-C/same-S negatives | Not measurable |
| Fully held-out attack families | Not measurable |
| Intent-changing/ambiguous attacks | Not auditable without rows and provenance fields |

## Decision

Do not construct training pairs and do not train a partition, SAE, or ConCA model. The required positive and negative pair definitions depend on row-level intent identity and attack-family provenance, neither of which can be verified from the public metadata alone.

To continue this exact audit, provide an approved local copy of the gated TSV files or authorize access with a Hugging Face account that has accepted AI2's Responsible Use Guidelines. The next action would remain read-only schema and pairability measurement.

Sources: [WildJailbreak dataset metadata](https://huggingface.co/datasets/allenai/wildjailbreak), [WildTeaming paper](https://arxiv.org/abs/2406.18510).
