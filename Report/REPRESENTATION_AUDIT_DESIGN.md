# Frozen-representation audit — fixed configuration

What we will do: extract frozen `google/gemma-2-2b` representations for every strict V1 text.

Why: Stage 2 asks whether the frozen representation contains both factual C structure and surface S structure before any new objective is introduced.

Fixed diagnostic sweep: hidden-state indices 5, 8, 13, and 21 (after blocks 4, 7, 12, and 20 of 26), masked mean pooling over non-padding tokens, and maximum length 128. These diagnostic layers are fixed before results; the sweep is diagnostic only.

Stored metadata: example ID, fact ID, domain, relation, subject ID, value ID, S family, S variant, C split, S split, and the row index into `activations.npy`.

Not included: InfoNCE, SAE training, contrastive pairs used for optimization, or any new learned objective. Linear probes are diagnostic measurements only.
