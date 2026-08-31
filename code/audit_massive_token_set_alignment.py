"""Frozen token-set alignment feasibility check; no decoder training."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART, OUT = ROOT / "data" / "massive_token_layer8_bridge_partition", ROOT / "Report" / "massive_token_set_alignment.json"
SEED, ENGLISH = 20260827, "en-US"


def score(query, target):
    query = query / np.linalg.norm(query, axis=1, keepdims=True); target = target / np.linalg.norm(target, axis=1, keepdims=True)
    return float((query @ target.T).max(1).mean())


def main():
    x = np.load(ART / "train_tokens.npy", mmap_mode="r"); md = pd.read_csv(ART / "train_metadata.csv")
    groups = {key: value.index.to_numpy() for key, value in md.groupby(["id", "locale"])}; english_ids = set(md.loc[md.locale == ENGLISH, "id"])
    rng, rows = np.random.default_rng(SEED), []
    for locale in sorted(set(md.locale) - {ENGLISH}):
        ids = np.array(sorted(english_ids & set(md.loc[md.locale == locale, "id"]))); shuffled = rng.permutation(ids)
        if np.any(ids == shuffled): shuffled = np.roll(shuffled, 1)
        for utterance_id, shuffled_id in zip(ids, shuffled):
            foreign = x[groups[(utterance_id, locale)]]
            same_english = x[groups[(utterance_id, ENGLISH)]]
            shuffled_english = x[groups[(shuffled_id, ENGLISH)]]
            rows.append({"locale": locale, "id": int(utterance_id), "same": score(foreign, same_english), "shuffled": score(foreign, shuffled_english)})
    result = pd.DataFrame(rows); result["delta"] = result.same - result.shuffled
    by_locale = result.groupby("locale")[["same", "shuffled", "delta"]].mean().reset_index(); by_locale.to_csv(OUT.with_name("massive_token_set_alignment_by_locale.csv"), index=False)
    report = {"metric": "mean over foreign tokens of max cosine to any English token", "alignment": "token sets only; no token-position correspondence", "pairs": len(result), "same_utterance_mean": float(result.same.mean()), "shuffled_utterance_mean": float(result.shuffled.mean()), "mean_delta": float(result.delta.mean()), "paired_delta_std": float(result.delta.std()), "fraction_same_beats_shuffled": float((result.delta > 0).mean()), "by_locale_csv": "Report/massive_token_set_alignment_by_locale.csv", "decoder_trained": False, "gemma_scope_loaded": False}
    OUT.write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
