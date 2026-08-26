"""Frozen Gemma layer-8 FLORES language coverage audit; no training."""
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

_stub = torch.library.Library("torchvision", "DEF"); _stub.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Report" / "flores_gemma_coverage_audit.json"
LANGUAGES = ("eng_Latn", "fra_Latn", "spa_Latn", "deu_Latn", "rus_Cyrl", "arb_Arab", "hin_Deva", "zho_Hans", "swh_Latn", "tur_Latn")
SAMPLE, BATCH, DEVICE = 200, 32, "cuda" if torch.cuda.is_available() else "cpu"

def main():
    ds = load_dataset("facebook/flores", "all", split="dev").select(range(SAMPLE))
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b", local_files_only=True); tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModel.from_pretrained("google/gemma-2-2b", local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEVICE).eval()
    embeddings, report = {}, {"source": "facebook/flores all/dev", "sample_sentences": SAMPLE, "languages": list(LANGUAGES), "device": DEVICE, "pooling": "masked mean residual stream layer 8", "training_performed": False}
    with torch.inference_mode():
        for language in LANGUAGES:
            texts = [row[f"sentence_{language}"] for row in ds]
            vectors, unk, tokens = [], 0, 0
            for start in range(0, SAMPLE, BATCH):
                encoded = tokenizer(texts[start:start+BATCH], padding=True, truncation=True, max_length=128, return_tensors="pt"); unk += int((encoded.input_ids == tokenizer.unk_token_id).sum()); tokens += int(encoded.attention_mask.sum()); encoded = encoded.to(DEVICE)
                hidden = model(**encoded, output_hidden_states=True, use_cache=False).hidden_states[8]; mask = encoded.attention_mask.unsqueeze(-1); vectors.append(((hidden * mask.to(hidden.dtype)).sum(1) / mask.sum(1)).float().cpu().numpy())
            embeddings[language] = np.concatenate(vectors); report.setdefault("coverage", {})[language] = {"nonempty": sum(bool(t.strip()) for t in texts), "unk_fraction": unk / max(tokens, 1), "mean_characters": float(np.mean([len(t) for t in texts]))}
            print(f"audited_language={language} rows={len(texts)}")
    english = embeddings["eng_Latn"] / np.linalg.norm(embeddings["eng_Latn"], axis=1, keepdims=True)
    report["cross_language_to_english"] = {}
    for language in LANGUAGES:
        other = embeddings[language] / np.linalg.norm(embeddings[language], axis=1, keepdims=True); scores = other @ english.T; order = np.argsort(-scores, axis=1); ranks = np.argmax(order == np.arange(SAMPLE)[:, None], axis=1) + 1
        report["cross_language_to_english"][language] = {"R@1": float(np.mean(ranks == 1)), "MRR": float(np.mean(1 / ranks)), "same_sentence_mean_cosine": float(np.mean(np.diag(scores))), "random_other_mean_cosine": float((scores.sum() - np.trace(scores)) / (SAMPLE * (SAMPLE - 1)))}
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8"); print(json.dumps(report, indent=2))

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        OUT.write_text(json.dumps({"status": "error", "error": repr(error), "training_performed": False}, indent=2) + "\n", encoding="utf-8")
        raise
