"""Q1 text features: BERT word embeddings + generic CTC forced alignment.

- bert-base-uncased (frozen, local) gives contextual word embeddings.
- wav2vec2-base-960h (generic ASR CTC, NOT sentiment-trained) emission aligned
  to the transcript via torchaudio's CTC forced_align (mature Viterbi).
CTC always traverses every target token, so a word is accepted only when its
alignment confidence clears a threshold; punctuation / pure-number tokens have
no spoken entity and are excluded. No timestamps are invented on failure.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

ROOT = Path(r"D:\研究生\jianmo\HuaweiCup-Codex-Final-2026E-SOTA")
BERT_DIR = ROOT / "workspace" / "references" / "bert-base-uncased"
W2V_DIR = ROOT / "workspace" / "references" / "wav2vec2-base-960h"
FRAME_HOP = 0.02     # wav2vec2 output stride (seconds)
WORD_CONF_THRESH = 0.30
SAMPLE_VERIFY_FRAC = 0.8


class TextModels:
    def __init__(self):
        from transformers import (BertModel, BertTokenizerFast, Wav2Vec2ForCTC,
                                  Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor)
        self.bert_tok = BertTokenizerFast.from_pretrained(str(BERT_DIR), local_files_only=True)
        self.bert = BertModel.from_pretrained(str(BERT_DIR), local_files_only=True).eval()
        self.w2v = Wav2Vec2ForCTC.from_pretrained(str(W2V_DIR), local_files_only=True).eval()
        self.w2v_tok = Wav2Vec2CTCTokenizer(str(W2V_DIR / "vocab.json"))
        self.w2v_feat = Wav2Vec2FeatureExtractor(str(W2V_DIR / "preprocessor_config.json"))
        self.blank = self.w2v_tok.pad_token_id

    # ---- BERT word-level embeddings ----
    def word_embeddings(self, text: str) -> list:
        enc = self.bert_tok(text, return_offsets_mapping=True, truncation=True,
                            max_length=512, return_tensors="pt")
        with torch.inference_mode():
            out = self.bert(input_ids=enc["input_ids"],
                            attention_mask=enc["attention_mask"],
                            token_type_ids=enc["token_type_ids"]).last_hidden_state[0].numpy()
        wid = enc.word_ids(0)
        spans = enc.pop("offset_mapping")[0]
        groups = {}
        for i, w in enumerate(wid):
            if w is None:
                continue
            groups.setdefault(w, []).append(i)
        words = []
        for w in sorted(groups):
            idxs = groups[w]
            cs = min(int(spans[i][0]) for i in idxs)
            ce = max(int(spans[i][1]) for i in idxs)
            words.append({"word": text[cs:ce], "char_start": cs, "char_end": ce,
                          "emb": out[idxs].mean(0).astype(np.float32)})
        return words

    # ---- wav2vec2 emission ----
    def emission(self, pcm16: np.ndarray) -> np.ndarray:
        x = pcm16.astype(np.float32) / 32768.0
        inp = self.w2v_feat(x, sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            logits = self.w2v(inp.input_values).logits[0]
        return torch.log_softmax(logits, dim=-1).numpy()

    # ---- CTC forced alignment via torchaudio ----
    def forced_align(self, words: list, pcm16: np.ndarray) -> dict:
        from torchaudio.functional import forced_align, merge_tokens
        vocab = self.w2v_tok.get_vocab()
        target_ids, tmap = [], []
        alignable = set()
        for wi, w in enumerate(words):
            chars = [ch for ch in w["word"].upper() if ch in vocab]
            if chars:
                alignable.add(wi)
            if wi > 0:
                target_ids.append(self.w2v_tok._convert_token_to_id("|"))
                tmap.append(None)
            for ch in chars:
                target_ids.append(self.w2v_tok._convert_token_to_id(ch))
                tmap.append(wi)
        if not target_ids or not alignable:
            for w in words:
                w["start_s"] = w["end_s"] = None
                w["align_conf"] = None
                w["align_status"] = "NON_SPEECH"
            return {"status": "UNRESOLVED", "reason": "no alignable speech tokens",
                    "n_words": len(words), "n_alignable": 0, "n_resolved": 0}

        logp = self.emission(pcm16)
        T = logp.shape[0]
        aligns, scores = forced_align(
            torch.tensor(logp).unsqueeze(0), torch.tensor([target_ids], dtype=torch.int32),
            torch.tensor([T], dtype=torch.int32),
            torch.tensor([len(target_ids)], dtype=torch.int32), blank=self.blank)
        spans = merge_tokens(aligns[0], scores[0], blank=self.blank)

        # Spans are 1:1 and in target order.
        wconf = {wi: [] for wi in range(len(words))}
        for i, sp in enumerate(spans):
            if i < len(tmap) and tmap[i] is not None:
                wconf[tmap[i]].append(float(np.exp(sp.score)))

        n_resolved = 0
        for wi, w in enumerate(words):
            if wi not in alignable:
                w["start_s"] = w["end_s"] = None
                w["align_conf"] = None
                w["align_status"] = "NON_SPEECH"
                continue
            conf = float(np.mean(wconf[wi])) if wconf[wi] else 0.0
            w["align_conf"] = conf
            if conf >= WORD_CONF_THRESH and wconf[wi]:
                sps = [sp for i, sp in enumerate(spans)
                       if i < len(tmap) and tmap[i] == wi]
                w["start_s"] = float(sps[0].start * FRAME_HOP)
                w["end_s"] = float((sps[-1].end + 1) * FRAME_HOP)
                w["align_status"] = "VERIFIED"
                n_resolved += 1
            else:
                w["start_s"] = w["end_s"] = None
                w["align_status"] = "UNRESOLVED"

        n_align = len(alignable)
        status = "VERIFIED" if n_resolved / n_align >= SAMPLE_VERIFY_FRAC else "UNRESOLVED"
        return {"status": status, "n_words": len(words), "n_alignable": n_align,
                "n_resolved": n_resolved, "n_frames": T}
