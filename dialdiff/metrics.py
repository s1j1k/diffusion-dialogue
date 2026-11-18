"""Turning generated ids back into text, and scoring it."""
from collections import Counter

from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu


def decode_response(ids, mask, tokenizer):
    """Text of the generated part of one sequence, up to its first [SEP]."""
    out = []
    for tok, m in zip(ids.tolist(), mask.tolist()):
        if not m:
            continue
        if tok == tokenizer.sep_token_id:
            break
        if tok in (tokenizer.pad_token_id, tokenizer.cls_token_id):
            continue
        out.append(tok)
    return tokenizer.decode(out, skip_special_tokens=True).strip()


def distinct_n(texts, n):
    """Fraction of n-grams across all outputs that are unique (diversity)."""
    grams = Counter()
    for t in texts:
        words = t.split()
        grams.update(zip(*[words[i:] for i in range(n)]))
    total = sum(grams.values())
    return len(grams) / total if total else 0.0


def score(hypotheses, references):
    """
    corpus_bleu: standard corpus-level BLEU-4.
    sentence_bleu: mean smoothed sentence BLEU-4, the figure DiffuSeq reports.
    """
    hyp = [h.split() for h in hypotheses]
    ref = [[r.split()] for r in references]
    smooth = SmoothingFunction().method4
    sent = [sentence_bleu(r, h, smoothing_function=smooth) if h else 0.0 for r, h in zip(ref, hyp)]
    return {
        "n": len(hyp),
        "corpus_bleu": corpus_bleu(ref, hyp, smoothing_function=SmoothingFunction().method1),
        "sentence_bleu": sum(sent) / len(sent) if sent else 0.0,
        "distinct_1": distinct_n(hypotheses, 1),
        "distinct_2": distinct_n(hypotheses, 2),
        "avg_len": sum(map(len, hyp)) / len(hyp) if hyp else 0.0,
    }
