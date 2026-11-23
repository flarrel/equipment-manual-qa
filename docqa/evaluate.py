"""Compare retrieval, modality and tool selection on fixed cases."""
from __future__ import annotations

import math
from statistics import median
from time import perf_counter

from .model import normalize
from .retrieval import BM25, tokens

MODES = ("text", "image", "agent")
QUESTION_WORDS = frozenset({
    "a", "an", "and", "are", "can", "do", "for", "how", "i", "in", "is",
    "many", "much", "of", "on", "should", "the", "to", "what", "which", "with",
})


def answer(mode, model, page, question, root):
    trace = [{"tool": "retrieve", "page": page["id"]}]
    outputs = []

    def call(use_image):
        trace.append({"tool": "inspect_image" if use_image else "read_text", "page": page["id"]})
        raw = model.answer(question, page["text"], root / page["image"] if use_image else None)
        outputs.append(raw)
        return normalize(raw)

    if mode == "agent":
        query_terms = set(tokens(question)) - QUESTION_WORDS - {page["document"].lower()}
        if not query_terms.intersection(tokens(page["text"])):
            trace.append({"tool": "abstain", "reason": "no topical text match", "page": page["id"]})
            return None, None, trace, outputs
    predicted = call(mode == "image" or (mode == "agent" and not page["text_complete"]))
    if mode == "agent" and page["text_complete"] and predicted is None:
        predicted = call(True)
    return predicted, page["id"] if predicted is not None else None, trace, outputs


def retrieval_metrics(ranks):
    return {
        "n": len(ranks),
        "recall_at_1": sum(rank == 1 for rank in ranks) / len(ranks),
        "recall_at_3": sum(rank <= 3 for rank in ranks) / len(ranks),
        "mrr": sum(1 / rank for rank in ranks) / len(ranks),
    }


def mean(values):
    return sum(values) / len(values) if values else None


def summarize(rows):
    answerable = [r for r in rows if r["expected_answer"] is not None]
    unanswerable = [r for r in rows if r["expected_answer"] is None]
    times = sorted(r["seconds"] for r in rows)
    return {
        "n": len(rows),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "answer_accuracy_answerable": mean([r["answer_correct"] for r in answerable]),
        "citation_accuracy_answerable": mean([r["citation_correct"] for r in answerable]),
        "grounded_outcome_accuracy": mean([r["answer_correct"] and r["citation_correct"] for r in rows]),
        "abstention_rate_answerable": mean([r["prediction"] is None for r in answerable]),
        "abstention_rate_unanswerable": mean([r["prediction"] is None for r in unanswerable]),
        "image_calls": sum(event["tool"] == "inspect_image" for r in rows for event in r["trace"]),
        "median_seconds": median(times),
        "p95_seconds": times[math.ceil(0.95 * len(times)) - 1],
    }


def benchmark(pages, questions, root, reranker, model, split="test", modes=MODES):
    index = BM25([p for p in pages if p["split"] == split])
    rows = []
    positions = {"bm25": [], "reranker": []}
    for q in (q for q in questions if q["split"] == split):
        start = perf_counter()
        lexical = index.rank(q["question"])
        ranked = index.rank(q["question"], reranker)
        retrieval_seconds = perf_counter() - start
        if q["page"] is not None:
            for name, candidates in zip(positions, (lexical, ranked), strict=True):
                rank = next(i for i, (p, _, _) in enumerate(candidates, 1) if p["id"] == q["page"])
                positions[name].append(rank)
        selected = ranked[0][0]
        for mode in modes:
            start = perf_counter()
            predicted, citation, trace, outputs = answer(mode, model, selected, q["question"], root)
            seconds = perf_counter() - start + retrieval_seconds
            if q["page"] is None:
                failure = None if predicted is None else "unsupported_answer"
            elif selected["id"] != q["page"]:
                failure = "wrong_retrieval"
            elif predicted is None:
                failure = "abstained_on_answerable"
            elif predicted != q["answer"]:
                failure = "incorrect_reading"
            else:
                failure = None
            rows.append({"id": q["id"], "kind": q["kind"], "mode": mode, "question": q["question"],
                         "expected_answer": q["answer"], "expected_page": q["page"],
                         "prediction": predicted, "citation": citation, "trace": trace,
                         "raw_outputs": outputs, "failure": failure, "seconds": seconds,
                         "answer_correct": predicted == q["answer"], "citation_correct": citation == q["page"]})
    by_mode = {}
    for mode in modes:
        selected_rows = [r for r in rows if r["mode"] == mode]
        by_mode[mode] = summarize(selected_rows)
        by_mode[mode]["by_kind"] = {
            kind: summarize([r for r in selected_rows if r["kind"] == kind])
            for kind in sorted({r["kind"] for r in selected_rows})
        }
    return {"retrieval": {name: retrieval_metrics(ranks) for name, ranks in positions.items()},
            "modes": by_mode, "cases": rows}
