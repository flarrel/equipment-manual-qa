"""Commands for building and running the benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import data, evaluate, model, retrieval

PROJECT = Path(__file__).resolve().parents[2]


class SmokeModel:
    """Small deterministic double for verifying pipeline mechanics."""

    def answer(self, question, text, image_path=None):
        if "RESET" in question and "Hold RESET for " in text:
            return text.split("Hold RESET for ", 1)[1].split()[0]
        return "UNKNOWN"


def write(path, output):
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("data", "download", "smoke", "benchmark", "ask"))
    parser.add_argument("--work", type=Path, default=PROJECT / "work")
    parser.add_argument("--cache", type=Path, default=PROJECT / "cache" / "hf" / "hub")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--mode", choices=evaluate.MODES, default="agent")
    parser.add_argument("--question")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.command == "ask" and not args.question:
        parser.error("ask requires --question")
    if args.command == "download":
        print(model.download(args.cache))
        return
    torch.set_num_threads(args.threads)
    work = args.work.resolve()
    dataset = work / "dataset"
    pages, questions = data.build(dataset)
    if args.command == "data":
        print(dataset / "manifest.json")
        return
    checkpoint = work / "reranker.pt"
    if args.command != "ask" or not checkpoint.exists():
        history = retrieval.train(pages, questions, checkpoint, epochs=35)
        write(work / "training.json", history)
    ranker = retrieval.restore(checkpoint)
    engine = SmokeModel() if args.command == "smoke" else model.SmolVLM(args.cache)
    if args.command == "ask":
        index = retrieval.BM25([p for p in pages if p["split"] == args.split])
        page = index.rank(args.question, ranker)[0][0]
        prediction, citation, trace, outputs = evaluate.answer(
            args.mode, engine, page, args.question, dataset
        )
        print(json.dumps({"answer": prediction, "citation": citation, "trace": trace,
                          "evidence_image": page["image"], "raw_outputs": outputs}, indent=2))
        return
    results = evaluate.benchmark(pages, questions, dataset, ranker, engine, args.split)
    results["split"] = args.split
    results["dataset_manifest_sha256"] = data.digest(dataset / "manifest.json")
    results["model"] = "deterministic test double" if args.command == "smoke" else {
        "id": model.MODEL_ID, "revision": model.MODEL_REVISION}
    if args.command == "smoke":
        assert len(results["cases"]) == 72 and all(results["retrieval"].values())
        assert any(row["prediction"] is None for row in results["cases"])
        assert any(event["tool"] == "inspect_image" for row in results["cases"] for event in row["trace"])
    path = work / f"{args.command}-{args.split}.json"
    write(path, results)
    print(path)


if __name__ == "__main__":
    main()
