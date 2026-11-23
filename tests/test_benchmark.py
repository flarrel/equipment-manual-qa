from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from docqa import data, evaluate, model, retrieval
from docqa.__main__ import SmokeModel


class BenchmarkTests(unittest.TestCase):
    def test_deterministic_data_and_no_cross_split_pages(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pages, questions = data.build(root)
            manifest = (root / "manifest.json").read_bytes()
            self.assertEqual((len(pages), len(questions)), (72, 96))
            data.build(root)
            self.assertEqual(manifest, (root / "manifest.json").read_bytes())
            leaked = [*pages, dict(pages[0], id="leak", split="test")]
            with self.assertRaisesRegex(ValueError, "leaks across splits"):
                data.validate(leaked, questions)

    def test_train_restore_and_score(self):
        torch.set_num_threads(1)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pages, questions = data.build(root)
            checkpoint = root / "reranker.pt"
            history = retrieval.train(pages, questions, checkpoint, epochs=3)
            self.assertEqual(len(history), 3)
            result = evaluate.benchmark(pages, questions, root, retrieval.restore(checkpoint), SmokeModel())
            self.assertEqual(result["modes"]["agent"]["n"], 24)
            self.assertEqual(result["retrieval"]["bm25"]["n"], 18)
            self.assertEqual(result["modes"]["agent"]["abstention_rate_unanswerable"], 1)
            self.assertEqual(len(result["cases"]), 72)

    def test_agent_retries_only_after_abstention(self):
        page = {"id": "x-p1", "text": "sample", "image": "page.png"}

        class Reader:
            def answer(self, question, text, image_path):
                return "UNKNOWN" if image_path is None else "Blue."

        prediction, citation, trace, outputs = evaluate.answer("agent", Reader(), page, "Color?", Path("."))
        self.assertEqual((prediction, citation), ("blue", "x-p1"))
        self.assertEqual([t["tool"] for t in trace], ["retrieve", "read_text", "inspect_image"])
        self.assertEqual(len(outputs), 2)
        self.assertEqual(model.normalize("8 seconds."), "8")
