"""BM25 baseline and a small trainable reranker."""
from __future__ import annotations

import math
import re
from collections import Counter

import torch
from torch import nn


def tokens(text):
    return re.findall(r"[a-z0-9]+(?:-[0-9]+)?", text.lower())


class BM25:
    def __init__(self, pages):
        self.pages = pages
        self.counts = [Counter(tokens(page["text"])) for page in pages]
        self.lengths = [sum(counts.values()) for counts in self.counts]
        self.average = sum(self.lengths) / len(pages)
        self.frequency = Counter(word for counts in self.counts for word in counts)

    def candidates(self, question):
        query = set(tokens(question))
        results = []
        for page, counts, length in zip(self.pages, self.counts, self.lengths, strict=True):
            normalizer = 1.2 * (0.25 + 0.75 * length / self.average)
            common = query & counts.keys()
            score = sum(
                math.log1p((len(self.pages) - self.frequency[word] + 0.5)
                           / (self.frequency[word] + 0.5))
                * (2.2 * counts[word]) / (counts[word] + normalizer)
                for word in sorted(common)
            )
            features = [score / 10, len(common) / len(query) if query else 0,
                        float(page["document"].lower() in query),
                        len(common) / len(counts), length / 30]
            results.append((page, score, features))
        return results

    def rank(self, question, model=None):
        results = self.candidates(question)
        if model is None:
            return sorted(results, key=lambda row: (-row[1], row[0]["id"]))
        with torch.inference_mode():
            scores = model(torch.tensor([row[2] for row in results])).flatten().tolist()
        scored = [(row[0], score, row[2]) for row, score in zip(results, scores, strict=True)]
        return sorted(scored, key=lambda row: (-row[1], row[0]["id"]))


def reranker():
    return nn.Sequential(nn.Linear(5, 12), nn.ReLU(), nn.Linear(12, 1))


def training_pairs(pages, questions, split):
    index = BM25([p for p in pages if p["split"] == split])
    features, labels = [], []
    for q in questions:
        if q["split"] != split or q["page"] is None:
            continue
        for p, _, row in index.candidates(q["question"]):
            features.append(row)
            labels.append(float(p["id"] == q["page"]))
    return torch.tensor(features), torch.tensor(labels)


def train(pages, questions, checkpoint, epochs=35, seed=581):
    torch.manual_seed(seed)
    training, labels = training_pairs(pages, questions, "train")
    validation, targets = training_pairs(pages, questions, "validation")
    net = reranker()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.02)
    criterion = nn.BCEWithLogitsLoss(pos_weight=(labels.numel() - labels.sum()) / labels.sum())
    history = []
    best_loss = math.inf
    for epoch in range(1, epochs + 1):
        net.train()
        optimizer.zero_grad()
        loss = criterion(net(training).flatten(), labels)
        loss.backward()
        optimizer.step()
        net.eval()
        with torch.inference_mode():
            val_loss = criterion(net(validation).flatten(), targets).item()
        history.append({"epoch": epoch, "train_loss": loss.item(), "validation_loss": val_loss})
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(net.state_dict(), checkpoint)
    return history


def restore(checkpoint):
    net = reranker()
    net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return net.eval()
