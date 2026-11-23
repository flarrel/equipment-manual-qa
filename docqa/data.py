"""Generate fictional manuals and labeled question sets."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BRANDS = (
    "Aster", "Birch", "Cedar", "Dune", "Elm", "Flint", "Glade", "Hazel",
    "Iris", "Jade", "Kite", "Larch",
)
SPLITS = ("train",) * 6 + ("validation",) * 3 + ("test",) * 3
COLORS = {"red": "#e53935", "blue": "#1e88e5", "green": "#27a357", "yellow": "#e9ba10"}


def draw_page(path, title, kind, items, split, layout):
    image = Image.new("RGB", (640, 480), "white")
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default(size=29)
    body = ImageFont.load_default(size=25)
    draw.text((30, 27), title, font=title_font, fill="#172d41")
    draw.line((28, 73, 610, 73), fill="#456783", width=3)
    if kind == "instructions":
        for i, line in enumerate(items):
            draw.text((30, 108 + 38 * i), line, font=body, fill="black")
    if kind == "specifications":
        left, top, gap = {"train": (35, 102, 57), "validation": (65, 119, 64),
                          "test": (90, 132, 66)}[split]
        rows = [("Setting", "Value"), *items]
        if layout and split != "train":
            rows = [(right, left) for left, right in rows]
        for i, (first, second) in enumerate(rows):
            y = top + i * gap
            draw.rectangle((left, y, left + 460, y + gap), fill="#e8f2fb" if i % 2 else "#f8fbff",
                           outline="#7795ac")
            draw.line((left + 260, y, left + 260, y + gap), fill="#7795ac")
            draw.text((left + 17, y + 12), first, font=body, fill="black")
            draw.text((left + 278, y + 12), second, font=body, fill="black")
    if kind == "diagram":
        X = {"train": (143, 288, 433), "validation": (125, 280, 435),
             "test": (100, 260, 420)}[split]
        y = {"train": 179, "validation": 192, "test": 219}[split]
        draw.rounded_rectangle((52, y - 34, 587, y + 127), radius=18, outline="#42576b", width=5)
        for x, (label, color) in zip(X, items, strict=True):
            draw.ellipse((x, y, x + 63, y + 63), fill=COLORS[color], outline="black", width=3)
            draw.text((x, y + 77), label, font=body, fill="black")
    image.save(path)


def build(root: Path, seed: int = 581):
    root.mkdir(parents=True, exist_ok=True)
    (root / "pages").mkdir(exist_ok=True)
    rng = random.Random(seed)
    pages, questions = [], []
    for family, (brand, split) in enumerate(zip(BRANDS, SPLITS, strict=True)):
        for variant in range(2):
            name = f"{brand}-{variant + 1}"
            seconds = rng.choice([5, 8, 10, 12, 15, 20, 25, 30])
            liters = rng.choice([2, 3, 4, 6, 7, 9])
            mode = rng.choice(["Eco", "Standard", "Boost"])
            colors = rng.sample(list(COLORS), 3)
            ports = list(zip(("IN", "AUX", "OUT"), colors, strict=True))
            if split != "train" and variant == 1:
                ports.reverse()
            instructions = [f"Hold RESET for {seconds} seconds.", "Wait for the status light.",
                            "Release the button before restarting."]
            data = [
                ("instructions", "Operating instructions", " ".join(instructions), instructions),
                ("specifications", "Specifications table", "Settings and capacity reference.",
                 [("Reservoir", f"{liters} L"), ("Startup", mode)]),
                ("diagram", "Port diagram", "Connector panel labels and colors.", ports),
            ]
            for number, (kind, title, body, items) in enumerate(data, 1):
                key = f"{name}-p{number}"
                file = f"pages/{key}.png"
                draw_page(root / file, f"{name} / {title}", kind, items, split, variant)
                pages.append({"id": key, "document": name, "family": family, "split": split,
                              "text": f"{name} {title}. {body}", "image": file})
            for number, (category, wording, answer, page) in enumerate([
                ("text", f"How many seconds should I hold RESET on {name}?", str(seconds), 1),
                ("table", f"What is the reservoir capacity in liters on {name}?", str(liters), 2),
                ("diagram", f"What color is the AUX connector on {name}?", colors[1], 3),
                ("unanswerable", f"What is the warranty length in months for {name}?", None, None),
            ], 1):
                questions.append({"id": f"{name}-q{number}", "document": name, "family": family,
                                  "split": split, "kind": category, "question": wording,
                                  "answer": answer, "page": f"{name}-p{page}" if page else None})
    validate(pages, questions)
    files = {}
    for label, rows in (("pages", pages), ("questions", questions)):
        path = root / f"{label}.json"
        path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        files[path.name] = digest(path)
    files.update({p["image"]: digest(root / p["image"]) for p in pages})
    manifest = {"seed": seed, "counts": {split: {
        kind: sum(q["split"] == split and q["kind"] == kind for q in questions)
        for kind in ("text", "table", "diagram", "unanswerable")
    } for split in dict.fromkeys(SPLITS)}, "sha256": files}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return pages, questions


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(pages, questions):
    indexed = {p["id"]: p for p in pages}
    if len(indexed) != len(pages) or len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Duplicate page or question ID")
    for field in ("family", "document"):
        for key in {p[field] for p in pages}:
            if len({p["split"] for p in pages if p[field] == key}) != 1:
                raise ValueError(f"{field} leaks across splits: {key}")
    for q in questions:
        if q["kind"] == "unanswerable":
            if q["page"] is not None or q["answer"] is not None:
                raise ValueError(f"Invalid abstention label: {q['id']}")
            continue
        p = indexed[q["page"]]
        if not q["answer"] or any(q[f] != p[f] for f in ("document", "family", "split")):
            raise ValueError(f"Invalid evidence label: {q['id']}")
