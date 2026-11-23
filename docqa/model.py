"""Load a pinned vision-language checkpoint and generate short answers."""
from __future__ import annotations

import re

from PIL import Image

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
MODEL_REVISION = "d039ab0d4b9311b617e8ffec1d06c8464364aeb3"
MODEL_FILES = ("*.json", "merges.txt", "model.safetensors")


def download(cache):
    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL_ID, revision=MODEL_REVISION, cache_dir=cache,
                             allow_patterns=list(MODEL_FILES))


class SmolVLM:
    def __init__(self, cache):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        settings = {"revision": MODEL_REVISION, "cache_dir": cache, "local_files_only": True}
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, **settings)
        # Use one bounded image view per page so CPU inference remains practical.
        self.processor.image_processor.do_image_splitting = False
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, dtype="float32", **settings
        ).eval()

    def answer(self, question, text, image_path=None):
        import torch

        prompt = (
            "Use only the evidence. If the evidence does not give the answer, reply UNKNOWN. "
            "Otherwise reply with only the number or color, without units or explanation.\n"
            f"Evidence text: {text}\nQuestion: {question}\nAnswer:"
        )
        content = [{"type": "text", "text": prompt}]
        if image_path is not None:
            content.insert(0, {"type": "image"})
        rendered = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False
        )
        if image_path is None:
            inputs = self.processor(text=rendered, return_tensors="pt")
        else:
            with Image.open(image_path) as image:
                inputs = self.processor(text=rendered, images=[image.convert("RGB")], return_tensors="pt")
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=16, do_sample=False)
        answer = self.processor.batch_decode(output[:, inputs.input_ids.shape[1]:],
                                             skip_special_tokens=True)[0]
        return answer.strip()


def normalize(answer):
    cleaned = answer.lower().strip().removeprefix("answer:").strip(" .!\n")
    if cleaned in {"", "unknown", "unanswerable", "i don't know", "not specified",
                   "not provided", "cannot determine", "no answer"}:
        return None
    match = re.fullmatch(r"(\d+)(?:\s*(?:seconds?|liters?|litres?|months?|l))?", cleaned)
    return str(int(match[1])) if match else cleaned
