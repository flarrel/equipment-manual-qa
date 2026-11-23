# Multimodal document QA benchmark

Evaluate whether page images improve answers to questions about equipment manuals. The CLI retrieves a page and returns a cited answer or abstains. It compares a text reader, an image-aware reader, and a bounded agent that decides when to inspect the image.

## Run locally

From this checkout, create a Python 3.13 environment and keep caches and generated files in its parent project directory.

```sh
python3.13 -m venv ../.venv
mkdir -p ../cache/pip ../cache/hf ../tmp ../work
export PIP_CACHE_DIR="$PWD/../cache/pip"
export HF_HOME="$PWD/../cache/hf"
export TMPDIR="$PWD/../tmp"
../.venv/bin/python -m pip install torch==2.8.0
../.venv/bin/python -m pip install -r requirements.txt
../.venv/bin/python -m unittest discover -s tests -v
../.venv/bin/python -m docqa smoke
```

On Linux, use the CPU wheel for the first install command.

```sh
../.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
```

The smoke run trains the PyTorch reranker and checks the pipeline with a deterministic reader. To measure the pretrained model, download the pinned 513 MB weights once and run the real benchmark offline.

```sh
../.venv/bin/python -m docqa download
HF_HUB_OFFLINE=1 ../.venv/bin/python -m docqa benchmark
HF_HUB_OFFLINE=1 ../.venv/bin/python -m docqa ask --mode agent \
  --question 'What color is the AUX connector on Jade-2?'
```

The run writes the dataset, SHA-256 manifest, training losses, checkpoint, and per-case judgments to `../work/`. The ask command prints the selected page, the path to its evidence image, the tool sequence, and raw model output. Set `--work` and `--cache` to choose different directories.

[Benchmark method and limitations](docs/method.md)
