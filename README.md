# Equipment Manual QA

Ask a question about the included fictional office-equipment manuals. The local CLI retrieves a page and returns an answer with a page citation or abstains. It also prints the path to the page image and a trace of the tools it used.

The benchmark measures when reading a page image improves answers. It compares a text-only reader, an always-image reader, and an agent that opens the image when searchable text is incomplete. It measures answer accuracy, citation accuracy, abstention, and latency.

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

The smoke run trains the PyTorch reranker and checks the pipeline with a deterministic reader. Download the pinned 513 MB vision-language model once to ask questions and run the full benchmark offline.

```sh
../.venv/bin/python -m docqa download
HF_HUB_OFFLINE=1 ../.venv/bin/python -m docqa ask --mode agent \
  --question 'What color is the AUX connector on Jade-2?'
HF_HUB_OFFLINE=1 ../.venv/bin/python -m docqa benchmark
```

The demo question returns `blue` and cites page `Jade-2-p3`. The benchmark writes the dataset manifest, training losses, checkpoint, and per-question results to `../work/`. Set `--work` and `--cache` to choose different directories.

[Benchmark method](docs/method.md) · [Held-out results and failure examples](docs/results.md)
