# Benchmark results

Run on an Apple Silicon CPU with four PyTorch threads and the pinned SmolVLM-256M checkpoint. The held-out dataset manifest SHA-256 was `b05adcfdeec183fbf5aa9c31c9e54457fd25debd53ee857b951143a918c76a0c`.

- **Text only:** 7/24 grounded outcomes, 7/18 correct answerable responses, 0/6 correct abstentions, and 0 image calls. Median and p95 latency were 0.125 s and 0.250 s.
- **Always image:** 16/24 grounded outcomes, 16/18 correct answerable responses, 0/6 correct abstentions, and 24 image calls. Median and p95 latency were 0.492 s and 0.527 s.
- **Bounded agent:** 22/24 grounded outcomes, 16/18 correct answerable responses, 6/6 correct abstentions, and 12 image calls. Median and p95 latency were 0.284 s and 0.485 s.

Both retrieval approaches selected the correct first page for all 18 answerable test questions. Recall@3 and mean reciprocal rank also tied at 1.0. The trained reranker's validation loss fell from 1.778 after the first epoch to 0.464 at epoch 35. Lower validation loss did not yield better held-out retrieval on this corpus. Manual names and topical keywords provide strong lexical signals.

The first validation agent invoked the image reader only after an explicit model abstention. It guessed on all six unanswerable questions and made no image calls. The final agent first checks for topical evidence and routes pages with incomplete text directly to the image reader. On the final validation split it abstained on all six unanswerable questions and made 12 image calls for 24 questions. The held-out comparison used this frozen policy.

## Failure examples

- **Diagram reading.** For `Jade-2-q3`, the text reader guessed red. The agent inspected the diagram on `Jade-2-p3` and returned the labeled AUX color, blue.
- **Table reading.** For `Jade-2-q2` and `Larch-2-q2`, the correct page was retrieved and viewed, but the model returned the row label “Reservoir” instead of its capacity, 6 liters. Both tables used the reversed-column layout. The reader handled one other reversed table correctly, so the pattern is suggestive and the sample is small.
- **Unsupported answers.** The always-image reader supplied a warranty value for all six manuals even though none contained warranty evidence. The agent's topic check abstained on those cases. Its behavior on valid paraphrases and subtler contradictions remains unmeasured.

The six cases per question type provide a small descriptive result. Shared question templates and controlled missing text make this useful for comparing evidence paths. The results do not estimate performance on real manuals.
