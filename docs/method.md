# Benchmark method

## Data

The corpus is generated from seed 581. It contains 24 fictional office-equipment manuals, 72 rendered pages and 96 questions. Each manual contributes one instruction, table, diagram and unanswerable question. Training uses 12 manuals across six families. Validation and test each use six manuals across three disjoint families. Table coordinates and diagram placements vary by split. Validation and test also contain reversed table columns and rotated port order. Page and question annotations are written separately from the PNG evidence and are indexed in a SHA-256 manifest.

Page titles and descriptions are searchable. Instruction values are available in the text layer. Table values and diagram colors exist only in the rendered evidence. This construction isolates access to pixels under controlled conditions. Manual language and layouts are synthetic, and the same question forms recur across the splits. The test measures transfer across generated families, values, and layout shifts within those forms. It says little about open-ended questions or production documents.

## Retrieval and reader

BM25 scores the full split corpus. A two-layer PyTorch network predicts page relevance from BM25, overlap, page length and query-to-document match features. Weighted binary cross-entropy handles the one-positive-to-many-negatives ratio. Adam updates 35 epochs with a fixed seed. The best validation-loss checkpoint is reloaded for inference. The retrieval benchmark scores BM25 and the network independently.

All answering modes consume the same top-ranked page, with a frozen SmolVLM checkpoint and greedy generation. Text mode receives the extracted evidence. Image mode also receives one scaled view of the page. The bounded agent checks for a topical match after excluding common question words and the manual name. It abstains when no topical term matches, inspects the page image directly when the text layer is incomplete, and can retry with the image once after an explicit text abstention. The system attaches the selected page ID to a returned answer and emits a tool trace. This citation identifies the evidence presented to the model. Its presence alone does not guarantee the evidence supports an answer.

Lewis et al. motivated keeping an inspectable source outside the language model. Their research combined dense retrieval and trained generation. Here BM25 and the reranker serve as inspectable retrieval baselines and the reader is frozen so the evidence comparison holds the model fixed. [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)

The SmolVLM model card describes the image-and-text chat format. The pipeline uses that format for the image path, sends the same evidence text on both paths, and pins the model revision to keep the comparison repeatable. [SmolVLM-256M-Instruct model card](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)

## Evaluation

- Retrieval recall@1, recall@3 and mean reciprocal rank use answerable questions with a labeled page.
- Answer accuracy and citation accuracy are reported separately on answerable questions. Grounded outcome accuracy requires both the expected answer and page, or an abstention for an unanswerable question.
- Abstention rates are reported separately by answerability. Results also break out instruction, table, diagram and unanswerable cases.
- Median and nearest-rank p95 times cover retrieval and answering after the model has loaded. Timings include the initial inference and exclude model loading and training. The shared retrieval time includes scoring both rankers. Image-call counts reveal the cost of the agent's routing.
- Raw responses and expected answers are saved per question. Failure categories distinguish wrong retrieval, abstention on answerable cases, incorrect reading and unsupported answers on unanswerable cases. These categories are diagnostic heuristics.
- The test split has six cases per question type. Use the reported counts as descriptive measurements. CPU time depends on the machine, and the benchmark does not estimate accuracy on real equipment manuals.
- The agent uses the generator's text-completeness metadata. A real ingestion pipeline needs to derive that signal independently. Exact topic matching can abstain on valid paraphrases or accept superficially related evidence.
