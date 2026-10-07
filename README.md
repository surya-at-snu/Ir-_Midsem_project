# VeriTrace-RAG

A RAG system for scientific questions that checks its own citations.

**CSD358 Information Retrieval, mid-semester hackathon, Track T1 (RAG and trustworthy answers)**
Team: Koneru Akhil (2410110564), Velagala Surya Prakash Reddy (2410110376)

## The idea

An LLM's answer is only as good as what was retrieved for it, and a citation is only useful if it really backs up the sentence it's attached to. VeriTrace-RAG answers questions (or checks claims) over the SciFact corpus. It traces every sentence of the answer back to a specific source sentence, fixes wrong citations, flags claims nothing supports, and says so when retrieval itself has failed.

The retrieval runs on our own inverted index, built from scratch.

## What's new

- **N1: Adaptive hybrid retrieval.** Some queries need exact words matched (BM25), while others need meaning matched (dense retrieval). A small learned router looks at index statistics for each query (idf, score shape, top-10 agreement) and decides how much weight to give each side.
- **N2: Claim-level citation audit.** Each answer sentence is treated as a claim and checked against its cited source. It gets one of five labels: verified, repaired (another source supports it), new source (found by re-searching the whole index), contradicted, or unsupported. The verifier mostly uses IR features (lnc.ltc cosine, idf-weighted coverage, numbers, negation) plus NLI.
- **N3: Fake errors for training.** We corrupt correct claims (negate them, flip a direction, swap a number, or swap the rarest word for another rare word) so the verifier learns to catch hallucinations without hand labels.
- **N4: Knowing when not to answer.** Off-topic questions get zero documents. A confidence model predicts when retrieval has probably failed, and the system abstains or warns instead of guessing.

## Results (SciFact test set, 300 claims)

| Retrieval | nDCG@10 | P@1 |
|---|---|---|
| tf-idf lnc.ltc (lecture baseline) | 0.647 | 0.510 |
| BM25 with learned title/body zones | 0.680 | 0.557 |
| Hybrid, best fixed weight | 0.735 | 0.627 |
| **Adaptive router (N1)** | **0.747** | 0.630 |
| **Router + LambdaMART** | **0.773** | **0.667** |

| Claim checking (908 labelled pairs) | AUROC | Negations caught |
|---|---|---|
| Plain cosine similarity | 0.775 | 56% (chance) |
| NLI only | 0.870 | 72% |
| **VeriTrace verifier** | **0.902** | **85%** |

A few more numbers:

- **End to end with Qwen2.5-1.5B:** the LLM's own citations support only 46% of its claims, while 88% of the claims VeriTrace keeps are confirmed by an independent judge.
- **Off-topic questions:** 39 of 40 general-knowledge questions get zero documents, and only 2.7% of real SciFact claims are wrongly rejected.
- **Spelling correction:** with two typos per query, nDCG@10 goes from 0.614 back up to 0.669.

All numbers come from the scripts in `scripts/` and are saved in `results/`.

## IR concepts from the course, and where they live

| Concept | File |
|---|---|
| Tokenising, stop words, stemming (Porter vs. S-stemmer vs. none) | `veritrace/text.py` |
| Positional inverted index, title/body zones | `veritrace/index.py` |
| Boolean and phrase queries, skip pointers | `veritrace/boolean.py` |
| tf-idf, SMART schemes, Jaccard, BM25 | `veritrace/sparse.py` |
| Heap top-K, champion lists, tiered index, index elimination | `veritrace/sparse.py`, `veritrace/fusion.py` |
| Spelling correction (k-grams, edit distance, Soundex) | `veritrace/spell.py` |
| Cluster pruning | `veritrace/cluster.py` |
| Evaluation (P@k, MAP, MRR, nDCG) | `veritrace/metrics.py` |

## Running it

Tested on Windows 11 with Python 3.13, CPU only (no GPU needed).

```bash
pip install -r requirements.txt
python scripts/download.py                          # datasets + models
python scripts/build_index.py --dataset scifact     # inverted index + dense vectors
python scripts/exp_retrieval.py --dataset scifact   # tunes BM25, trains the router
python scripts/exp_ltr.py --dataset scifact         # LambdaMART
python scripts/train_abstention.py --dataset scifact
python scripts/calibrate_scope_gate.py
python scripts/train_verifier.py
```

Then launch the demo:

```bash
streamlit run app.py
```

Or use the command line, which prints every step: postings, idf, scores, router weight, and the claim audit.

```bash
python cli.py "Does vitamin D deficiency increase the risk of multiple sclerosis?" --generator extractive
python cli.py "Does vitamin D deficency increse the risk of multiple sclerosis?" --spell
python cli.py --term methadone --boolean "\"liver transplantation\" AND methadone NOT children"
```

In PowerShell, put `--%` before `--boolean` so the quotes reach Python unchanged.

Generators: `extractive` (fast, no LLM), `qwen` (Qwen2.5-1.5B, about 40 s on CPU), or `openai` (any OpenAI-compatible endpoint).

The other scripts in `scripts/` reproduce the remaining experiments: efficiency, chunking, the SMART and spelling studies, the end-to-end RAG evaluation, and the figures.

## Data and models

- SciFact (Wadden et al., 2020), plus NFCorpus and FiQA from BEIR to test generalisation
- Hugging Face models: BGE-small (dense), MiniLM cross-encoder, DeBERTa-v3 NLI, Qwen2.5 (generation)

No crawling, no personal data.

## Limitations

Claims are checked as whole sentences, not broken into atomic facts. The audit checks whether a claim is supported, not whether it actually answers the question. Next steps would be claim decomposition, fine-tuning the NLI model on SciFact, and an approximate-nearest-neighbour index for larger corpora.

## Project structure

```
Ir-_Midsem_project/
├── app.py                      Streamlit demo
├── cli.py                      command-line demo (prints every intermediate step)
├── requirements.txt
├── README.md
│
├── veritrace/                  the IR system, one module per stage
│   ├── config.py               paths and settings
│   ├── corpus.py               dataset loading
│   ├── text.py                 tokenising, stop words, stemmers
│   ├── chunking.py             splitting abstracts into sentence windows
│   ├── index.py                positional inverted index with title/body zones
│   ├── boolean.py              Boolean and phrase queries, skip pointers
│   ├── sparse.py               tf-idf, SMART schemes, BM25, champion lists, tiered index
│   ├── spell.py                spelling correction
│   ├── dense.py                dense retrieval (BGE)
│   ├── cluster.py              cluster pruning for the dense index
│   ├── qpp.py                  query performance prediction features
│   ├── router.py               adaptive hybrid router (N1)
│   ├── fusion.py               score fusion and heap top-K
│   ├── rerank.py               cross-encoder reranking
│   ├── ltr.py                  LambdaMART learning to rank
│   ├── generate.py             answer generation (extractive, Qwen, OpenAI-compatible)
│   ├── nli.py                  NLI model wrapper
│   ├── verify.py               claim-level citation audit (N2)
│   ├── corrupt.py              hallucination injection (N3)
│   ├── abstain.py              retrieval confidence and abstention (N4)
│   ├── metrics.py              P@k, MAP, MRR, nDCG
│   └── pipeline.py             ties everything together
│
├── scripts/                    experiments, training and evaluation
│   ├── download.py             datasets and models
│   ├── build_index.py          builds the inverted index and dense vectors
│   ├── exp_retrieval.py        retrieval experiments, trains the router
│   ├── exp_ltr.py              LambdaMART experiments
│   ├── exp_chunking.py         what counts as a document
│   ├── exp_efficiency.py       champion lists, tiered index, index elimination
│   ├── exp_ir_extras.py        SMART schemes, stemming, spelling, cluster pruning
│   ├── train_dense.py          fine-tunes the dense encoder
│   ├── train_verifier.py       trains the claim verifier
│   ├── train_abstention.py     trains the confidence model
│   ├── calibrate_scope_gate.py off-topic gate
│   ├── eval_rag.py             end-to-end RAG evaluation
│   ├── verifier_pairwise.py    injected-hallucination test
│   ├── judge_pool.py           pooling for our self-judged queries
│   ├── measure_latency.py      timing
│   ├── worked_example.py       every number for one query
│   └── make_figures.py         all figures
│
├── data/
│   ├── custom_queries.tsv      our own natural-language queries
│   └── custom_judgments.csv    our relevance judgments for them
│
└── results/                    every metric as JSON
    └── figures/                all plots used in the report
```

`artifacts/` (indexes and trained models) and `data/raw/` (datasets) are not in the repo. The scripts create them when you run the setup steps above.
