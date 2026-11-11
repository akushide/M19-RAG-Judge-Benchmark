# Validity of automatic judges for retrieval-augmented question answering on technical regulations: a Persian-language framework evaluated on Section 19 (Energy Management) of the Iranian National Building Regulations

Code for the paper's pipeline: data preparation, embedding and indexing, batch QA, automatic judges,
the interactive platform, benchmark evaluation (M19-100) and statistics.

## Setup
1. `pip install -r requirements-core.txt`
2. Copy `Keys.env.example` to `Keys.env` and set `GROQ_API_KEY` (or export it as an environment variable). **Never commit `Keys.env`.**
3. Set `M19_BASE_PATH` to the folder that holds your data, `models/multilingual-e5-large` and the ChromaDB index
   (default: the script's own folder).

## Folders
| Folder | Content |
|---|---|
| 01_data_preparation | docx to Markdown, header-based chunking, parent-child tree |
| 02_embedding_indexing | E5 embeddings; ChromaDB / FAISS indexing |
| 03_batch_qa | Batch QA pipeline (v3.73) |
| 04_judges | Reference-agreement judge (v2, v3 numeric fix), NLI / LLM / claim judges |
| 05_interactive_platform | Streamlit platform versions used in the benchmark (v20, v23, v24) and the latest (v25) |
| 06_benchmark_evaluation | Benchmark runner, reference judge, controlled tests |
| 07_statistics | ROC-AUC with cluster bootstrap, kappa |
| 08_tests | Unit tests for the numeric fixes |

## Notes
- Judge scores measure agreement with the reference or the retrieved evidence. They are not proofs of regulatory correctness.
- The source regulation text is not redistributed here; obtain it from the official publisher.
- Benchmark: M19-100 (see the dataset folder or release assets).
