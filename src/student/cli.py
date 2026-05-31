

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

import fire
from tqdm import tqdm

from student.chunking import build_chunks
from student.evaluation import evaluate, load_ground_truth
from student.generation import AnswerGenerator, ContextSnippet
from student.indexing import (
    DEFAULT_PROCESSED_DIR,
    BM25Retriever,
    build_and_save_index,
)
from student.models import (
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from student.cli_helpers import (
    print_header,
    print_section,
    print_success,
    print_error,
    print_info,
    print_param,
    print_examples,
    print_system_requirements,
    print_prereqs,
    print_troubleshooting,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_RAW_DIR = Path("data/raw")


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


def _find_corpus_root(raw_dir: Path) -> Path:
    """Pick a sensible root inside ``data/raw``.

    The spec ships ``data/raw/vllm-0.10.1`` plus a zip. If a unique
    subdirectory exists, prefer it over the bare ``data/raw`` so the chunk
    paths are anchored at the project root (e.g. ``vllm/...``).
    """
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw corpus directory not found: {raw_dir}. "
            "Place the vLLM repository under data/raw/."
        )
    subdirs = [p for p in raw_dir.iterdir() if p.is_dir()]
    if len(subdirs) == 1:
        return subdirs[0]
    return raw_dir


def _load_dataset(path: Path) -> RagDataset:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return RagDataset.model_validate(raw)


def _save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "model_dump"):
        data = payload.model_dump()  # type: ignore[union-attr]
    else:
        data = payload
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


class StudentCLI:
    """CLI commands exposed by ``python -m student``.
    
    Quick start:
        python -m student help                    # Show this help
        python -m student index                   # Build BM25 index
        python -m student search "your question"  # Single query
        python -m student answer "your question"  # Get answer
    """

    # --- Help --------------------------------------------------------

    def help(self) -> None:
        """Show help and usage information."""
        print_header("RAG Against the Machine - CLI Help")
        
        print_section("What is this?")
        print()
        print("  Retrieval-Augmented Generation over the vLLM codebase.")
        print("  Ask questions about vLLM and get grounded answers powered by")
        print("  a local LLM (Qwen/Qwen3-0.6B) and BM25 retrieval.")
        print()
        
        print_system_requirements()
        print_prereqs()
        
        print_section("Available Commands")
        print()
        print(f"  {self._fmt_cmd('index')} - Build the BM25 index from corpus")
        print(f"  {self._fmt_cmd('search')} - Retrieve chunks for a query")
        print(f"  {self._fmt_cmd('search_dataset')} - Batch retrieve from dataset")
        print(f"  {self._fmt_cmd('answer')} - Get answer to a single question")
        print(f"  {self._fmt_cmd('answer_dataset')} - Batch generate answers")
        print(f"  {self._fmt_cmd('evaluate')} - Compute recall@k metrics")
        print()
        
        print_examples()
        print_troubleshooting()
    
    @staticmethod
    def _fmt_cmd(name: str) -> str:
        """Format command name with color."""
        from student.cli_helpers import Colors
        return f"{Colors.BOLD}{Colors.CYAN}{name:<20}{Colors.RESET}"

    # --- Indexing --------------------------------------------------------

    def index(
        self,
        raw_dir: str = str(DEFAULT_RAW_DIR),
        processed_dir: str = str(DEFAULT_PROCESSED_DIR),
        max_chunk_size: int = 2000,
    ) -> None:
        """Ingest the corpus and build a BM25 index.
        
        This command walks through the vLLM codebase, chunks Python code
        using AST-aware strategies and text/Markdown by paragraph, then
        builds a sparse BM25 index for fast retrieval.

        Args:
            raw_dir: Directory containing the raw corpus (e.g. vLLM repo).
                    Default: 'data/raw' (will auto-detect subdirs)
            processed_dir: Where to write the chunks/ and bm25_index/.
                          Default: 'data/processed'
            max_chunk_size: Maximum characters per chunk (≤ 8000 recommended).
                           Default: 2000
                           
        Examples:
            # Standard indexing
            python -m student index
            
            # Custom chunk size
            python -m student index --max_chunk_size 4000
            
            # Custom directories
            python -m student index --raw_dir /path/to/corpus --processed_dir /path/to/output
        """
        raw = Path(raw_dir)
        processed = Path(processed_dir)
        try:
            root = _find_corpus_root(raw)
        except FileNotFoundError as exc:
            _err(str(exc))
            sys.exit(1)
        t0 = time.time()
        print_info(f"Starting indexing from {root}")
        print_param("Output directory", str(processed))
        print_param("Max chunk size", f"{max_chunk_size} chars")
        print()
        chunks = build_chunks(root, max_chunk_size=max_chunk_size)
        elapsed = time.time() - t0
        print_success(f"Built {len(chunks)} chunks in {elapsed:.1f}s")
        build_and_save_index(chunks, processed)
        print_success(f"Ingestion complete! Indices saved to {processed}/")
        print()

    # --- Single-query search --------------------------------------------

    def search(
        self,
        query: str,
        k: int = 10,
        processed_dir: str = str(DEFAULT_PROCESSED_DIR),
    ) -> None:
        """Search the index for a single query and print the results.
        
        Performs BM25 search returning top-k most relevant code/doc snippets
        with their locations in the source codebase.

        Args:
            query: The search query (e.g. "How to configure OpenAI server?")
            k: Number of top results to return. Default: 10
            processed_dir: Directory containing the BM25 index.
                          Default: 'data/processed'
                          
        Examples:
            python -m student search "initialization" --k 5
            python -m student search "TokenizerConfig" --k 20
        """
        retriever = self._load_retriever(processed_dir)
        if retriever is None:
            return
        print_info(f"Searching for: '{query}'")
        results = retriever.search(query, k=k)
        print_success(f"Found {len(results)} results")
        print()
        payload = {
            "query": query,
            "k": k,
            "results": [
                {
                    "file_path": c.file_path,
                    "first_character_index": c.first_character_index,
                    "last_character_index": c.last_character_index,
                    "score": score,
                    "preview": c.text[:200],
                }
                for c, score in results
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    # --- Batch search ----------------------------------------------------

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
        processed_dir: str = str(DEFAULT_PROCESSED_DIR),
    ) -> None:
        """Run retrieval over every question in a dataset and save results.
        
        Batch process a JSON dataset containing multiple questions, retrieve
        top-k sources for each, and save results to disk.

        Args:
            dataset_path: Path to JSON file with questions (UnansweredQuestions format)
            k: Number of top results per question. Default: 10
            save_directory: Where to save results (creates if needed).
                           Default: 'data/output/search_results'
            processed_dir: Directory containing the BM25 index.
                          Default: 'data/processed'
                          
        Output format: StudentSearchResults JSON with question_id, question, and
                      retrieved_sources for each.
                          
        Examples:
            python -m student search_dataset \\
                --dataset_path data/datasets/UnansweredQuestions/dataset.json \\
                --k 10
        """
        dataset_file = Path(dataset_path)
        if not dataset_file.exists():
            print_error(f"Dataset not found: {dataset_file}")
            sys.exit(1)
        retriever = self._load_retriever(processed_dir)
        if retriever is None:
            return
        try:
            dataset = _load_dataset(dataset_file)
        except Exception as exc:
            print_error(f"Failed to parse dataset {dataset_file}: {exc}")
            sys.exit(1)

        print_info(f"Loading dataset from {dataset_file}")
        print_param("Questions", str(len(dataset.rag_questions)))
        print_param("Top-k results", str(k))
        print()
        
        results: List[MinimalSearchResults] = []
        for q in tqdm(dataset.rag_questions, desc="Searching", unit="q"):
            sources = retriever.search_to_sources(q.question, k=k)
            results.append(
                MinimalSearchResults(
                    question_id=q.question_id,
                    question=q.question,
                    retrieved_sources=sources,
                )
            )
        out = StudentSearchResults(search_results=results, k=k)
        out_path = Path(save_directory) / dataset_file.name
        _save_json(out_path, out)
        print_success(f"Saved search results to {out_path}")
        print_param("Total results", str(len(results)))

    # --- Single-query answer --------------------------------------------

    def answer(
        self,
        query: str,
        k: int = 10,
        processed_dir: str = str(DEFAULT_PROCESSED_DIR),
        model: str = "Qwen/Qwen3-0.6B",
        max_context_chars: int = 8000,
        max_new_tokens: int = 384,
        thinking: bool = False,
        temperature: float = 0.0,
    ) -> None:
        """Retrieve top-k context and generate an answer to ``query``.
        
        This is the full RAG pipeline: retrieves relevant code/docs,
        assembles them into a context block, then prompts a local LLM
        to answer using only that retrieved context (no hallucination).

        Args:
            query: The question (e.g. "How to configure OpenAI server?")
            k: Number of source snippets to retrieve. Default: 10
            processed_dir: Directory containing the BM25 index.
                          Default: 'data/processed'
            model: Hugging Face model ID for generation.
                  Default: 'Qwen/Qwen3-0.6B' (small, fast, local)
            max_context_chars: Max characters in context block. Default: 8000
            max_new_tokens: Max tokens to generate. Default: 384
            thinking: Enable chain-of-thought reasoning (if supported).
                     Default: False
            temperature: Sampling temperature (0.0 = deterministic).
                        Default: 0.0
                        
        Examples:
            python -m student answer "How do I use vLLM?"
            python -m student answer "What is sampling_params?" --k 5
            python -m student answer "Explain caching" --max_new_tokens 512
        """
        retriever = self._load_retriever(processed_dir)
        if retriever is None:
            return
        print_info(f"Retrieving top-{k} snippets...")
        results = retriever.search(query, k=k)
        snippets = [
            ContextSnippet(
                file_path=c.file_path,
                first_character_index=c.first_character_index,
                last_character_index=c.last_character_index,
                text=c.text,
            )
            for c, _ in results
        ]
        sources = [
            MinimalSource(
                file_path=c.file_path,
                first_character_index=c.first_character_index,
                last_character_index=c.last_character_index,
            )
            for c, _ in results
        ]
        try:
            print_info("Generating answer with local LLM...")
            generator = AnswerGenerator(
                model_id=model,
                max_context_chars=max_context_chars,
                max_new_tokens=max_new_tokens,
                enable_thinking=thinking,
                temperature=temperature,
            )
            answer_text = generator.generate(query, snippets, sources)
        except Exception as exc:
            print_error(f"Generation failed: {exc}")
            sys.exit(1)
        print_success("Answer generated!")
        print()
        payload = {
            "question": query,
            "answer": answer_text,
            "retrieved_sources": [s.model_dump() for s in sources],
            "k": k,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    # --- Batch answers ---------------------------------------------------

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
        processed_dir: str = str(DEFAULT_PROCESSED_DIR),
        model: str = "Qwen/Qwen3-0.6B",
        max_context_chars: int = 8000,
        max_new_tokens: int = 384,
        thinking: bool = False,
        temperature: float = 0.0,
    ) -> None:
        """Generate answers for every search result previously saved to disk."""
        in_path = Path(student_search_results_path)
        if not in_path.exists():
            _err(f"Search results not found: {in_path}")
            sys.exit(1)
        try:
            results = StudentSearchResults.model_validate_json(
                in_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            _err(f"Failed to parse search results {in_path}: {exc}")
            sys.exit(1)

        # Map chunk lookups by (file_path, first_idx, last_idx) so the
        # generator can show the exact text the retriever pointed to.
        chunk_index = self._build_chunk_index(processed_dir)
        if chunk_index is None:
            return
        print(
            f"Loaded {len(results.search_results)} questions "
            f"from {in_path}"
        )
        generator = AnswerGenerator(
            model_id=model,
            max_context_chars=max_context_chars,
            max_new_tokens=max_new_tokens,
            enable_thinking=thinking,
            temperature=temperature,
        )

        answered: List[MinimalAnswer] = []
        total = len(results.search_results)
        for i, item in enumerate(
            tqdm(results.search_results, desc="Answering", unit="q")
        ):
            snippets = [
                ContextSnippet(
                    file_path=s.file_path,
                    first_character_index=s.first_character_index,
                    last_character_index=s.last_character_index,
                    text=chunk_index.get(
                        (s.file_path, s.first_character_index,
                         s.last_character_index), "",
                    ),
                )
                for s in item.retrieved_sources
            ]
            try:
                answer_text = generator.generate(
                    item.question, snippets, item.retrieved_sources
                )
            except Exception as exc:
                _err(f"Generation failed for question {item.question_id}: {exc}")
                answer_text = ""
            answered.append(
                MinimalAnswer(
                    question_id=item.question_id,
                    question=item.question,
                    retrieved_sources=item.retrieved_sources,
                    answer=answer_text,
                )
            )
            if (i + 1) % 10 == 0 or i + 1 == total:
                tqdm.write(f"Processed {i + 1} of {total} questions")

        out = StudentSearchResultsAndAnswer(
            search_results=answered, k=results.k
        )
        out_path = Path(save_directory) / in_path.name
        _save_json(out_path, out)
        print(
            f"Saved student_search_results_and_answer to {out_path}"
        )

    # --- Evaluation ------------------------------------------------------

    def evaluate(
        self,
        student_answer_path: str,
        dataset_path: str,
        k: int = 10,
        max_context_length: int = 2000,
        ks: Optional[Sequence[int]] = None,
        min_overlap_ratio: float = 0.05,
    ) -> None:
        """Compute recall@k between student results and an answered dataset.

        Evaluates how well the retrieval system performs by comparing the
        retrieved sources against ground-truth sources in an AnsweredQuestions
        dataset. Uses 5% overlap threshold as per spec.

        Args:
            student_answer_path: Path to StudentSearchResults or 
                                StudentSearchResultsAndAnswer JSON
            dataset_path: Path to AnsweredQuestions dataset JSON
            k: Primary k value for recall (also used for defaults).
              Default: 10
            max_context_length: (Accepted for API compatibility, not used).
            ks: Additional k values to evaluate (e.g. [1, 3, 5, 10]).
               Default: (1, 3, 5, k)
            min_overlap_ratio: Overlap threshold to count as match.
                              Default: 0.05 (5% per spec)
                              
        Output: Renders recall@k table for each k value.
        
        Examples:
            python -m student evaluate \\
                --student_answer_path data/output/results.json \\
                --dataset_path data/datasets/AnsweredQuestions/dataset.json
        """
        del max_context_length  # accepted for compatibility
        student_path = Path(student_answer_path)
        truth_path = Path(dataset_path)
        if not student_path.exists():
            print_error(f"Student results not found: {student_path}")
            sys.exit(1)
        if not truth_path.exists():
            print_error(f"Ground-truth dataset not found: {truth_path}")
            sys.exit(1)
        try:
            student = StudentSearchResults.model_validate_json(
                student_path.read_text(encoding="utf-8")
            )
        except Exception:
            # The student file may also be a StudentSearchResultsAndAnswer.
            full = StudentSearchResultsAndAnswer.model_validate_json(
                student_path.read_text(encoding="utf-8")
            )
            student = StudentSearchResults(
                search_results=[
                    MinimalSearchResults(
                        question_id=a.question_id,
                        question=a.question,
                        retrieved_sources=a.retrieved_sources,
                    )
                    for a in full.search_results
                ],
                k=full.k,
            )
        ground = load_ground_truth(str(truth_path))
        ks_eff = tuple(ks) if ks else (1, 3, 5, k)
        ks_eff = tuple(sorted(set(ks_eff)))
        
        print_info("Evaluating retrieval performance...")
        print_param("Student results", str(len(student.search_results)))
        print_param("Ground truth questions", str(len(ground)))
        print_param("K values", str(ks_eff))
        print_param("Overlap threshold", f"{min_overlap_ratio*100:.1f}%")
        print()
        
        report = evaluate(
            student, ground, ks=ks_eff, min_overlap_ratio=min_overlap_ratio
        )
        print_success("Evaluation complete!")
        print()
        print(f"✓ Student data is valid")
        with_sources = sum(1 for g in ground if g.sources)
        print(f"✓ Total questions with sources: {with_sources}/{len(ground)}")
        print()
        print(report.render())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_retriever(self, processed_dir: str) -> Optional[BM25Retriever]:
        try:
            return BM25Retriever(Path(processed_dir))
        except FileNotFoundError as exc:
            _err(str(exc))
            return None
        except Exception as exc:  # pragma: no cover - defensive
            _err(f"Failed to load index: {exc}")
            return None

    def _build_chunk_index(
        self, processed_dir: str
    ) -> Optional[dict]:
        from student.indexing import load_chunks

        try:
            chunks = load_chunks(Path(processed_dir))
        except FileNotFoundError as exc:
            _err(str(exc))
            return None
        return {
            (c.file_path, c.first_character_index, c.last_character_index): c.text
            for c in chunks
        }


def main() -> None:
    """Fire entry point."""
    fire.Fire(StudentCLI())


if __name__ == "__main__":  # pragma: no cover
    main()
