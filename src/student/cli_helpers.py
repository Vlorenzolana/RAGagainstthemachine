

from __future__ import annotations

import sys
from typing import Optional


class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Foreground colors
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    
    # Background colors
    BG_GREEN = "\033[102m"
    BG_YELLOW = "\033[103m"
    BG_RED = "\033[101m"


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text:^70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.RESET}\n")


def print_section(text: str) -> None:
    """Print a formatted section title."""
    print(f"{Colors.BOLD}{Colors.BLUE}▸ {text}{Colors.RESET}")


def print_success(text: str) -> None:
    """Print a success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")


def print_error(text: str) -> None:
    """Print an error message to stderr."""
    print(f"{Colors.RED}✗ {text}{Colors.RESET}", file=sys.stderr)


def print_warning(text: str) -> None:
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")


def print_info(text: str) -> None:
    """Print an info message."""
    print(f"{Colors.CYAN}ℹ {text}{Colors.RESET}")


def print_param(name: str, value: str) -> None:
    """Print a parameter with its value."""
    print(f"  {Colors.DIM}{name:<25}{Colors.RESET} {value}")


def print_example(title: str, command: str) -> None:
    """Print an example command."""
    print(f"{Colors.BOLD}{title}:{Colors.RESET}")
    print(f"  {Colors.CYAN}$ {command}{Colors.RESET}")


def print_examples() -> None:
    """Print common usage examples."""
    print_section("Quick Start Examples")
    print()
    print_example(
        "1. Build the index",
        "python -m student index --max_chunk_size 2000"
    )
    print()
    print_example(
        "2. Search for a single query",
        'python -m student search "How to configure OpenAI server?" --k 10'
    )
    print()
    print_example(
        "3. Generate an answer",
        'python -m student answer "How to configure OpenAI server?" --k 10'
    )
    print()
    print_example(
        "4. Batch process dataset",
        "python -m student search_dataset \\\n"
        "    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \\\n"
        "    --k 10"
    )
    print()
    print_example(
        "5. Evaluate results",
        "python -m student evaluate \\\n"
        "    --student_answer_path data/output/search_results/dataset_docs_public.json \\\n"
        "    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json"
    )
    print()


def print_system_requirements() -> None:
    """Print system requirements."""
    print_section("System Requirements")
    print()
    print_param("Python", "≥ 3.10, < 3.13")
    print_param("RAM", "≥ 8 GB recommended (4 GB minimum)")
    print_param("Disk", "≥ 15 GB for models + corpus")
    print_param("Time", "≤ 5 min for indexing (reference corpus)")
    print()


def print_prereqs() -> None:
    """Print prerequisites."""
    print_section("Prerequisites")
    print()
    print("  1. Python 3.10+ installed")
    print("  2. uv package manager (https://github.com/astral-sh/uv)")
    print("  3. vLLM repository extracted to data/raw/vllm-*/")
    print()


def print_troubleshooting() -> None:
    """Print troubleshooting guide."""
    print_section("Troubleshooting")
    print()
    print(f"  {Colors.BOLD}Issue: 'Raw corpus directory not found'{Colors.RESET}")
    print("    → Extract vLLM zip to: data/raw/vllm-0.10.1/")
    print()
    print(f"  {Colors.BOLD}Issue: Out of memory errors{Colors.RESET}")
    print("    → Reduce max_chunk_size or increase system RAM")
    print()
    print(f"  {Colors.BOLD}Issue: Generation is very slow{Colors.RESET}")
    print("    → Use GPU if available: CUDA_VISIBLE_DEVICES=0")
    print()
    print(f"  {Colors.BOLD}Issue: Index not found{Colors.RESET}")
    print("    → Run 'python -m student index' first")
    print()
