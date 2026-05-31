#!/usr/bin/env python3
"""
System capacity checker for Docker deployment.
Verifies that the system meets requirements for running RAG.
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_status(name: str, status: bool, value: str = "") -> None:
    symbol = f"{Colors.GREEN}✓{Colors.RESET}" if status else f"{Colors.RED}✗{Colors.RESET}"
    msg = f"  {symbol} {name}"
    if value:
        msg += f": {value}"
    print(msg)


def get_disk_space() -> tuple[float, float]:
    """Get total and available disk space in GB."""
    stat = shutil.disk_usage("/")
    return stat.total / (1024**3), stat.free / (1024**3)


def get_memory() -> tuple[float, float]:
    """Get total and available memory in GB."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.total / (1024**3), mem.available / (1024**3)
    except ImportError:
        # Fallback for Windows without psutil
        if platform.system() == "Windows":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                c_ulong = ctypes.c_ulong

                class MemoryStatus(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", c_ulong),
                        ("dwMemoryLoad", c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                    ]

                stat = MemoryStatus()
                stat.dwLength = ctypes.sizeof(MemoryStatus)
                kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                return (
                    stat.ullTotalPhys / (1024**3),
                    stat.ullAvailPhys / (1024**3),
                )
            except Exception:
                return 0, 0
        return 0, 0


def get_cpu_count() -> int:
    """Get number of CPU cores."""
    return os.cpu_count() or 1


def check_docker() -> bool:
    """Check if Docker is installed and running."""
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            timeout=5,
            text=True,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_docker_running() -> bool:
    """Check if Docker daemon is running."""
    try:
        subprocess.run(
            ["docker", "ps"],
            capture_output=True,
            timeout=5,
        )
        return True
    except Exception:
        return False


def main() -> int:
    print(
        f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.RESET}"
    )
    print(
        f"{Colors.BOLD}{Colors.CYAN}RAG Against the Machine - System Capacity Check{Colors.RESET}"
    )
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.RESET}\n")

    issues = []

    # ===== Python ====
    print(f"{Colors.BOLD}Python{Colors.RESET}")
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    ok = sys.version_info >= (3, 10)
    print_status("Version", ok, f"{py_version} (requires ≥ 3.10)")
    if not ok:
        issues.append("Python version must be 3.10 or higher")

    # ===== System Resources ====
    print(f"\n{Colors.BOLD}System Resources{Colors.RESET}")
    
    # CPU
    cpus = get_cpu_count()
    ok = cpus >= 4
    print_status("CPU Cores", ok, f"{cpus} (≥ 4 recommended)")
    if not ok:
        issues.append("Less than 4 CPU cores; performance may be degraded")

    # Memory
    total_mem, avail_mem = get_memory()
    if total_mem > 0:
        ok_total = total_mem >= 4
        ok_avail = avail_mem >= 2
        print_status("Total RAM", ok_total, f"{total_mem:.1f} GB (≥ 4 GB min, 8 GB recommended)")
        print_status("Available RAM", ok_avail, f"{avail_mem:.1f} GB (≥ 2 GB available)")
        if not ok_total:
            issues.append("Total RAM < 4 GB; RAG may fail or be very slow")
        if not ok_avail:
            issues.append("Less than 2 GB available RAM; close some applications")
    
    # Disk
    total_disk, avail_disk = get_disk_space()
    ok = avail_disk >= 20
    print_status("Disk Space", ok, f"{avail_disk:.1f} GB free (≥ 20 GB recommended)")
    if not ok:
        issues.append("Less than 20 GB free disk space; models and index may not fit")

    # ===== Docker ====
    print(f"\n{Colors.BOLD}Docker Setup{Colors.RESET}")
    docker_installed = check_docker()
    docker_running = check_docker_running() if docker_installed else False
    
    print_status("Docker Installed", docker_installed)
    if not docker_installed:
        issues.append(
            "Docker is not installed. Install from https://www.docker.com/products/docker-desktop"
        )
    else:
        print_status("Docker Daemon Running", docker_running)
        if not docker_running:
            issues.append("Docker daemon is not running. Start Docker Desktop or daemon.")

    # ===== vLLM Corpus ====
    print(f"\n{Colors.BOLD}Project Structure{Colors.RESET}")
    corpus_path = Path("data/raw")
    corpus_exists = corpus_path.exists()
    print_status("Corpus Directory", corpus_exists, str(corpus_path))
    
    if corpus_exists:
        subdirs = list(corpus_path.glob("vllm-*"))
        if subdirs:
            print_status("vLLM Subdirectory", True, subdirs[0].name)
        else:
            issues.append(
                f"No vLLM-* subdirectory found in {corpus_path}. "
                "Extract vLLM zip or clone the repository."
            )
    else:
        issues.append(
            f"Corpus directory not found: {corpus_path}. "
            "Create and extract vLLM repository there."
        )

    # ===== Summary ====
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.RESET}")
    
    if not issues:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ All checks passed! System is ready for deployment.{Colors.RESET}\n")
        return 0
    else:
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠ {len(issues)} issue(s) found:{Colors.RESET}\n")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
