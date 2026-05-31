#!/usr/bin/env pwsh
# RAG Against the Machine - Quick Setup Script for Windows
# Run: .\setup.ps1

$ErrorActionPreference = "Stop"

Write-Host "`n" + ("=" * 70) -ForegroundColor Cyan
Write-Host "RAG Against the Machine - Quick Setup" -ForegroundColor Cyan
Write-Host ("=" * 70) + "`n" -ForegroundColor Cyan

# ============================================================================
# Step 1: Add Docker to PATH
# ============================================================================
Write-Host "Step 1: Configuring Docker" -ForegroundColor Blue
$dockerPath = "C:\Program Files\Docker\Docker\resources\bin"
if (Test-Path $dockerPath) {
    $env:PATH += ";$dockerPath"
    Write-Host "  ✓ Docker path added" -ForegroundColor Green
} else {
    Write-Host "  ✗ Docker not found at $dockerPath" -ForegroundColor Red
    Write-Host "    Please install Docker Desktop from https://www.docker.com" -ForegroundColor Yellow
    exit 1
}

# ============================================================================
# Step 2: Verify Docker is working
# ============================================================================
Write-Host "Step 2: Verifying Docker installation" -ForegroundColor Blue
try {
    $version = docker --version
    Write-Host "  ✓ $version" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Docker command failed. Make sure Docker Desktop is running." -ForegroundColor Red
    exit 1
}

try {
    docker ps > $null 2>&1
    Write-Host "  ✓ Docker daemon is running" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Docker daemon not running. Start Docker Desktop and try again." -ForegroundColor Red
    exit 1
}

# ============================================================================
# Step 3: Check system capacity
# ============================================================================
Write-Host "Step 3: Checking system capacity" -ForegroundColor Blue
python check_capacity.py
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ All checks passed" -ForegroundColor Green
} else {
    Write-Host "  ⚠ Some checks failed (see above)" -ForegroundColor Yellow
}

# ============================================================================
# Step 4: Check for vLLM corpus
# ============================================================================
Write-Host "Step 4: Checking for vLLM corpus" -ForegroundColor Blue
$corpusPaths = @(
    "data/raw/vllm-0.10.1",
    "data/raw/vllm",
    "./data/raw/vllm-*"
)

$found = $false
foreach ($path in $corpusPaths) {
    if (Test-Path $path) {
        Write-Host "  ✓ Found corpus at: $path" -ForegroundColor Green
        $found = $true
        break
    }
}

if (-not $found) {
    Write-Host "  ✗ vLLM corpus not found" -ForegroundColor Yellow
    Write-Host "    Please extract the vLLM repository to data/raw/" -ForegroundColor Yellow
    Write-Host "    Example:" -ForegroundColor Cyan
    Write-Host "      Expand-Archive -Path vllm-0.10.1.zip -DestinationPath data/raw/" -ForegroundColor Cyan
}

# ============================================================================
# Step 5: Build Docker image (optional)
# ============================================================================
Write-Host "Step 5: Docker image build" -ForegroundColor Blue
Write-Host "  To build the Docker image, run:" -ForegroundColor Cyan
Write-Host "    docker build -t ragagainsthemachine:latest ." -ForegroundColor Cyan
Write-Host "  Or use the Makefile:" -ForegroundColor Cyan
Write-Host "    make docker-build" -ForegroundColor Cyan

# ============================================================================
# Summary
# ============================================================================
Write-Host "`n" + ("=" * 70) -ForegroundColor Cyan
Write-Host "Setup Summary" -ForegroundColor Cyan
Write-Host ("=" * 70) -ForegroundColor Cyan

Write-Host "`n✓ Docker is configured and working" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "  1. Extract vLLM corpus to data/raw/vllm-0.10.1/" -ForegroundColor White
Write-Host "  2. Run: docker build -t ragagainsthemachine:latest ." -ForegroundColor White
Write-Host "  3. Run: docker-compose run --rm rag help" -ForegroundColor White
Write-Host "  4. Run: docker-compose run --rm rag index" -ForegroundColor White
Write-Host "  5. Run: docker-compose run --rm rag answer 'your question'" -ForegroundColor White

Write-Host "`nDocumentation:" -ForegroundColor Cyan
Write-Host "  • README.md - Project overview and commands" -ForegroundColor DarkGray
Write-Host "  • DEPLOYMENT.md - Detailed deployment guide" -ForegroundColor DarkGray
Write-Host "  • IMPROVEMENTS_SUMMARY.md - What's new in this version" -ForegroundColor DarkGray

Write-Host "`n" + ("=" * 70) + "`n" -ForegroundColor Cyan
