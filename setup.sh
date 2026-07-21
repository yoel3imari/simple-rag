#!/usr/bin/env bash
# ===========================================================================
#  setup.sh — Portable one-shot environment bootstrap for simple-rag
#
#  Detects the host OS (macOS / Linux / Windows via WSL or Git Bash),
#  installs system dependencies, Python tooling, Ollama, and verifies
#  the entire pipeline in one shot.
#
#  Usage:
#    bash setup.sh
#
#  Idempotent: safe to re-run; skips steps that are already satisfied.
# ===========================================================================
set -euo pipefail

# ---- Constants -------------------------------------------------------------

TESSDATA_URL="https://github.com/tesseract-ocr/tessdata/raw/main/ara.traineddata"
UV_INSTALL_URL="https://astral.sh/uv/install.sh"
OLLAMA_LINUX_URL="https://ollama.com/install.sh"
OLLAMA_WINDOWS_URL="https://ollama.com/download/windows"
CROSS_ENCODER_REPO="Xenova/ms-marco-MiniLM-L-6-v2"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$PROJECT_DIR/src"

# ---- Helper functions ------------------------------------------------------

print_step() {
  printf "\n━━━ [%s] %s ━━━\n" "$(date +%H:%M:%S)" "$*"
}

print_success() {
  printf "  ✓ %s\n" "$*"
}

print_info() {
  printf "  ℹ %s\n" "$*"
}

print_warning() {
  printf "  ⚠ %s\n" "$*"
}

print_error() {
  printf "  ✘ %s\n" "$*"
}

check_command() {
  if command -v "$1" &>/dev/null; then
    print_success "$1 found: $(command -v "$1")"
    return 0
  fi
  return 1
}

# ---- OS Detection ----------------------------------------------------------

detect_os() {
  local uname_s
  uname_s="$(uname -s)"

  case "$uname_s" in
    Darwin) 
      echo "macos"
      ;;
    Linux)
      echo "linux"
      ;;
    MINGW*|MSYS*|CYGWIN*)
      echo "windows"
      ;;
    *)
      print_error "Unsupported OS: $uname_s"
      exit 1
      ;;
  esac
}

detect_linux_distro() {
  if command -v apt-get &>/dev/null; then
    echo "debian"
  elif command -v dnf &>/dev/null; then
    echo "fedora"
  elif command -v yum &>/dev/null; then
    echo "rhel"
  elif command -v pacman &>/dev/null; then
    echo "arch"
  elif command -v zypper &>/dev/null; then
    echo "suse"
  else
    echo "unknown"
  fi
}

is_wsl() {
  # Check if we're running inside WSL
  if [ -f /proc/version ] && grep -qi microsoft /proc/version 2>/dev/null; then
    return 0
  fi
  return 1
}

# ---- Step 2: Install Tesseract + Arabic language data ----------------------

install_tesseract() {
  print_step "Installing Tesseract OCR + Arabic language data"

  if check_command tesseract; then
    print_info "Tesseract already installed, checking Arabic data..."
    if tesseract --list-langs 2>&1 | grep -q "^ara$"; then
      print_success "Arabic language data already available"
      return 0
    fi
    print_info "Arabic data missing — will attempt install"
  fi

  local os
  os="$(detect_os)"

  case "$os" in
    macos)
      if ! check_command brew; then
        print_error "Homebrew is required. Install from https://brew.sh/"
        exit 1
      fi
      brew install tesseract tesseract-lang
      ;;

    linux)
      local distro
      distro="$(detect_linux_distro)"

      case "$distro" in
        debian)
          sudo apt-get update -qq
          sudo apt-get install -y tesseract-ocr tesseract-ocr-ara
          ;;
        fedora)
          sudo dnf install -y tesseract tesseract-langpack-ara
          ;;
        rhel)
          sudo yum install -y tesseract tesseract-langpack-ara
          ;;
        arch)
          sudo pacman -S --noconfirm tesseract tesseract-data-ara
          ;;
        suse)
          sudo zypper install -y tesseract tesseract-ocr-langpack-ara
          ;;
        *)
          print_error "Unsupported Linux distro. Install Tesseract manually."
          print_info "See: https://github.com/tesseract-ocr/tesseract"
          exit 1
          ;;
      esac
      ;;

    windows)
      # Git Bash / WSL on Windows
      if check_command choco; then
        choco install tesseract -y
      elif check_command winget; then
        winget install --id=Tesseract-OCR.Tesseract -e
      elif check_command scoop; then
        scoop install tesseract
      else
        print_warning "No supported package manager found (choco/winget/scoop)."
        print_info "Download Tesseract manually from: https://github.com/UB-Mannheim/tesseract/wiki"
        print_info "Then re-run this script."
        exit 1
      fi
      ;;
  esac

  # Verify tesseract is now available
  if ! check_command tesseract; then
    print_error "Tesseract installation failed."
    exit 1
  fi

  # Check if Arabic data is now available
  if tesseract --list-langs 2>&1 | grep -q "^ara$"; then
    print_success "Arabic language data installed successfully"
  else
    print_warning "Arabic data not bundled with the package — will download manually"
  fi
}

# ---- Step 3: Download Arabic tessdata fallback -----------------------------

download_ara_tessdata() {
  print_step "Downloading Arabic tessdata fallback"

  # First check if ara is already available
  if command -v tesseract &>/dev/null && tesseract --list-langs 2>&1 | grep -q "^ara$"; then
    print_success "Arabic data already present — skipping download"
    return 0
  fi

  # Locate tessdata directory
  local tessdata_dir=""

  # Method 1: Parse tesseract --print-parameters
  if command -v tesseract &>/dev/null; then
    tessdata_dir="$(tesseract --print-parameters 2>/dev/null | \
      grep -E '^tessdata_dir[[:space:]]+' | \
      awk '{print $2}')"
  fi

  # Method 2: Common system paths
  if [ -z "$tessdata_dir" ] || [ ! -d "$tessdata_dir" ]; then
    for candidate in \
      /opt/homebrew/share/tessdata \
      /usr/local/share/tessdata \
      /usr/share/tesseract-ocr/5/tessdata \
      /usr/share/tesseract-ocr/4/tessdata \
      /usr/share/tesseract-ocr/tessdata \
      /usr/share/tessdata; do
      if [ -d "$candidate" ]; then
        tessdata_dir="$candidate"
        break
      fi
    done
  fi

  # Method 3: Ask tesseract for its version-specific path
  if [ -z "$tessdata_dir" ] || [ ! -d "$tessdata_dir" ]; then
    if command -v tesseract &>/dev/null; then
      local ver
      ver="$(tesseract --version 2>&1 | head -1 | grep -oP '\d+\.\d+\.\d+' | head -1)"
      if [ -n "$ver" ]; then
        local major
        major="${ver%%.*}"
        if [ -d "/usr/share/tesseract-ocr/${major}/tessdata" ]; then
          tessdata_dir="/usr/share/tesseract-ocr/${major}/tessdata"
        fi
      fi
    fi
  fi

  # Method 4: On Windows, check common install paths
  if [ -z "$tessdata_dir" ] || [ ! -d "$tessdata_dir" ]; then
    local os
    os="$(detect_os)"
    if [ "$os" = "windows" ]; then
      # Common Windows paths
      for candidate in \
        "/c/Program Files/Tesseract-OCR/tessdata" \
        "/c/Program Files (x86)/Tesseract-OCR/tessdata" \
        "$PROGRAMFILES/Tesseract-OCR/tessdata" \
        "$LOCALAPPDATA/Programs/Tesseract-OCR/tessdata"; do
        if [ -d "$candidate" ]; then
          tessdata_dir="$candidate"
          break
        fi
      done
    fi
  fi

  # If still not found, create a fallback directory
  if [ -z "$tessdata_dir" ] || [ ! -d "$tessdata_dir" ]; then
    tessdata_dir="/tmp/tessdata"
    mkdir -p "$tessdata_dir"
    print_warning "Could not detect tessdata directory — using $tessdata_dir"
    print_warning "Set TESSDATA_PREFIX=$tessdata_dir if needed"
  fi

  print_info "Tessdata directory: $tessdata_dir"

  # Download ara.traineddata if not present
  if [ -f "$tessdata_dir/ara.traineddata" ]; then
    print_success "ara.traineddata already exists in $tessdata_dir"
  else
    print_info "Downloading ara.traineddata..."
    curl -#fSL "$TESSDATA_URL" -o "$tessdata_dir/ara.traineddata"
    print_success "Downloaded ara.traineddata to $tessdata_dir"
  fi

  # Verify the download was successful and tesseract can see it
  if command -v tesseract &>/dev/null; then
    if tesseract --list-langs 2>&1 | grep -q "^ara$"; then
      print_success "Arabic language data verified"
    else
      # Try passing --tessdata-dir to see if it's picked up
      if tesseract --tessdata-dir "$tessdata_dir" --list-langs 2>&1 | grep -q "^ara$"; then
        print_success "Arabic data available with --tessdata-dir $tessdata_dir"
        print_warning "You may need to set TESSDATA_PREFIX=$tessdata_dir"
        print_warning "Add the following to your shell profile:"
        printf '  export TESSDATA_PREFIX=%s\n' "$tessdata_dir"
      else
        print_warning "Arabic data downloaded but tesseract cannot find it"
        print_warning "The app's _resolve_tessdata_dir() will attempt to locate it at runtime"
      fi
    fi
  fi
}

# ---- Step 4: Install uv (Python package manager) ---------------------------

install_uv() {
  print_step "Installing uv (Python package manager)"

  if check_command uv; then
    local uv_ver
    uv_ver="$(uv --version)"
    print_success "uv already installed: $uv_ver"
    return 0
  fi

  print_info "Downloading uv installer..."
  curl -LsSf "$UV_INSTALL_URL" | sh

  # Add uv to PATH for this session
  if [ -f "$HOME/.cargo/env" ]; then
    # shellcheck disable=SC1091
    . "$HOME/.cargo/env"
  fi
  export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

  if check_command uv; then
    print_success "uv installed: $(uv --version)"
  else
    print_error "uv installation failed."
    print_info "Try installing manually: https://docs.astral.sh/uv/#installation"
    exit 1
  fi
}

# ---- Step 5: Install Python dependencies -----------------------------------

install_python_deps() {
  print_step "Installing Python dependencies (uv sync)"

  if [ ! -f "$PROJECT_DIR/pyproject.toml" ]; then
    print_error "pyproject.toml not found in $PROJECT_DIR"
    exit 1
  fi

  cd "$PROJECT_DIR"
  uv sync
  print_success "Python dependencies installed via uv sync"
}

# ---- Step 6: Install Ollama ------------------------------------------------

install_ollama() {
  print_step "Installing Ollama"

  if check_command ollama; then
    print_success "Ollama already installed: $(ollama --version 2>/dev/null || true)"
  else
    local os
    os="$(detect_os)"

    case "$os" in
      macos)
        if ! check_command brew; then
          print_error "Homebrew is required. Install from https://brew.sh/"
          exit 1
        fi
        brew install --cask ollama
        ;;

      linux)
        print_info "Downloading and running Ollama install script..."
        curl -fsSL "$OLLAMA_LINUX_URL" | sh
        ;;

      windows)
        print_info "Windows detected."
        print_info "Download Ollama from: $OLLAMA_WINDOWS_URL"
        print_info "After installing, re-run this script."
        exit 1
        ;;
    esac

    if ! check_command ollama; then
      print_error "Ollama installation failed."
      exit 1
    fi
  fi

  # Start Ollama service if not running
  print_info "Ensuring Ollama service is running..."
  if ! curl -sf http://localhost:11434 >/dev/null 2>&1; then
    print_info "Starting Ollama in background..."
    # Try launching via systemd (Linux) first
    if command -v systemctl &>/dev/null; then
      sudo systemctl start ollama 2>/dev/null || true
    fi
    # Fall back to background process
    if ! curl -sf http://localhost:11434 >/dev/null 2>&1; then
      ollama serve &
      # Give it time to start
      sleep 2
    fi
  fi

  # Wait for Ollama to be ready
  print_info "Waiting for Ollama to be ready..."
  local max_attempts=30
  local attempt=0
  while ! curl -sf http://localhost:11434 >/dev/null 2>&1; do
    attempt=$((attempt + 1))
    if [ "$attempt" -ge "$max_attempts" ]; then
      print_error "Ollama did not start within $max_attempts seconds."
      print_info "Try starting Ollama manually and re-run this script."
      exit 1
    fi
    sleep 2
  done
  print_success "Ollama is ready"
}

# ---- Step 7: Pull Ollama models -------------------------------------------

pull_ollama_models() {
  print_step "Pulling Ollama models"

  local models=("nomic-embed-text" "gemma3")

  for model in "${models[@]}"; do
    # Check if model already pulled
    if ollama list 2>/dev/null | grep -q "^${model}\s"; then
      print_success "Model '$model' already pulled"
    else
      print_info "Pulling model '$model' (this may take a while)..."
      ollama pull "$model"
      print_success "Model '$model' pulled successfully"
    fi
  done
}

# ---- Step 8: Download cross-encoder ONNX model ----------------------------

download_reranker() {
  print_step "Downloading cross-encoder ONNX model"

  if [ ! -d "$SRC_DIR" ]; then
    print_error "Source directory not found: $SRC_DIR"
    exit 1
  fi

  # Check if model files are already cached by huggingface-hub
  print_info "Triggering cross-encoder download via rerank module..."
  if cd "$PROJECT_DIR" && uv run python3 -c "
from src.rerank import _snapshot_dir
snap = _snapshot_dir()
print(f'Cross-encoder cached at: {snap}')
" 2>/dev/null; then
    print_success "Cross-encoder ONNX model downloaded and cached"
  else
    print_warning "Could not verify cross-encoder download via Python."
    print_warning "The model will be auto-downloaded on first use."
    print_info "You can manually trigger it later with:"
    print_info "  cd $PROJECT_DIR && uv run python3 -c \"from src.rerank import _snapshot_dir; print(_snapshot_dir())\""
  fi
}

# ---- Step 9: Verify everything -------------------------------------------

verify_all() {
  print_step "Verifying installation"

  local errors=0

  # 1. Check tesseract + Arabic
  if command -v tesseract &>/dev/null; then
    print_success "tesseract: $(tesseract --version 2>&1 | head -1)"
    if tesseract --list-langs 2>&1 | grep -q "^ara$"; then
      print_success "tesseract Arabic (ara): available"
    else
      print_warning "tesseract Arabic (ara): NOT found (download may be needed)"
    fi
  else
    print_warning "tesseract: NOT found (OCR fallback will be unavailable)"
  fi

  # 2. Check uv
  if command -v uv &>/dev/null; then
    print_success "uv: $(uv --version)"
  else
    print_error "uv: NOT found"
    errors=$((errors + 1))
  fi

  # 3. Check Python virtual env / deps
  if [ -d "$PROJECT_DIR/.venv" ]; then
    print_success "Python virtual env: $PROJECT_DIR/.venv"
  else
    print_warning "Python virtual env not found (.venv)"
  fi

  # 4. Check Ollama
  if command -v ollama &>/dev/null; then
    print_success "ollama: $(ollama --version 2>/dev/null || echo 'installed')"
    if curl -sf http://localhost:11434 >/dev/null 2>&1; then
      print_success "Ollama service: running"

      # Check models
      for model in nomic-embed-text gemma3; do
        if ollama list 2>/dev/null | grep -q "^${model}\s"; then
          print_success "ollama model '$model': pulled"
        else
          print_warning "ollama model '$model': NOT pulled"
        fi
      done
    else
      print_warning "Ollama service: NOT running (start with 'ollama serve')"
    fi
  else
    print_warning "ollama: NOT installed"
  fi

  # 5. Quick pipeline check (ingest --status)
  if [ -f "$PROJECT_DIR/.venv/bin/python" ]; then
    print_info "Running 'uv run -m src.main ingest --status'..."
    cd "$PROJECT_DIR"
    uv run -m src.main ingest --status 2>&1 || {
      print_warning "Pipeline check returned non-zero (may be expected if no data ingested)"
    }
    print_success "Pipeline check completed"
  else
    print_warning "Python venv not found — skipping pipeline check"
  fi

  echo ""
  if [ "$errors" -eq 0 ]; then
    print_success "All critical components verified!"
  else
    print_warning "$errors critical component(s) missing — see above"
  fi
}

# ---- Main ------------------------------------------------------------------

main() {
  echo ""
  echo "  ╔══════════════════════════════════════════════════╗"
  echo "  ║        simple-rag — Environment Setup            ║"
  echo "  ╚══════════════════════════════════════════════════╝"
  echo ""
  echo "  Project  : $PROJECT_DIR"
  echo "  OS       : $(detect_os)"
  if [ "$(detect_os)" = "linux" ]; then
    echo "  Distro   : $(detect_linux_distro)"
  fi
  echo "  Date     : $(date)"
  echo ""

  # ---- Run all steps sequentially ----

  install_tesseract
  download_ara_tessdata
  install_uv
  install_python_deps
  install_ollama
  pull_ollama_models
  download_reranker
  verify_all

  # ---- Final message ----
  echo ""
  echo "  ╔══════════════════════════════════════════════════╗"
  echo "  ║                 Setup Complete!                  ║"
  echo "  ╚══════════════════════════════════════════════════╝"
  echo ""
  echo "  Next steps:"
  echo ""
  echo "    Ingest a document:"
  echo "      uv run -m src.main ingest data/sample.txt"
  echo ""
  echo "    Ask a question:"
  echo "      uv run -m src.main query \"What are the failure modes of RAG?\""
  echo ""
  echo "    Check ingested files:"
  echo "      uv run -m src.main ingest --status"
  echo ""
  echo "  Happy RAGging! 🚀"
  echo ""
}

main "$@"
