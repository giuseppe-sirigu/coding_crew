#!/usr/bin/env bash
set -euo pipefail

########################################
# Coding Swarm – Full Setup Script
########################################

usage() {
  cat <<'EOF'
Usage: ./configure_swarm.sh -p PORT -m MODEL -d DIR [-b BACKEND]

Required:
  -p PORT      Port for the LLM API server (e.g. 8000)
  -m MODEL     Model identifier
                 vLLM:   Qwen/Qwen2.5-Coder-14B-Instruct-AWQ
                 Ollama: qwen2.5-coder:14b
  -d DIR       Absolute path to the project working directory

Optional:
  -b BACKEND   LLM backend: "vllm" or "ollama" (default: ollama)
  -h           Show this help

Examples:
  ./configure_swarm.sh -p 8000 -m Qwen/Qwen2.5-Coder-14B-Instruct-AWQ -d ~/my-project -b vllm
  ./configure_swarm.sh -p 11434 -m qwen2.5-coder:14b -d ~/my-project -b ollama
EOF
  exit 1
}

PORT=""
MODEL=""
DIR=""
BACKEND="ollama"

while getopts ":p:m:d:b:h" opt; do
  case ${opt} in
    p) PORT="${OPTARG}" ;;
    m) MODEL="${OPTARG}" ;;
    d) DIR="${OPTARG}" ;;
    b) BACKEND="${OPTARG}" ;;
    h) usage ;;
    :) echo "Error: -${OPTARG} requires an argument." >&2; usage ;;
    \?) echo "Error: invalid option -${OPTARG}" >&2; usage ;;
  esac
done

# Resolve ~ in DIR
DIR="${DIR/#\~/$HOME}"

if [[ -z "${PORT}" || -z "${MODEL}" || -z "${DIR}" ]]; then
  echo "Error: -p, -m, and -d are all required." >&2
  usage
fi

if [[ "${BACKEND}" != "vllm" && "${BACKEND}" != "ollama" ]]; then
  echo "Error: -b must be 'vllm' or 'ollama'." >&2
  usage
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo " Coding Swarm Setup"
echo "============================================"
echo " Port:      ${PORT}"
echo " Model:     ${MODEL}"
echo " Directory: ${DIR}"
echo " Backend:   ${BACKEND}"
echo "============================================"
echo ""

########################################
# Step 1: Python Virtual Environment
########################################
echo "--- Step 1: Python Virtual Environment ---"

mkdir -p "${DIR}"
VENV_DIR="${DIR}/.venv"

if [[ -d "${VENV_DIR}" ]]; then
  echo "Virtual environment already exists at ${VENV_DIR}"
else
  echo "Creating virtual environment at ${VENV_DIR}..."
  python3 -m venv "${VENV_DIR}"
  echo "Virtual environment created."
fi

echo "Installing Python dependencies..."
"${VENV_DIR}/bin/pip" install --upgrade pip --quiet
"${VENV_DIR}/bin/pip" install \
  mcp litellm crewai crewai-tools langchain-community langchain-openai \
  --quiet

if [[ "${BACKEND}" == "vllm" ]]; then
  echo "Installing vLLM..."
  "${VENV_DIR}/bin/pip" install vllm --quiet
fi

echo "Dependencies installed."
echo ""

########################################
# Step 2: Install LLM Backend
########################################
echo "--- Step 2: LLM Backend (${BACKEND}) ---"

if [[ "${BACKEND}" == "ollama" ]]; then
  if command -v ollama &>/dev/null; then
    echo "Ollama is already installed: $(ollama --version)"
  else
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "Ollama installed."
  fi

  echo "Pulling model ${MODEL}..."
  ollama pull "${MODEL}"
  echo "Model pulled successfully."

elif [[ "${BACKEND}" == "vllm" ]]; then
  echo "vLLM installed in virtual environment."
  echo "You can pre-download the model with:"
  echo "  huggingface-cli download ${MODEL}"
fi
echo ""

########################################
# Step 3: Install Continue VS Code Extension
########################################
echo "--- Step 3: Continue VS Code Extension ---"

if command -v code &>/dev/null; then
  if code --list-extensions 2>/dev/null | grep -qi "continue.continue"; then
    echo "Continue extension is already installed."
  else
    echo "Installing Continue extension..."
    code --install-extension continue.continue
    echo "Continue extension installed."
  fi
else
  echo "Warning: 'code' CLI not found. Install Continue manually from VS Code:"
  echo "  Extensions (Ctrl+Shift+X) → search 'Continue' → Install"
fi
echo ""

########################################
# Step 4: Configure Project Files
########################################
echo "--- Step 4: Configure Project Files ---"

CONFIG_TEMPLATE="${SCRIPT_DIR}/config.yaml"
PY_SRC="${SCRIPT_DIR}/swarm_mcp_server.py"

if [[ ! -f "${CONFIG_TEMPLATE}" ]]; then
  echo "Error: config.yaml not found at ${CONFIG_TEMPLATE}" >&2
  exit 2
fi
if [[ ! -f "${PY_SRC}" ]]; then
  echo "Error: swarm_mcp_server.py not found at ${PY_SRC}" >&2
  exit 2
fi

# Copy source files to project directory (if it differs from repo dir)
if [[ "$(realpath "${SCRIPT_DIR}")" != "$(realpath "${DIR}")" ]]; then
  cp "${PY_SRC}" "${DIR}/swarm_mcp_server.py"
  cp "${SCRIPT_DIR}/dashboard.py" "${DIR}/dashboard.py" 2>/dev/null || true
  echo "Copied swarm_mcp_server.py → ${DIR}/"
fi

chmod +x "${DIR}/dashboard.py" 2>/dev/null || true
chmod +x "${SCRIPT_DIR}/dashboard.py" 2>/dev/null || true

# Build config.yaml from template with placeholder substitution
CONFIG_OUT="${DIR}/config.yaml"
cp "${CONFIG_TEMPLATE}" "${CONFIG_OUT}"

sed -i "s|%%PORT%%|${PORT}|g"        "${CONFIG_OUT}"
sed -i "s|%%MODEL%%|${MODEL}|g"      "${CONFIG_OUT}"
sed -i "s|%%PROJECT_DIR%%|${DIR}|g"  "${CONFIG_OUT}"

echo "Generated ${CONFIG_OUT}"
echo ""

########################################
# Step 5: Install Continue Configuration
########################################
echo "--- Step 5: Install Continue Configuration ---"

CONTINUE_DIR="${HOME}/.continue"
mkdir -p "${CONTINUE_DIR}"

if [[ -f "${CONTINUE_DIR}/config.yaml" ]]; then
  BACKUP="${CONTINUE_DIR}/config.yaml.bak.$(date +%s)"
  cp "${CONTINUE_DIR}/config.yaml" "${BACKUP}"
  echo "Backed up existing config → ${BACKUP}"
fi

cp "${CONFIG_OUT}" "${CONTINUE_DIR}/config.yaml"
echo "Installed Continue config at ${CONTINUE_DIR}/config.yaml"
echo ""

########################################
# Step 6: Generate start_swarm.sh
########################################
echo "--- Step 6: Generate Startup Script ---"

START_SCRIPT="${DIR}/start_swarm.sh"

if [[ "${BACKEND}" == "vllm" ]]; then
  # Build the vLLM launch command
  VLLM_ARGS="--model ${MODEL} --host 0.0.0.0 --port ${PORT} --dtype auto --gpu-memory-utilization 0.85 --max-model-len 12000"
  if [[ "${MODEL}" == *"AWQ"* || "${MODEL}" == *"awq"* ]]; then
    VLLM_ARGS="--model ${MODEL} --quantization awq --host 0.0.0.0 --port ${PORT} --dtype auto --gpu-memory-utilization 0.85 --max-model-len 12000"
  fi

  cat > "${START_SCRIPT}" <<EOF
#!/usr/bin/env bash
set -euo pipefail

echo "Starting vLLM server on port ${PORT}..."
source "${VENV_DIR}/bin/activate"

python -m vllm.entrypoints.openai.api_server ${VLLM_ARGS} &
VLLM_PID=\$!
echo "vLLM started (PID: \$VLLM_PID)"

echo "Waiting for vLLM to be ready..."
for i in \$(seq 1 60); do
  if curl -s "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "vLLM is ready!"
    break
  fi
  if ! kill -0 \$VLLM_PID 2>/dev/null; then
    echo "Error: vLLM process exited unexpectedly." >&2
    exit 1
  fi
  sleep 5
done

echo ""
echo "Swarm is ready! Open VS Code and use Continue."
echo "To stop: kill \$VLLM_PID"
wait
EOF

elif [[ "${BACKEND}" == "ollama" ]]; then
  cat > "${START_SCRIPT}" <<EOF
#!/usr/bin/env bash
set -euo pipefail

echo "Starting Ollama..."

# Start Ollama if not already running
if ! pgrep -x ollama >/dev/null 2>&1; then
  OLLAMA_HOST=0.0.0.0:${PORT} ollama serve &
  OLLAMA_PID=\$!
  echo "Ollama started (PID: \$OLLAMA_PID)"
  sleep 5
else
  echo "Ollama is already running."
fi

# Ensure model is loaded
echo "Loading model ${MODEL}..."
ollama pull "${MODEL}" 2>/dev/null || true

echo ""
echo "Swarm is ready! Open VS Code and use Continue."
echo "Ollama API: http://localhost:${PORT}"
wait 2>/dev/null || true
EOF
fi

chmod +x "${START_SCRIPT}"
chmod +x "${DIR}/swarm_mcp_server.py" 2>/dev/null || true
echo "Created ${START_SCRIPT}"
echo ""

########################################
# Done
########################################
echo "============================================"
echo " Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Start the LLM server:"
echo "       ${START_SCRIPT}"
echo "  2. Open VS Code"
echo "  3. Use Continue (Ctrl+L) to chat with your coding swarm"
echo ""
echo "Optional – launch the monitoring dashboard:"
echo "  ${VENV_DIR}/bin/python ${DIR}/dashboard.py --port ${PORT} --backend ${BACKEND}"
echo ""
echo "The MCP server is launched automatically by Continue."
echo "MCP log file: /tmp/swarm_mcp_server.log"
