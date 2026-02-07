#!/bin/bash

# Create a tmux session with multiple panes
tmux new-session -d -s llm-monitor

# Pane 1: GPU monitoring
tmux send-keys -t llm-monitor 'watch -n 1 nvidia-smi' C-m

# Split horizontally
tmux split-window -h -t llm-monitor

# Pane 2: vLLM logs
tmux send-keys -t llm-monitor 'tail -f /tmp/vllm.log' C-m

# Split vertically
tmux split-window -v -t llm-monitor

# Pane 3: MCP server logs
tmux send-keys -t llm-monitor 'tail -f /tmp/swarm_mcp_server.log' C-m

# Attach to the session
tmux attach-session -t llm-monitor