#!/usr/bin/env python3
"""
Launch Arize Phoenix tracing server for the Coding Crew.

Usage:
    python start_phoenix.py [--port PORT]

The Phoenix UI will be available at http://localhost:<PORT>
Traces are received at http://localhost:<PORT>/v1/traces
"""

import argparse
import os
import signal
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Start Phoenix tracing server")
    parser.add_argument(
        "--port",
        type=int,
        default=6006,
        help="Port for Phoenix server (default: 6006)",
    )
    args = parser.parse_args()

    try:
        import phoenix as px
    except ImportError:
        print(
            "Error: arize-phoenix is not installed.\n"
            "Install with: pip install arize-phoenix",
            file=sys.stderr,
        )
        sys.exit(1)

    # Use absolute path so data persists regardless of working directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    phoenix_dir = os.path.join(script_dir, ".phoenix_data")
    os.makedirs(phoenix_dir, exist_ok=True)
    os.environ.setdefault("PHOENIX_WORKING_DIR", phoenix_dir)

    print(f"Starting Phoenix tracing server on port {args.port}...")
    print(f"  UI:     http://localhost:{args.port}")
    print(f"  Traces: http://localhost:{args.port}/v1/traces")
    print(f"  Data:   {os.environ['PHOENIX_WORKING_DIR']}/")
    print()

    os.environ["PHOENIX_PORT"] = str(args.port)
    px.launch_app(use_temp_dir=False)

    if sys.stdin.isatty():
        try:
            input("Phoenix is running. Press Enter to stop...\n")
        except (KeyboardInterrupt, EOFError):
            pass
        print("\nShutting down Phoenix...")
    else:
        # Running in background (e.g. from start_swarm.sh &) — keep alive
        print("Phoenix is running in background mode (kill PID to stop).")
        try:
            while True:
                time.sleep(3600)
        except (KeyboardInterrupt, SystemExit):
            print("\nShutting down Phoenix...")


if __name__ == "__main__":
    main()
