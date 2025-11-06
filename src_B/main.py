"""Main entry point for the meeting transcription system.

This module provides the main entry point for running the system in either:
1. Web API mode (FastAPI server)
2. Terminal mode (CLI interface)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def run_api_mode(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """
    Run the system in API mode (FastAPI server).

    Args:
        host: Host address to bind to
        port: Port number to listen on
        reload: Enable auto-reload for development
    """
    import uvicorn

    from src_B.api.app import app

    logger = logging.getLogger(__name__)
    logger.info(f"Starting API server on {host}:{port}")
    logger.info("Web UI has been removed. API endpoints only.")
    logger.info("API documentation available at: http://{}:{}/docs".format(host if host != "0.0.0.0" else "localhost", port))

    uvicorn.run(
        "src_B.api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


def run_terminal_mode() -> None:
    """Run the system in terminal mode (CLI interface)."""
    from src_B.terminal_interface import main as terminal_main

    terminal_main()


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Meeting Transcription System with Speaker Diarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run API server (default mode)
  python -m src_B.main

  # Run API server on custom port
  python -m src_B.main --host 0.0.0.0 --port 8080

  # Run API server with auto-reload (development)
  python -m src_B.main --reload

  # Run in terminal mode (interactive CLI)
  python -m src_B.main --terminal

  # Terminal mode with specific topic and speakers
  python -m src_B.main --terminal --topic "Sprint Planning" --speakers 4
        """,
    )

    # Mode selection
    parser.add_argument(
        "--terminal",
        action="store_true",
        help="Run in terminal mode (CLI interface)",
    )

    # API mode arguments
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address for API server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number for API server (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development (API mode only)",
    )

    # Terminal mode arguments
    parser.add_argument(
        "--topic",
        type=str,
        help="Meeting topic (terminal mode only)",
    )
    parser.add_argument(
        "--speakers",
        type=int,
        default=2,
        help="Expected number of speakers (terminal mode only, default: 2)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        help="Force specific device (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to Whisper configuration file",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )

    # Run in appropriate mode
    if args.terminal:
        # Terminal mode
        run_terminal_mode()
    else:
        # API mode (default)
        run_api_mode(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
