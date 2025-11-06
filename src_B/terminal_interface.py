"""Terminal interface for running the meeting transcription system from CLI."""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from src_B.api.state import (
    meeting_state,
    meeting_stats,
    transcription_service,
    diarization_service,
    ollama_evaluator,
    ensure_diarization_service,
    handle_transcription,
    handle_diarization_segment,
)
from src_B.config.config import select_device
from src_B.models.data_models import TranscribedSegment, DiarizationSegment
from src_B.reports import finalize_meeting_report

logger = logging.getLogger(__name__)


class TerminalInterface:
    """Terminal interface for running meeting transcription."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize terminal interface.

        Args:
            config_path: Optional path to Whisper config file
        """
        self.config_path = config_path
        self.running = False
        self.meeting_started = False
        self.device = None

    def setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(sig, frame):
            logger.info("\n\nReceived interrupt signal. Stopping...")
            self.stop_meeting()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def initialize_services(self, topic: str, expected_speakers: int, force_device: Optional[str] = None) -> bool:
        """
        Initialize transcription and diarization services.

        Args:
            topic: Meeting topic
            expected_speakers: Number of expected speakers
            force_device: Force specific device (cpu, cuda, mps)

        Returns:
            True if initialization successful
        """
        try:
            # Select device
            self.device, dtype = select_device(force_device)
            logger.info(f"Using device: {self.device}")

            # Initialize transcription service
            global transcription_service
            if not transcription_service.initialize():
                logger.error("Failed to initialize transcription service")
                return False

            # Initialize diarization service
            diarization_svc = ensure_diarization_service(self.device, expected_speakers)
            diarization_svc.set_segment_callback(self._handle_diarization_segment)

            # Attach diarization to transcription
            transcription_service.attach_diarization(diarization_svc)

            # Initialize evaluator
            global ollama_evaluator
            ollama_evaluator.initialize(topic)

            logger.info("All services initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize services: {e}", exc_info=True)
            return False

    def start_meeting(self, topic: str, expected_speakers: int, speaker_id: str = "Speaker 1") -> bool:
        """
        Start a meeting transcription session.

        Args:
            topic: Meeting topic
            expected_speakers: Number of expected speakers
            speaker_id: Default speaker ID

        Returns:
            True if started successfully
        """
        try:
            # Initialize services
            if not self.initialize_services(topic, expected_speakers):
                return False

            # Update meeting state
            meeting_state.topic = topic
            meeting_state.speaker_id = speaker_id
            meeting_state.expected_speakers = expected_speakers
            meeting_state.is_active = True
            meeting_state.session_id = datetime.utcnow().isoformat()

            # Reset statistics
            meeting_stats.reset()

            # Start transcription
            transcription_service.start(callback=self._handle_transcription)

            # Start diarization
            global diarization_service
            if diarization_service:
                diarization_service.start()

            self.meeting_started = True
            self.running = True

            logger.info(f"\n{'=' * 60}")
            logger.info(f"Meeting Started: {topic}")
            logger.info(f"Session ID: {meeting_state.session_id}")
            logger.info(f"Expected Speakers: {expected_speakers}")
            logger.info(f"{'=' * 60}\n")

            return True

        except Exception as e:
            logger.error(f"Failed to start meeting: {e}", exc_info=True)
            return False

    def stop_meeting(self) -> Optional[Path]:
        """
        Stop the current meeting session.

        Returns:
            Path to generated report PDF, or None if failed
        """
        if not self.meeting_started:
            logger.warning("No active meeting to stop")
            return None

        try:
            logger.info("\nStopping meeting...")

            # Stop services
            transcription_service.stop()

            global diarization_service
            if diarization_service:
                diarization_service.stop()

            # Update meeting state
            meeting_state.is_active = False
            self.running = False
            self.meeting_started = False

            logger.info("Services stopped successfully")

            # Generate report
            logger.info("Generating meeting report...")

            from src_B.api.state import meeting_history

            report_path, summary = finalize_meeting_report(
                session_id=meeting_state.session_id or "unknown",
                topic=meeting_state.topic,
                overall_stats=meeting_stats.overall_dict(),
                speaker_stats=meeting_stats.speaker_dict(transcription_service),
                history=list(meeting_history),
            )

            if report_path:
                logger.info(f"\nReport generated: {report_path}")
                logger.info(f"{'=' * 60}\n")
            else:
                logger.warning("Failed to generate report")

            return report_path

        except Exception as e:
            logger.error(f"Error stopping meeting: {e}", exc_info=True)
            return None

    def _handle_transcription(self, segment: TranscribedSegment) -> None:
        """Handle transcription callback (terminal display)."""
        handle_transcription(segment)

        # Print to terminal
        timestamp = segment.start_time.strftime("%H:%M:%S")
        similarity_str = f"({segment.similarity:.2f})" if segment.similarity < 1.0 else ""
        print(f"[{timestamp}] {segment.speaker_name}{similarity_str}: {segment.text}")

    def _handle_diarization_segment(self, segment: DiarizationSegment) -> None:
        """Handle diarization callback."""
        handle_diarization_segment(segment)

    def run_interactive(self) -> None:
        """Run interactive terminal session."""
        print("\n" + "=" * 60)
        print("Meeting Transcription System - Terminal Mode")
        print("=" * 60 + "\n")

        # Get meeting details
        topic = input("Enter meeting topic: ").strip()
        if not topic:
            topic = "General Meeting"

        try:
            expected_speakers_str = input("Enter expected number of speakers (default: 2): ").strip()
            expected_speakers = int(expected_speakers_str) if expected_speakers_str else 2
        except ValueError:
            expected_speakers = 2

        # Start meeting
        if not self.start_meeting(topic, expected_speakers):
            logger.error("Failed to start meeting")
            return

        # Main loop
        print("\nTranscription started. Press Ctrl+C to stop...\n")

        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass

        # Stop meeting
        self.stop_meeting()


def main():
    """Main entry point for terminal interface."""
    parser = argparse.ArgumentParser(
        description="Meeting Transcription System - Terminal Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python -m src_B.terminal_interface --terminal

  # Specify topic and speakers
  python -m src_B.terminal_interface --terminal --topic "Project Planning" --speakers 3

  # Use specific device
  python -m src_B.terminal_interface --terminal --device cuda

  # With custom config
  python -m src_B.terminal_interface --terminal --config config_whisper.json
        """,
    )

    parser.add_argument(
        "--terminal",
        action="store_true",
        help="Run in terminal mode (interactive CLI)",
    )
    parser.add_argument(
        "--topic",
        type=str,
        help="Meeting topic (if not provided, will prompt)",
    )
    parser.add_argument(
        "--speakers",
        type=int,
        default=2,
        help="Expected number of speakers (default: 2)",
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
    )

    # Check if terminal mode is requested
    if not args.terminal:
        parser.print_help()
        sys.exit(1)

    # Create terminal interface
    interface = TerminalInterface(config_path=args.config)
    interface.setup_signal_handlers()

    # Run interactive or with provided args
    if args.topic:
        # Non-interactive mode with provided args
        if not interface.start_meeting(args.topic, args.speakers):
            sys.exit(1)

        print("\nTranscription started. Press Ctrl+C to stop...\n")

        try:
            while interface.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass

        interface.stop_meeting()
    else:
        # Interactive mode
        interface.run_interactive()


if __name__ == "__main__":
    main()
