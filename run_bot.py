#!/usr/bin/env python3
"""Main Entrypoint for Portal-BOT: Autonomous Portal 1 Player."""

import argparse
import sys
import uvicorn

from portal_bot.config import BotConfig, CaptureConfig, UIConfig
from portal_bot.core.bot_engine import BotEngine
from portal_bot.ui.web_server import create_app
from portal_bot.utils.logger import bot_log


def main():
    parser = argparse.ArgumentParser(description="Portal 1 Autonomous Bot")
    parser.add_argument("--mock", action="store_true", help="Run with synthetic Portal chamber simulator")
    parser.add_argument("--chamber", type=int, default=0, help="Starting chamber index for simulator (0, 1, 2)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Web dashboard host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Web dashboard port (default: 8000)")
    parser.add_argument("--autostart", action="store_true", help="Automatically start bot without clicking Start in GUI")

    args = parser.parse_args()

    # Build configuration
    config = BotConfig(
        capture=CaptureConfig(
            use_mock=args.mock,
            mock_chamber_index=args.chamber,
            width=1280,
            height=720,
            target_fps=30
        ),
        ui=UIConfig(
            host=args.host,
            port=args.port
        )
    )

    bot_log.info("=" * 60)
    bot_log.info("  PORTAL 1 AUTONOMOUS BOT - APERTURE SCIENCE AGENT")
    bot_log.info(f"  Mode: {'MOCK SIMULATOR' if args.mock else 'REAL GAME WINDOW CAPTURE'}")
    bot_log.info(f"  Web Dashboard: http://{args.host}:{args.port}")
    bot_log.info("  Emergency Stop Key: F8")
    bot_log.info("=" * 60)

    # Initialize Bot Engine
    engine = BotEngine(config)

    if args.autostart:
        engine.start()

    # Create FastAPI app
    app = create_app(engine)

    # Launch Uvicorn Server
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    except KeyboardInterrupt:
        bot_log.info("Shutting down Portal-BOT...")
        engine.stop()


if __name__ == "__main__":
    main()
