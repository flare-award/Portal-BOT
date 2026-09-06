"""Structured logger with live event broadcasting for Portal-BOT."""

from collections import deque
from datetime import datetime
import logging
import sys
from typing import Deque, Dict, List

from portal_bot.core.events import event_bus


class BotLogFormatter(logging.Formatter):
    """Formats log messages into readable timestamped entries."""
    def format(self, record: logging.LogRecord) -> str:
        t = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        msg = record.getMessage()
        return f"[{t}] {msg}"


class BotLogger:
    """Central logging coordinator."""

    def __init__(self, max_history: int = 500):
        self.history: Deque[Dict[str, str]] = deque(maxlen=max_history)
        self.logger = logging.getLogger("portal_bot")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        # Clear existing handlers if re-initialized
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(BotLogFormatter())
        self.logger.addHandler(handler)

    def log(self, category: str, message: str, level: int = logging.INFO):
        t_str = datetime.now().strftime("%H:%M:%S")
        full_msg = f"{category}: {message}" if category else message
        
        entry = {
            "time": t_str,
            "category": category,
            "message": message,
            "full_text": f"[{t_str}] {full_msg}",
            "level": logging.getLevelName(level)
        }
        self.history.append(entry)
        self.logger.log(level, full_msg)
        event_bus.publish("log_entry", entry)

    def info(self, msg: str):
        self.log("", msg, logging.INFO)

    def goal(self, msg: str):
        self.log("Goal", msg, logging.INFO)

    def vision(self, msg: str):
        self.log("Vision", msg, logging.INFO)

    def action(self, msg: str):
        self.log("Action", msg, logging.INFO)

    def state(self, msg: str):
        self.log("State", msg, logging.INFO)

    def recovery(self, msg: str):
        self.log("Recovery", msg, logging.WARNING)

    def warn(self, msg: str):
        self.log("Warning", msg, logging.WARNING)

    def error(self, msg: str):
        self.log("Error", msg, logging.ERROR)

    def get_recent_logs(self, limit: int = 100) -> List[Dict[str, str]]:
        return list(self.history)[-limit:]


# Global singleton logger
bot_log = BotLogger()
