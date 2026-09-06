"""Helper to generate and install Portal 1 config for log streaming."""

import os
import platform
from typing import List, Optional

from portal_bot.utils.logger import bot_log

PORTAL_CFG_CONTENT = """// Portal-BOT Auto-Configuration
// Automatically enables console logging for real-time state analysis
con_logfile "portal_bot_log.txt"
developer 1
echo "[PORTAL_BOT_CONFIG_LOADED]"
"""


def find_portal_cfg_dirs() -> List[str]:
    """Finds directories where Portal 1 cfg files live."""
    system = platform.system()
    home = os.path.expanduser("~")
    dirs: List[str] = []

    if system == "Windows":
        drives = ["C:", "D:", "E:", "F:"]
        for d in drives:
            p1 = os.path.join(d, "Program Files (x86)", "Steam", "steamapps", "common", "Portal", "portal", "cfg")
            p2 = os.path.join(d, "SteamLibrary", "steamapps", "common", "Portal", "portal", "cfg")
            if os.path.exists(p1):
                dirs.append(p1)
            if os.path.exists(p2):
                dirs.append(p2)
    elif system == "Linux":
        p1 = os.path.join(home, ".local", "share", "Steam", "steamapps", "common", "Portal", "portal", "cfg")
        p2 = os.path.join(home, ".steam", "steam", "steamapps", "common", "Portal", "portal", "cfg")
        if os.path.exists(p1):
            dirs.append(p1)
        if os.path.exists(p2):
            dirs.append(p2)

    return dirs


def install_portal_cfg() -> bool:
    """Installs portal_bot.cfg into detected Portal 1 game directories."""
    cfg_dirs = find_portal_cfg_dirs()
    if not cfg_dirs:
        # Write to local current directory as fallback
        with open("portal_bot.cfg", "w", encoding="utf-8") as f:
            f.write(PORTAL_CFG_CONTENT)
        bot_log.info("portal_bot.cfg created in current directory (copy to portal/cfg/ if needed)")
        return True

    success = False
    for cd in cfg_dirs:
        try:
            target = os.path.join(cd, "portal_bot.cfg")
            with open(target, "w", encoding="utf-8") as f:
                f.write(PORTAL_CFG_CONTENT)
            bot_log.info(f"Installed portal_bot.cfg into: {target}")
            success = True
        except Exception as e:
            bot_log.warn(f"Failed writing to {cd}: {e}")

    return success
