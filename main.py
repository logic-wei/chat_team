"""Entry point: ``python main.py`` launches the WeCom long-connection bot.

Configuration lives at ``~/.chat_team/`` (override with ``CHAT_TEAM_HOME``).
"""
from __future__ import annotations

from chat_team.app import run

if __name__ == "__main__":
    run()
