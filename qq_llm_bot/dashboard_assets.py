from __future__ import annotations

from qq_llm_bot.dashboard_markup import (
    DASHBOARD_HTML_AFTER_STYLE,
    DASHBOARD_HTML_PREFIX,
    DASHBOARD_HTML_SUFFIX,
)
from qq_llm_bot.dashboard_scripts import DASHBOARD_JS
from qq_llm_bot.dashboard_styles import DASHBOARD_CSS

DASHBOARD_HTML = (
    DASHBOARD_HTML_PREFIX
    + DASHBOARD_CSS
    + DASHBOARD_HTML_AFTER_STYLE
    + DASHBOARD_JS
    + DASHBOARD_HTML_SUFFIX
)