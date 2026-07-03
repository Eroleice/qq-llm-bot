from __future__ import annotations

import json
import os

from qq_llm_bot.config import AppConfig, load_config


def main() -> None:
    config = load_config()
    _configure_nonebot_env(config)

    import nonebot
    from nonebot.adapters.onebot.v11 import Adapter as OneBotV11Adapter

    nonebot.init()
    driver = nonebot.get_driver()
    driver.register_adapter(OneBotV11Adapter)

    nonebot.load_plugin("plugins.llm_group_bot")
    nonebot.run()


def _configure_nonebot_env(config: AppConfig) -> None:
    os.environ["QQ_LLM_BOT_CONFIG"] = str(config.config_path)

    os.environ.setdefault("ENVIRONMENT", "prod")
    os.environ.setdefault("LOG_LEVEL", "INFO")

    os.environ["DRIVER"] = "~fastapi+~httpx+~websockets"
    os.environ["ONEBOT_WS_URLS"] = json.dumps([config.napcat.ws_url], ensure_ascii=False)

    if config.napcat.access_token:
        os.environ["ONEBOT_ACCESS_TOKEN"] = config.napcat.access_token

    os.environ["SUPERUSERS"] = json.dumps(config.bot.admin_ids, ensure_ascii=False)
    os.environ["NICKNAME"] = json.dumps(config.bot.nicknames, ensure_ascii=False)
    os.environ["COMMAND_START"] = json.dumps(config.bot.command_start, ensure_ascii=False)
