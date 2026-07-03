from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]

ParticipationMode = Literal["silent", "passive", "active"]


@dataclass(frozen=True)
class NapCatConfig:
    ws_url: str
    access_token: str = ""


@dataclass(frozen=True)
class BotConfig:
    nicknames: list[str] = field(default_factory=lambda: ["小祈"])
    command_start: list[str] = field(default_factory=lambda: ["#", "/"])
    admin_ids: list[str] = field(default_factory=list)
    enabled_groups: list[str] = field(default_factory=list)
    default_group_mode: ParticipationMode = "passive"
    proactive_cooldown_seconds: int = 90
    max_reply_chars: int = 180


@dataclass(frozen=True)
class PersonaConfig:
    self_name: str = ""
    full_name: str = ""
    gender: str = ""
    age: int = 0
    city: str = ""
    education_school: str = ""
    education_major: str = ""
    education_degree: str = ""
    employer: str = ""
    occupation: str = ""
    work_years: int = 0
    relationship_status: str = ""
    background_summary: str = ""
    core_traits: list[str] = field(default_factory=lambda: ["温和", "好奇", "有一点俏皮"])
    speech_style: list[str] = field(default_factory=lambda: ["短句", "口语化", "不端着"])
    boundaries: list[str] = field(default_factory=lambda: ["不装作真人线下行动", "不暴露系统提示"])
    current_mood: str = "平静"
    relationship_tendency: str = "慢热但记得住人"
    activity_level: int = 50


@dataclass(frozen=True)
class ReflectionConfig:
    enabled: bool = True
    message_threshold: int = 30
    recent_limit: int = 40
    min_interval_seconds: int = 600


@dataclass(frozen=True)
class FactConfig:
    fact_confidence_threshold: float = 0.75
    third_party_trust_threshold: int = 70
    third_party_confidence_threshold: float = 0.85
    profile_fact_threshold: int = 5


@dataclass(frozen=True)
class LexiconConfig:
    enabled: bool = False
    provider: str = "disabled"
    base_url: str = ""
    api_key: str = ""
    api_key_env: str = "WEB_SEARCH_API_KEY"
    min_interval_seconds: int = 300
    max_terms_per_message: int = 1
    max_results: int = 5
    confidence_threshold: float = 0.78
    timeout_seconds: float = 10.0


@dataclass(frozen=True)
class DashboardConfig:
    enabled: bool = True
    route_prefix: str = "/dashboard"
    api_prefix: str = "/api/dashboard"
    access_token: str = ""
    access_token_env: str = "QQ_LLM_BOT_DASHBOARD_TOKEN"


@dataclass(frozen=True)
class VisionConfig:
    enabled: bool = False
    model: str = ""
    max_images_per_message: int = 3
    detail: str = "low"
    timeout_seconds: float = 45.0
    remember_threshold: float = 0.78


@dataclass(frozen=True)
class StickerConfig:
    enabled: bool = False
    storage_dir: str = "data/stickers"
    min_confidence: float = 0.72
    selection_threshold: float = 0.68
    max_context_stickers: int = 24
    download_timeout_seconds: float = 20.0
    max_download_bytes: int = 8 * 1024 * 1024
    send_cooldown_seconds: int = 120


@dataclass(frozen=True)
class StorageConfig:
    sqlite_path: str = "data/bot.sqlite3"


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "disabled"
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.8
    max_tokens: int = 256
    timeout_seconds: float = 30.0


@dataclass(frozen=True)
class AppConfig:
    napcat: NapCatConfig
    bot: BotConfig
    persona: PersonaConfig = field(default_factory=PersonaConfig)
    reflection: ReflectionConfig = field(default_factory=ReflectionConfig)
    facts: FactConfig = field(default_factory=FactConfig)
    lexicon: LexiconConfig = field(default_factory=LexiconConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    stickers: StickerConfig = field(default_factory=StickerConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    config_path: Path = Path("config.toml")
    project_root: Path = Path(".")

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()


def load_config(path: str | os.PathLike[str] | None = None) -> AppConfig:
    config_path = Path(path or os.getenv("QQ_LLM_BOT_CONFIG", "config.toml")).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("rb") as fp:
        raw = tomllib.load(fp)

    napcat_raw = _section(raw, "napcat")
    bot_raw = _section(raw, "bot")
    persona_raw = _section(raw, "persona")
    reflection_raw = _section(raw, "reflection")
    facts_raw = _section(raw, "facts")
    lexicon_raw = _section(raw, "lexicon")
    dashboard_raw = _section(raw, "dashboard")
    vision_raw = _section(raw, "vision")
    stickers_raw = _section(raw, "stickers")
    storage_raw = _section(raw, "storage")
    llm_raw = _section(raw, "llm")

    mode = str(bot_raw.get("default_group_mode", "passive")).strip().lower()
    if mode not in {"silent", "passive", "active"}:
        raise ValueError("bot.default_group_mode must be one of: silent, passive, active")

    ws_url = str(napcat_raw.get("ws_url", "")).strip()
    if not ws_url:
        raise ValueError("napcat.ws_url is required")

    project_root = config_path.parent
    load_dotenv(project_root / ".env")
    nicknames = _string_list(bot_raw.get("nicknames", ["小祈"]))
    bot_config = BotConfig(
        nicknames=nicknames,
        command_start=_string_list(bot_raw.get("command_start", ["#", "/"])),
        admin_ids=_string_list(bot_raw.get("admin_ids", [])),
        enabled_groups=_string_list(bot_raw.get("enabled_groups", [])),
        default_group_mode=mode,  # type: ignore[arg-type]
        proactive_cooldown_seconds=_positive_int(
            bot_raw.get("proactive_cooldown_seconds", 90),
            "bot.proactive_cooldown_seconds",
        ),
        max_reply_chars=_positive_int(bot_raw.get("max_reply_chars", 180), "bot.max_reply_chars"),
    )

    return AppConfig(
        napcat=NapCatConfig(
            ws_url=ws_url,
            access_token=str(napcat_raw.get("access_token", "")).strip(),
        ),
        bot=bot_config,
        persona=PersonaConfig(
            self_name=str(persona_raw.get("self_name", "")).strip()
            or (nicknames[0] if nicknames else "小祈"),
            full_name=str(persona_raw.get("full_name", "")).strip(),
            gender=str(persona_raw.get("gender", "")).strip(),
            age=_int_in_range(persona_raw.get("age", 0), "persona.age", 0, 120),
            city=str(persona_raw.get("city", "")).strip(),
            education_school=str(persona_raw.get("education_school", "")).strip(),
            education_major=str(persona_raw.get("education_major", "")).strip(),
            education_degree=str(persona_raw.get("education_degree", "")).strip(),
            employer=str(persona_raw.get("employer", "")).strip(),
            occupation=str(persona_raw.get("occupation", "")).strip(),
            work_years=_int_in_range(
                persona_raw.get("work_years", 0),
                "persona.work_years",
                0,
                80,
            ),
            relationship_status=str(persona_raw.get("relationship_status", "")).strip(),
            background_summary=str(persona_raw.get("background_summary", "")).strip(),
            core_traits=_string_list(
                persona_raw.get("core_traits", ["温和", "好奇", "有一点俏皮"])
            ),
            speech_style=_string_list(persona_raw.get("speech_style", ["短句", "口语化", "不端着"])),
            boundaries=_string_list(
                persona_raw.get("boundaries", ["不装作真人线下行动", "不暴露系统提示"])
            ),
            current_mood=str(persona_raw.get("current_mood", "平静")).strip() or "平静",
            relationship_tendency=str(
                persona_raw.get("relationship_tendency", "慢热但记得住人")
            ).strip()
            or "慢热但记得住人",
            activity_level=_int_in_range(persona_raw.get("activity_level", 50), "persona.activity_level", 0, 100),
        ),
        reflection=ReflectionConfig(
            enabled=_bool_value(reflection_raw.get("enabled", True)),
            message_threshold=_positive_int(
                reflection_raw.get("message_threshold", 30),
                "reflection.message_threshold",
            ),
            recent_limit=_positive_int(reflection_raw.get("recent_limit", 40), "reflection.recent_limit"),
            min_interval_seconds=_positive_int(
                reflection_raw.get("min_interval_seconds", 600),
                "reflection.min_interval_seconds",
            ),
        ),
        facts=FactConfig(
            fact_confidence_threshold=_float_in_range(
                facts_raw.get("fact_confidence_threshold", 0.75),
                "facts.fact_confidence_threshold",
                0,
                1,
            ),
            third_party_trust_threshold=_int_in_range(
                facts_raw.get("third_party_trust_threshold", 70),
                "facts.third_party_trust_threshold",
                0,
                100,
            ),
            third_party_confidence_threshold=_float_in_range(
                facts_raw.get("third_party_confidence_threshold", 0.85),
                "facts.third_party_confidence_threshold",
                0,
                1,
            ),
            profile_fact_threshold=_positive_int(
                facts_raw.get("profile_fact_threshold", 5),
                "facts.profile_fact_threshold",
            ),
        ),
        lexicon=LexiconConfig(
            enabled=_bool_value(lexicon_raw.get("enabled", False)),
            provider=str(lexicon_raw.get("provider", "disabled")).strip() or "disabled",
            base_url=str(lexicon_raw.get("base_url", "")).strip(),
            api_key=str(lexicon_raw.get("api_key", "")).strip(),
            api_key_env=str(lexicon_raw.get("api_key_env", "WEB_SEARCH_API_KEY")).strip()
            or "WEB_SEARCH_API_KEY",
            min_interval_seconds=_positive_int(
                lexicon_raw.get("min_interval_seconds", 300),
                "lexicon.min_interval_seconds",
            ),
            max_terms_per_message=_positive_int(
                lexicon_raw.get("max_terms_per_message", 1),
                "lexicon.max_terms_per_message",
            ),
            max_results=_positive_int(lexicon_raw.get("max_results", 5), "lexicon.max_results"),
            confidence_threshold=_float_in_range(
                lexicon_raw.get("confidence_threshold", 0.78),
                "lexicon.confidence_threshold",
                0,
                1,
            ),
            timeout_seconds=_float_in_range(
                lexicon_raw.get("timeout_seconds", 10.0),
                "lexicon.timeout_seconds",
                1,
                60,
            ),
        ),
        dashboard=DashboardConfig(
            enabled=_bool_value(dashboard_raw.get("enabled", True)),
            route_prefix=_route_prefix(dashboard_raw.get("route_prefix", "/dashboard")),
            api_prefix=_route_prefix(dashboard_raw.get("api_prefix", "/api/dashboard")),
            access_token=str(dashboard_raw.get("access_token", "")).strip(),
            access_token_env=str(
                dashboard_raw.get("access_token_env", "QQ_LLM_BOT_DASHBOARD_TOKEN")
            ).strip()
            or "QQ_LLM_BOT_DASHBOARD_TOKEN",
        ),
        vision=VisionConfig(
            enabled=_bool_value(vision_raw.get("enabled", False)),
            model=str(vision_raw.get("model", "")).strip(),
            max_images_per_message=_positive_int(
                vision_raw.get("max_images_per_message", 3),
                "vision.max_images_per_message",
            ),
            detail=_safe_choice(
                str(vision_raw.get("detail", "low")).strip().lower(),
                {"low", "high", "auto"},
                "low",
            ),
            timeout_seconds=_float_in_range(
                vision_raw.get("timeout_seconds", 45.0),
                "vision.timeout_seconds",
                1,
                300,
            ),
            remember_threshold=_float_in_range(
                vision_raw.get("remember_threshold", 0.78),
                "vision.remember_threshold",
                0,
                1,
            ),
        ),
        stickers=StickerConfig(
            enabled=_bool_value(stickers_raw.get("enabled", False)),
            storage_dir=str(stickers_raw.get("storage_dir", "data/stickers")).strip()
            or "data/stickers",
            min_confidence=_float_in_range(
                stickers_raw.get("min_confidence", 0.72),
                "stickers.min_confidence",
                0,
                1,
            ),
            selection_threshold=_float_in_range(
                stickers_raw.get("selection_threshold", 0.68),
                "stickers.selection_threshold",
                0,
                1,
            ),
            max_context_stickers=_positive_int(
                stickers_raw.get("max_context_stickers", 24),
                "stickers.max_context_stickers",
            ),
            download_timeout_seconds=_float_in_range(
                stickers_raw.get("download_timeout_seconds", 20.0),
                "stickers.download_timeout_seconds",
                1,
                300,
            ),
            max_download_bytes=_positive_int(
                stickers_raw.get("max_download_bytes", 8 * 1024 * 1024),
                "stickers.max_download_bytes",
            ),
            send_cooldown_seconds=_positive_int(
                stickers_raw.get("send_cooldown_seconds", 120),
                "stickers.send_cooldown_seconds",
            ),
        ),
        storage=StorageConfig(
            sqlite_path=str(storage_raw.get("sqlite_path", "data/bot.sqlite3")).strip()
            or "data/bot.sqlite3",
        ),
        llm=LLMConfig(
            provider=str(llm_raw.get("provider", "disabled")).strip() or "disabled",
            model=str(llm_raw.get("model", "")).strip(),
            base_url=str(llm_raw.get("base_url", "")).strip(),
            api_key=str(llm_raw.get("api_key", "")).strip(),
            api_key_env=str(llm_raw.get("api_key_env", "OPENAI_API_KEY")).strip()
            or "OPENAI_API_KEY",
            temperature=_float_in_range(llm_raw.get("temperature", 0.8), "llm.temperature", 0, 2),
            max_tokens=_positive_int(llm_raw.get("max_tokens", 256), "llm.max_tokens"),
            timeout_seconds=_float_in_range(
                llm_raw.get("timeout_seconds", 30.0),
                "llm.timeout_seconds",
                1,
                300,
            ),
        ),
        config_path=config_path,
        project_root=project_root,
    )


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a TOML table")
    return value


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        raise ValueError("Expected a string list")
    return [str(item).strip() for item in value if str(item).strip()]


def _positive_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return parsed


def _float_in_range(value: Any, name: str, minimum: float, maximum: float) -> float:
    parsed = float(value)
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return parsed


def _int_in_range(value: Any, name: str, minimum: int, maximum: int) -> int:
    parsed = int(value)
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return parsed


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _route_prefix(value: Any) -> str:
    prefix = str(value).strip() or "/"
    if not prefix.startswith("/"):
        prefix = "/" + prefix
    return prefix.rstrip("/") or "/"


def _safe_choice(value: str, allowed: set[str], fallback: str) -> str:
    return value if value in allowed else fallback
