from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


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
    ignored_user_ids: list[str] = field(default_factory=list)
    enabled_groups: list[str] = field(default_factory=list)
    default_group_mode: ParticipationMode = "passive"
    proactive_cooldown_seconds: int = 90
    proactive_value_threshold: float = 0.65
    proactive_busy_value_threshold: float = 0.78
    proactive_busy_human_messages: int = 6
    interaction_followup_seconds: int = 180
    max_reply_chars: int = 180
    reply_style_enabled: bool = True
    reply_bubbles_enabled: bool = True
    reply_bubble_trigger_chars: int = 45
    reply_bubble_target_chars: int = 28
    reply_bubble_max_parts: int = 3
    reply_bubble_delay_seconds: float = 0.9
    reply_emoji_cooldown_messages: int = 10
    realtime_merge_enabled: bool = True
    realtime_merge_grace_seconds: float = 3.0
    realtime_merge_max_messages: int = 5
    realtime_merge_max_window_seconds: float = 12.0
    send_retry_enabled: bool = True
    send_retry_max_attempts: int = 6
    send_retry_max_age_seconds: int = 180
    send_retry_queue_limit: int = 100
    send_retry_base_delay_seconds: float = 2.0
    send_retry_max_delay_seconds: float = 30.0
    final_qa_enabled: bool = True
    context_understanding_enabled: bool = True


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
    appearance_prompt: str = ""
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
class ObservationBatchConfig:
    enabled: bool = True
    batch_size: int = 30
    max_interval_seconds: int = 600
    max_messages_per_batch: int = 40
    max_message_chars: int = 300


@dataclass(frozen=True)
class FactConfig:
    fact_confidence_threshold: float = 0.75
    third_party_trust_threshold: int = 70
    third_party_confidence_threshold: float = 0.85
    profile_fact_threshold: int = 5
    context_fact_limit: int = 8
    target_user_limit: int = 5
    low_importance_threshold: float = 0.35
    fact_context_ttl_days: int = 30


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
class ImageGenerationConfig:
    enabled: bool = False
    model: str = ""
    storage_dir: str = "data/generated_images"
    size: str = "512x512"
    quality: str = "low"
    output_format: str = "jpeg"
    output_compression: int = 65
    timeout_seconds: float = 240.0
    max_send_dimension: int = 832
    min_trust: int = 5
    daily_limit: int = 5
    max_prompt_chars: int = 800
    max_reference_images: int = 3
    reference_image_max_bytes: int = 4 * 1024 * 1024
    reference_image_max_dimension: int = 1536
    reference_image_quality: int = 85


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
    unused_ttl_hours: int = 72
    cleanup_interval_hours: int = 24


@dataclass(frozen=True)
class StorageConfig:
    sqlite_path: str = "data/bot.sqlite3"


@dataclass(frozen=True)
class LLMRoutingConfig:
    enabled: bool = False
    base_model: str = ""
    flagship_model: str = ""
    vision_base_model: str = ""


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "disabled"
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.8
    max_tokens: int = 4096
    timeout_seconds: float = 30.0
    routing: LLMRoutingConfig = field(default_factory=LLMRoutingConfig)


@dataclass(frozen=True)
class AppConfig:
    napcat: NapCatConfig
    bot: BotConfig
    persona: PersonaConfig = field(default_factory=PersonaConfig)
    reflection: ReflectionConfig = field(default_factory=ReflectionConfig)
    observation_batch: ObservationBatchConfig = field(default_factory=ObservationBatchConfig)
    facts: FactConfig = field(default_factory=FactConfig)
    lexicon: LexiconConfig = field(default_factory=LexiconConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    image_generation: ImageGenerationConfig = field(default_factory=ImageGenerationConfig)
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
