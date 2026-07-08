from __future__ import annotations

import ast
import sqlite3
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from qq_llm_bot.cognitive_storage import BotStorage
from qq_llm_bot.config import (
    AppConfig,
    BotConfig,
    ImageGenerationConfig,
    LLMConfig,
    NapCatConfig,
    PersonaConfig,
    ReflectionConfig,
    StorageConfig,
    VisionConfig,
)
from qq_llm_bot.llm import GeneratedImage
from qq_llm_bot.web_search import SearchResult


class FakeLLM:
    def __init__(
        self,
        replies: list[str] | None = None,
        vision_replies: list[str] | None = None,
        multimodal_replies: list[str] | None = None,
        image_replies: list[GeneratedImage | None] | None = None,
    ) -> None:
        self.replies = replies or []
        self.vision_replies = vision_replies or []
        self.multimodal_replies = multimodal_replies or []
        self.image_replies = image_replies or []
        self.text_calls: list[tuple[str, str]] = []
        self.text_call_purposes: list[str] = []
        self.text_call_tiers: list[str] = []
        self.vision_calls: list[list[str]] = []
        self.vision_call_tiers: list[str] = []
        self.multimodal_calls: list[list[str]] = []
        self.multimodal_text_calls: list[tuple[str, str]] = []
        self.multimodal_call_purposes: list[str] = []
        self.multimodal_call_tiers: list[str] = []
        self.image_calls: list[str] = []
        self.image_url_calls: list[list[str]] = []
        self.last_image_generation_error = ""

    async def complete_text(
        self,
        system_prompt: str,
        user_prompt: str,
        purpose: str = "",
        model_tier: str = "",
    ) -> str | None:
        self.text_calls.append((system_prompt, user_prompt))
        self.text_call_purposes.append(purpose)
        self.text_call_tiers.append(model_tier)
        if self.replies:
            return self.replies.pop(0)
        return None

    async def complete_vision(
        self,
        system_prompt: str,
        user_prompt: str,
        image_urls: list[str],
        vision_config: VisionConfig,
        purpose: str = "vision",
        model_tier: str = "",
    ) -> str | None:
        self.vision_calls.append(image_urls)
        self.vision_call_tiers.append(model_tier)
        if self.vision_replies:
            return self.vision_replies.pop(0)
        return None

    async def complete_multimodal(
        self,
        system_prompt: str,
        user_prompt: str,
        image_urls: list[str],
        vision_config: VisionConfig,
        purpose: str = "response",
        model_tier: str = "",
    ) -> str | None:
        self.multimodal_calls.append(list(image_urls))
        self.multimodal_text_calls.append((system_prompt, user_prompt))
        self.multimodal_call_purposes.append(purpose)
        self.multimodal_call_tiers.append(model_tier)
        if self.multimodal_replies:
            return self.multimodal_replies.pop(0)
        if self.replies:
            return self.replies.pop(0)
        return None

    async def generate_image(
        self,
        prompt: str,
        image_config: ImageGenerationConfig,
        image_urls: list[str] | None = None,
    ) -> GeneratedImage | None:
        self.image_calls.append(prompt)
        self.image_url_calls.append(list(image_urls or []))
        if self.image_replies:
            return self.image_replies.pop(0)
        return None


class RetryCapableFakeLLM(FakeLLM):
    def should_retry_with_flagship(self, purpose: str) -> bool:
        return True

    def should_retry_vision_with_flagship(self, vision_config: VisionConfig) -> bool:
        return True


class FakeSearch:
    def __init__(
        self,
        results: list[SearchResult] | None = None,
        should_raise: bool = False,
    ) -> None:
        self.results = results or []
        self.should_raise = should_raise
        self.calls: list[str] = []

    async def search(self, query: str, max_results: int | None = None) -> list[SearchResult]:
        self.calls.append(query)
        if self.should_raise:
            raise RuntimeError("search unavailable")
        return self.results[: max_results or len(self.results)]


class InMemoryBotStorage(BotStorage):
    def __init__(self) -> None:
        super().__init__(
            Path(":memory:"),
            initial_admins=[],
            initial_ignored_users=[],
            initial_groups=[],
            initial_persona={},
        )
        self.connection = sqlite3.connect(":memory:", check_same_thread=False)
        self.connection.row_factory = sqlite3.Row

    @contextmanager
    def _connect(self):
        with self._lock:
            try:
                yield self.connection
                self.connection.commit()
            except Exception:
                self.connection.rollback()
                raise


PROJECT_TMP = Path(__file__).resolve().parents[1] / ".tmp"


def test_config(db_path: Path) -> AppConfig:
    if not db_path.is_absolute():
        PROJECT_TMP.mkdir(parents=True, exist_ok=True)
        db_path = PROJECT_TMP / db_path

    return AppConfig(
        napcat=NapCatConfig(ws_url="ws://example.test"),
        bot=BotConfig(
            nicknames=["可可"],
            admin_ids=["1"],
            enabled_groups=["100"],
            default_group_mode="passive",
            proactive_cooldown_seconds=90,
            max_reply_chars=80,
        ),
        persona=PersonaConfig(self_name="可可"),
        reflection=ReflectionConfig(enabled=True, message_threshold=30),
        storage=StorageConfig(sqlite_path=str(db_path)),
        llm=LLMConfig(provider="disabled"),
        project_root=db_path.parent,
    )


test_config.__test__ = False


@contextmanager
def project_temp_directory():
    PROJECT_TMP.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=PROJECT_TMP) as tmp:
        yield tmp


class FakeDashboardDriver:
    def __init__(self, app: object) -> None:
        self.server_app = app


def dashboard_test_tools() -> tuple[object, object, object]:
    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from qq_llm_bot.dashboard import register_dashboard_routes
    except ModuleNotFoundError as exc:
        raise unittest.SkipTest("fastapi is not installed in this environment") from exc
    return FastAPI, TestClient, register_dashboard_routes


def first_attribute_call_line(node: ast.AST, attribute_name: str) -> int:
    lines = [
        child.lineno
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == attribute_name
    ]
    if not lines:
        raise AssertionError(f"Call not found: {attribute_name}")
    return min(lines)


def first_name_call_line(node: ast.AST, function_name: str) -> int:
    lines = [
        child.lineno
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id == function_name
    ]
    if not lines:
        raise AssertionError(f"Call not found: {function_name}")
    return min(lines)
