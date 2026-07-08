from __future__ import annotations

import ast
import asyncio
import base64
import io
import json
import unittest
from dataclasses import replace
from pathlib import Path

from qq_llm_bot.cognitive_agents import _complete_json, _complete_vision_json
from qq_llm_bot.config import (
    ImageGenerationConfig,
    LLMConfig,
    LLMRoutingConfig,
    VisionConfig,
    load_config,
)
from qq_llm_bot.draw_images import prepare_draw_reference_images
from qq_llm_bot.draw_reference import DrawIntentPlanner
from qq_llm_bot.image_generation import GeneratedImageStore
from qq_llm_bot.llm import (
    GeneratedImage,
    OpenAICompatibleLLMClient,
    _image_generation_input,
    extract_generated_image,
    normalize_responses_url,
)
from qq_llm_bot.models import MessageAttachment, MessageContext
from qq_llm_bot.web_search import default_slang_query
from tests.helpers import FakeLLM, RetryCapableFakeLLM, project_temp_directory, test_config

class WebSearchTests(unittest.TestCase):
    def test_default_slang_query_uses_readable_chinese_terms(self) -> None:
        self.assertEqual(default_slang_query("内卷"), "内卷 网络用语 梗 意思")


class DrawIntentTests(unittest.TestCase):
    def test_intent_planner_keeps_references_without_searching(self) -> None:
        llm = FakeLLM(
            [
                (
                    '{"cleaned_draw_request":"参考异环里的真红和伊洛伊，画一个融合他俩特征的二创角色。",'
                    '"bot_mention_role":"addressing_bot","include_bot_appearance":false,'
                    '"reference_notes":"参考异环里的真红和伊洛伊，融合两者特征"}'
                ),
            ]
        )
        planner = DrawIntentPlanner(llm, bot_names=("可可",))

        plan = asyncio.run(
            planner.plan("可可，参考异环里的真红和伊洛伊，画一个融合他俩特征的二创角色。")
        )

        self.assertEqual(plan.cleaned_draw_request, "参考异环里的真红和伊洛伊，画一个融合他俩特征的二创角色。")
        self.assertEqual(plan.bot_mention_role, "addressing_bot")
        self.assertFalse(plan.include_bot_appearance)
        self.assertIn("真红", plan.reference_notes)
        self.assertEqual(llm.text_call_purposes, ["draw_intent"])

    def test_intent_planner_includes_bot_appearance_only_when_bot_is_subject(self) -> None:
        llm = FakeLLM(
            [
                (
                    '{"cleaned_draw_request":"画可可的拟人头像",'
                    '"bot_mention_role":"draw_bot","include_bot_appearance":true,'
                    '"reference_notes":""}'
                )
            ]
        )
        planner = DrawIntentPlanner(llm, bot_names=("可可",))

        plan = asyncio.run(planner.plan("画可可的拟人头像"))

        self.assertEqual(plan.cleaned_draw_request, "画可可的拟人头像")
        self.assertEqual(plan.bot_mention_role, "draw_bot")
        self.assertTrue(plan.include_bot_appearance)


class ImageGenerationTests(unittest.TestCase):
    def test_responses_url_is_derived_from_openai_compatible_base_url(self) -> None:
        self.assertEqual(
            normalize_responses_url("https://example.test/v1"),
            "https://example.test/v1/responses",
        )
        self.assertEqual(
            normalize_responses_url("https://example.test"),
            "https://example.test/v1/responses",
        )

    def test_extract_generated_image_reads_responses_image_generation_call(self) -> None:
        encoded = base64.b64encode(b"fake-png").decode("ascii")

        image = extract_generated_image(
            {"output": [{"type": "image_generation_call", "result": encoded}]}
        )

        self.assertIsNotNone(image)
        self.assertEqual(image.data, b"fake-png")  # type: ignore[union-attr]
        self.assertEqual(image.mime_type, "image/png")  # type: ignore[union-attr]

    def test_generated_image_store_saves_single_image_file(self) -> None:
        with project_temp_directory() as tmp:
            base_config = test_config(Path(tmp) / "bot.sqlite3")
            config = replace(
                base_config,
                image_generation=ImageGenerationConfig(
                    enabled=True,
                    storage_dir="generated",
                ),
            )
            context = MessageContext(
                group_id="100",
                user_id="42",
                message_id="m-draw",
                plain_text="#draw cat",
                raw_message="#draw cat",
            )

            saved = GeneratedImageStore(config).save(
                context,
                GeneratedImage(data=b"fake-png", mime_type="image/png"),
            )

            self.assertIsNotNone(saved)
            self.assertTrue(Path(saved.local_path).exists())  # type: ignore[union-attr]
            self.assertEqual(saved.file_ref, saved.local_path)  # type: ignore[union-attr]

    def test_generated_image_store_compresses_large_images_for_chat(self) -> None:
        try:
            from PIL import Image
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest("Pillow is not installed in this environment") from exc

        with project_temp_directory() as tmp:
            source = io.BytesIO()
            Image.new("RGB", (1400, 1000), "white").save(source, format="PNG")
            base_config = test_config(Path(tmp) / "bot.sqlite3")
            config = replace(
                base_config,
                image_generation=ImageGenerationConfig(
                    enabled=True,
                    storage_dir="generated",
                    output_format="jpeg",
                    output_compression=65,
                    max_send_dimension=512,
                ),
            )
            context = MessageContext(
                group_id="100",
                user_id="42",
                message_id="m-draw",
                plain_text="#draw cat",
                raw_message="#draw cat",
            )

            saved = GeneratedImageStore(config).save(
                context,
                GeneratedImage(data=source.getvalue(), mime_type="image/png"),
            )

            self.assertIsNotNone(saved)
            self.assertEqual(saved.mime_type, "image/jpeg")  # type: ignore[union-attr]
            with Image.open(saved.local_path) as image:  # type: ignore[arg-type, union-attr]
                self.assertLessEqual(max(image.size), 512)
                self.assertEqual(image.format, "JPEG")

    def test_image_generation_config_uses_small_chat_defaults(self) -> None:
        with project_temp_directory() as tmp:
            config_path = Path(tmp) / "config.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[napcat]",
                        'ws_url = "ws://example.test"',
                        "",
                        "[image_generation]",
                        'size = "1024x1024"',
                        'quality = "auto"',
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertEqual(config.image_generation.size, "832x832")
            self.assertEqual(config.image_generation.output_format, "jpeg")
            self.assertEqual(config.image_generation.output_compression, 65)
            self.assertEqual(config.image_generation.timeout_seconds, 240.0)
            self.assertEqual(config.image_generation.max_send_dimension, 832)

    def test_image_generation_retries_once_when_response_has_no_image(self) -> None:
        client = OpenAICompatibleLLMClient(
            LLMConfig(
                provider="openai-compatible",
                model="gpt-image-test",
                base_url="https://example.test/v1",
                api_key="test-key",
            )
        )
        calls = 0
        captured_payload: dict = {}

        async def fake_post_image_generation_response(
            payload: dict,
            timeout_seconds: float,
        ) -> GeneratedImage | None:
            nonlocal calls
            calls += 1
            captured_payload.update(payload)
            if calls == 1:
                client._last_image_generation_failure_kind = "no_image"  # type: ignore[attr-defined]
                client.last_image_generation_error = "output=[message/result=empty]"
                return None
            return GeneratedImage(data=b"fake-png", mime_type="image/png")

        client._post_image_generation_response = fake_post_image_generation_response  # type: ignore[method-assign]

        image = asyncio.run(
            client.generate_image("cat", ImageGenerationConfig(model="gpt-image-test"))
        )

        self.assertIsNotNone(image)
        self.assertEqual(image.data, b"fake-png")  # type: ignore[union-attr]
        self.assertEqual(calls, 2)
        self.assertEqual(captured_payload["model"], "gpt-image-test")
        tool = captured_payload["tools"][0]
        self.assertEqual(tool["size"], "512x512")
        self.assertEqual(tool["quality"], "low")
        self.assertEqual(tool["output_format"], "jpeg")
        self.assertEqual(tool["output_compression"], 65)

    def test_image_generation_input_includes_reference_images(self) -> None:
        payload = _image_generation_input(
            "画一只猫",
            ["", "data:image/png;base64,ZmFrZQ==", "https://example.test/ref.png"],
        )

        self.assertIsInstance(payload, list)
        message = payload[0]  # type: ignore[index]
        content = message["content"]  # type: ignore[index]
        self.assertEqual(content[0], {"type": "input_text", "text": "画一只猫"})
        self.assertEqual(
            content[1],
            {"type": "input_image", "image_url": "data:image/png;base64,ZmFrZQ=="},
        )
        self.assertEqual(
            content[2],
            {"type": "input_image", "image_url": "https://example.test/ref.png"},
        )

    def test_image_generation_request_passes_reference_images_to_responses_api(self) -> None:
        client = OpenAICompatibleLLMClient(
            LLMConfig(
                provider="openai-compatible",
                model="gpt-image-test",
                base_url="https://example.test/v1",
                api_key="test-key",
            )
        )
        captured_payload: dict = {}

        async def fake_post_image_generation_response(
            payload: dict,
            timeout_seconds: float,
        ) -> GeneratedImage | None:
            captured_payload.update(payload)
            return GeneratedImage(data=b"fake-png", mime_type="image/png")

        client._post_image_generation_response = fake_post_image_generation_response  # type: ignore[method-assign]

        image = asyncio.run(
            client.generate_image(
                "cat",
                ImageGenerationConfig(model="gpt-image-test"),
                image_urls=["data:image/png;base64,ZmFrZQ=="],
            )
        )

        self.assertIsNotNone(image)
        content = captured_payload["input"][0]["content"]
        self.assertEqual(content[0], {"type": "input_text", "text": "cat"})
        self.assertEqual(
            content[1],
            {"type": "input_image", "image_url": "data:image/png;base64,ZmFrZQ=="},
        )

    def test_prepare_draw_reference_images_rejects_more_than_three_images(self) -> None:
        attachments = [
            MessageAttachment(attachment_type="image", url=f"https://example.test/{index}.png")
            for index in range(4)
        ]

        prepared = asyncio.run(
            prepare_draw_reference_images(
                attachments,
                max_images=3,
                max_bytes=1024 * 1024,
                max_dimension=512,
                quality=85,
                timeout_seconds=1.0,
            )
        )

        self.assertEqual(prepared.image_urls, [])
        self.assertIn("参考图最多支持 3 张", prepared.error)

    def test_prepare_draw_reference_images_keeps_small_data_url(self) -> None:
        try:
            from PIL import Image
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest("Pillow is not installed in this environment") from exc

        source = io.BytesIO()
        Image.new("RGB", (32, 32), "white").save(source, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(source.getvalue()).decode("ascii")

        prepared = asyncio.run(
            prepare_draw_reference_images(
                [MessageAttachment(attachment_type="image", url=data_url)],
                max_images=3,
                max_bytes=1024 * 1024,
                max_dimension=512,
                quality=85,
                timeout_seconds=1.0,
            )
        )

        self.assertEqual(prepared.error, "")
        self.assertEqual(prepared.image_urls, [data_url])

    def test_prepare_draw_reference_images_compresses_large_reference(self) -> None:
        try:
            from PIL import Image
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest("Pillow is not installed in this environment") from exc

        source = io.BytesIO()
        Image.new("RGB", (1024, 768), "purple").save(source, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(source.getvalue()).decode("ascii")

        prepared = asyncio.run(
            prepare_draw_reference_images(
                [MessageAttachment(attachment_type="image", url=data_url)],
                max_images=3,
                max_bytes=128 * 1024,
                max_dimension=128,
                quality=85,
                timeout_seconds=1.0,
            )
        )

        self.assertEqual(prepared.error, "")
        self.assertEqual(len(prepared.image_urls), 1)
        compressed_url = prepared.image_urls[0]
        self.assertTrue(compressed_url.startswith("data:image/jpeg;base64,"))
        compressed = base64.b64decode(compressed_url.split(",", 1)[1])
        self.assertLessEqual(len(compressed), 128 * 1024)
        with Image.open(io.BytesIO(compressed)) as image:
            self.assertLessEqual(max(image.size), 128)
            self.assertEqual(image.format, "JPEG")

    def test_processing_ack_uses_napcat_emoji_like_action(self) -> None:
        plugin_path = (
            Path(__file__).resolve().parents[1] / "plugins" / "llm_group_bot" / "draw_command.py"
        )
        source = plugin_path.read_text(encoding="utf-8")

        self.assertIn('_PROCESSING_ACK_EMOJI_ID = "124"', source)
        self.assertIn('"set_msg_emoji_like"', source)
        self.assertIn("await bot.call_api(", source)

    def test_draw_command_acknowledges_before_slow_processing(self) -> None:
        plugin_path = (
            Path(__file__).resolve().parents[1] / "plugins" / "llm_group_bot" / "draw_command.py"
        )
        source = plugin_path.read_text(encoding="utf-8")

        ack_position = source.index('await _acknowledge_processing(bot, event.message_id, "draw")')
        prompt_position = source.index("image_prompt = await _compose_draw_prompt")

        self.assertLess(ack_position, prompt_position)

    def test_processing_ack_is_limited_to_draw_command(self) -> None:
        plugin_path = (
            Path(__file__).resolve().parents[1] / "plugins" / "llm_group_bot" / "draw_command.py"
        )
        source = plugin_path.read_text(encoding="utf-8")

        self.assertEqual(source.count("await _acknowledge_processing("), 1)
        self.assertIn('await _acknowledge_processing(bot, event.message_id, "draw")', source)
        self.assertNotIn('"realtime pipeline"', source)
        self.assertNotIn('"llm test"', source)

    def test_draw_command_exempts_admins_from_trust_and_daily_limit(self) -> None:
        plugin_path = (
            Path(__file__).resolve().parents[1] / "plugins" / "llm_group_bot" / "draw_command.py"
        )
        source = plugin_path.read_text(encoding="utf-8")

        self.assertIn("is_admin = storage.is_admin(user_id)", source)
        self.assertIn(
            "if not is_admin and relation.trust < config.image_generation.min_trust:",
            source,
        )
        self.assertIn(
            "if not is_admin and used_count >= config.image_generation.daily_limit:",
            source,
        )

    def test_draw_command_retries_image_send_with_base64_fallback(self) -> None:
        sender_path = (
            Path(__file__).resolve().parents[1]
            / "plugins"
            / "llm_group_bot"
            / "generated_images.py"
        )
        source = sender_path.read_text(encoding="utf-8")

        self.assertIn("base64_ref = _generated_image_base64_ref(saved.local_path)", source)
        self.assertIn("except ActionFailed as exc:", source)
        self.assertIn("except Exception as exc:", source)
        self.assertIn("outbound_queue.queue_group_attempts(", source)
        self.assertIn("QueuedSendAttempt(attempt_message)", source)
        self.assertIn("for include_reply in (True, False)", source)
        self.assertIn('return "base64://" + base64.b64encode(data).decode("ascii")', source)

    def test_send_retry_queue_is_flushed_on_bot_reconnect(self) -> None:
        root = Path(__file__).resolve().parents[1]
        plugin_source = (root / "plugins" / "llm_group_bot" / "__init__.py").read_text(
            encoding="utf-8"
        )
        queue_source = (root / "qq_llm_bot" / "outbound_queue.py").read_text(encoding="utf-8")
        config_source = (root / "qq_llm_bot" / "config_models.py").read_text(encoding="utf-8")

        self.assertIn("@driver.on_bot_connect", plugin_source)
        self.assertIn('await outbound_queue.flush(bot, "bot connected")', plugin_source)
        self.assertIn("class OutboundGroupSendQueue:", queue_source)
        self.assertIn("def should_queue_send_error(exc: BaseException) -> bool:", queue_source)
        self.assertIn("send_retry_max_attempts: int = 6", config_source)
        self.assertIn("send_retry_max_age_seconds: int = 180", config_source)

    def test_draw_command_reports_image_generation_failure_detail_to_admins(self) -> None:
        plugin_path = (
            Path(__file__).resolve().parents[1] / "plugins" / "llm_group_bot" / "draw_command.py"
        )
        source = plugin_path.read_text(encoding="utf-8")

        self.assertIn('getattr(llm, "last_image_generation_error"', source)
        self.assertIn(
            '_draw_failure_reply("Responses image_generation 没有返回图片", is_admin, detail)',
            source,
        )

    def test_draw_command_uses_persona_appearance_without_fixed_outfit_or_scene(self) -> None:
        root = Path(__file__).resolve().parents[1]
        draw_prompt_source = (root / "plugins" / "llm_group_bot" / "draw_prompts.py").read_text(
            encoding="utf-8"
        )
        config_source = (root / "qq_llm_bot" / "config_models.py").read_text(encoding="utf-8")
        storage_source = (root / "qq_llm_bot" / "storage_lifecycle.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("appearance_prompt: str = \"\"", config_source)
        self.assertIn('"appearance_prompt": config.persona.appearance_prompt', storage_source)
        self.assertIn("appearance_prompt 只约束人物样貌，不固定服装、场景", draw_prompt_source)

    def test_draw_prompt_uses_intent_plan_without_reference_search(self) -> None:
        draw_prompt_path = (
            Path(__file__).resolve().parents[1]
            / "plugins"
            / "llm_group_bot"
            / "draw_prompts.py"
        )
        source = draw_prompt_path.read_text(encoding="utf-8")

        self.assertIn("draw_intent_planner.plan(draw_request)", source)
        self.assertIn("用户显式参考要求", source)
        self.assertIn("不要联网查证", source)
        self.assertIn("_draw_bot_appearance_context", source)
        self.assertNotIn('f"人设：\\n{_draw_join(snapshot.persona_lines)}\\n"', source)

    def test_draw_command_passes_prepared_reference_images_to_image_model(self) -> None:
        plugin_path = (
            Path(__file__).resolve().parents[1] / "plugins" / "llm_group_bot" / "draw_command.py"
        )
        source = plugin_path.read_text(encoding="utf-8")

        self.assertIn("prepare_draw_reference_images(", source)
        self.assertIn("max_images=config.image_generation.max_reference_images", source)
        self.assertIn("reference_image_count=len(reference_images.image_urls)", source)
        self.assertIn("image_urls=reference_images.image_urls", source)

    def test_draw_prompt_extractor_recovers_truncated_json_string(self) -> None:
        draw_prompt_path = (
            Path(__file__).resolve().parents[1]
            / "plugins"
            / "llm_group_bot"
            / "draw_prompts.py"
        )
        source = draw_prompt_path.read_text(encoding="utf-8")
        module = ast.parse(source)
        selected = [
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"_extract_draw_prompt", "_extract_truncated_draw_prompt"}
        ]
        test_module = ast.Module(body=selected, type_ignores=[])
        ast.fix_missing_locations(test_module)
        namespace = {"json": json, "re": __import__("re")}
        exec(compile(test_module, str(draw_prompt_path), "exec"), namespace)

        extract_draw_prompt = namespace["_extract_draw_prompt"]

        self.assertEqual(
            extract_draw_prompt('{"prompt":"白发角色，融合红黑服饰和金色装饰'),
            "白发角色，融合红黑服饰和金色装饰",
        )


class LLMRoutingTests(unittest.TestCase):
    def test_llm_routing_config_is_loaded_from_nested_table(self) -> None:
        with project_temp_directory() as tmp:
            config_path = Path(tmp) / "config.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[napcat]",
                        'ws_url = "ws://example.test"',
                        "",
                        "[llm]",
                        'provider = "openai-compatible"',
                        'model = "gpt-5.5"',
                        'base_url = "https://example.test/v1"',
                        "",
                        "[llm.routing]",
                        "enabled = true",
                        'base_model = "gpt-5.4-mini"',
                        'flagship_model = "gpt-5.5"',
                        'vision_base_model = "gpt-5.4-mini"',
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertTrue(config.llm.routing.enabled)
            self.assertEqual(config.llm.routing.base_model, "gpt-5.4-mini")
            self.assertEqual(config.llm.routing.flagship_model, "gpt-5.5")
            self.assertEqual(config.llm.routing.vision_base_model, "gpt-5.4-mini")

    def test_openai_client_routes_text_purposes_to_expected_models(self) -> None:
        client = OpenAICompatibleLLMClient(
            LLMConfig(
                provider="openai-compatible",
                model="gpt-5.5",
                base_url="https://example.test/v1",
                api_key="test-key",
                routing=LLMRoutingConfig(
                    enabled=True,
                    base_model="gpt-5.4-mini",
                    flagship_model="gpt-5.5",
                    vision_base_model="gpt-5.4-mini",
                ),
            )
        )
        captured: list[tuple[str, str, int]] = []

        async def fake_post_chat_completion(
            payload: dict,
            timeout_seconds: float,
            purpose: str,
            prompt_chars: int,
        ) -> str | None:
            captured.append((purpose, str(payload["model"]), int(payload["max_tokens"])))
            return "ok"

        client._post_chat_completion = fake_post_chat_completion  # type: ignore[method-assign]

        asyncio.run(client.complete_text("s", "u", purpose="perception"))
        asyncio.run(client.complete_text("s", "u", purpose="draw_intent"))
        asyncio.run(client.complete_text("s", "u", purpose="draw_prompt"))
        asyncio.run(client.complete_text("s", "u", purpose="response"))
        asyncio.run(client.complete_text("s", "u", purpose="final_qa"))
        asyncio.run(client.complete_text("s", "u", purpose="final_qa_repair"))
        asyncio.run(client.complete_text("s", "u", purpose="final_qa", model_tier="flagship"))

        self.assertEqual(
            captured,
            [
                ("perception", "gpt-5.4-mini", 4096),
                ("draw_intent", "gpt-5.4-mini", 4096),
                ("draw_prompt", "gpt-5.4-mini", 4096),
                ("response", "gpt-5.5", 4096),
                ("final_qa", "gpt-5.4-mini", 4096),
                ("final_qa_repair", "gpt-5.4-mini", 4096),
                ("final_qa", "gpt-5.5", 4096),
            ],
        )

    def test_openai_client_routes_vision_to_base_and_flagship_models(self) -> None:
        client = OpenAICompatibleLLMClient(
            LLMConfig(
                provider="openai-compatible",
                model="gpt-5.5",
                base_url="https://example.test/v1",
                api_key="test-key",
                routing=LLMRoutingConfig(
                    enabled=True,
                    base_model="gpt-5.4-mini",
                    flagship_model="gpt-5.5",
                    vision_base_model="gpt-5.4-mini",
                ),
            )
        )
        captured: list[str] = []

        async def fake_post_chat_completion(
            payload: dict,
            timeout_seconds: float,
            purpose: str,
            prompt_chars: int,
        ) -> str | None:
            captured.append(str(payload["model"]))
            return "{}"

        client._post_chat_completion = fake_post_chat_completion  # type: ignore[method-assign]

        asyncio.run(
            client.complete_vision(
                "s",
                "u",
                ["https://example.test/a.png"],
                VisionConfig(model="gpt-5.5"),
            )
        )
        asyncio.run(
            client.complete_vision(
                "s",
                "u",
                ["https://example.test/a.png"],
                VisionConfig(model="gpt-5.5"),
                model_tier="flagship",
            )
        )

        self.assertEqual(captured, ["gpt-5.4-mini", "gpt-5.5"])

    def test_structured_json_retries_once_with_flagship_on_parse_failure(self) -> None:
        llm = RetryCapableFakeLLM(["not json", '{"facts":[]}'])

        data = asyncio.run(_complete_json(llm, "s", "u", purpose="fact_extract"))

        self.assertEqual(data, {"facts": []})
        self.assertEqual(llm.text_call_tiers, ["", "flagship"])

    def test_vision_json_retries_once_with_flagship_on_empty_direct_result(self) -> None:
        llm = RetryCapableFakeLLM(
            vision_replies=[
                '{"images":[{"description":"","ocr_text":"","confidence":0.5}]}',
                '{"images":[{"description":"chart","ocr_text":"","confidence":0.9}]}',
            ]
        )

        data = asyncio.run(
            _complete_vision_json(
                llm,
                test_config(Path("unused.sqlite3")),
                "s",
                "u",
                ["https://example.test/a.png"],
                direct_image_hint=True,
            )
        )

        self.assertEqual(data["images"][0]["description"], "chart")  # type: ignore[index]
        self.assertEqual(llm.vision_call_tiers, ["", "flagship"])


