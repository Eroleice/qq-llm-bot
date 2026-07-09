from __future__ import annotations

import base64
import unittest
from dataclasses import replace
from pathlib import Path


from qq_llm_bot.cognitive_storage import BotStorage
from qq_llm_bot.models import (
    MessageContext,
    StickerAssetRecord,
    StickerCandidate,
)
from qq_llm_bot.stickers import StickerLocalStore, sticker_file_ref, sticker_file_refs
from tests.helpers import (
    InMemoryBotStorage,
    project_temp_directory,
    test_config,
)


class StickerStorageTests(unittest.TestCase):
    def test_sticker_asset_is_saved_and_can_be_disabled(self) -> None:
        with project_temp_directory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()
            local_path = str(Path(tmp) / "meme.png")
            context = MessageContext(
                group_id="100",
                user_id="42",
                message_id="m-sticker",
                plain_text="",
                raw_message="[CQ:image]",
            )
            candidate = StickerCandidate(
                url="https://example.test/meme.png",
                file="meme.png",
                description="一张猫猫下班表情包",
                ocr_text="下班了",
                mood="疲惫",
                usage="适合聊到下班、犯困或想摆一下时使用",
                tags=("下班", "困", "猫猫"),
                confidence=0.86,
            )

            asset = storage.upsert_sticker_asset(context, candidate, local_path=local_path, sha256="abc")
            active = storage.list_sticker_assets("100")

            self.assertIsNotNone(asset)
            self.assertEqual(active[0].usage, "适合聊到下班、犯困或想摆一下时使用")
            self.assertEqual(active[0].tags, ("下班", "困", "猫猫"))

            self.assertTrue(storage.set_sticker_enabled(active[0].id, False))
            self.assertEqual(storage.list_sticker_assets("100"), [])
            self.assertEqual(len(storage.list_sticker_assets("100", enabled_only=False)), 1)

    def test_sticker_asset_reuses_same_ocr_with_different_hash(self) -> None:
        with project_temp_directory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()
            context = MessageContext(
                group_id="100",
                user_id="42",
                message_id="m-sticker-1",
                plain_text="",
                raw_message="[CQ:image]",
            )
            first = StickerCandidate(
                url="https://example.test/a.png",
                file="a.png",
                description="terminal screenshot sticker",
                ocr_text=">> XiaoMM >> so late, still awake",
                mood="sleepy",
                usage="use when someone is still awake late",
                tags=("late", "sleep"),
                confidence=0.86,
            )
            second = StickerCandidate(
                url="https://example.test/b.jpg",
                file="b.jpg",
                description="same terminal text with different compression",
                ocr_text=">> XiaoMM >> so late, still awake",
                mood="sleepy",
                usage="use as a late-night reminder",
                tags=("late", "awake"),
                confidence=0.91,
            )

            first_asset = storage.upsert_sticker_asset(
                context,
                first,
                local_path=str(Path(tmp) / "a.png"),
                sha256="hash-a",
            )
            second_asset = storage.upsert_sticker_asset(
                replace(context, message_id="m-sticker-2"),
                second,
                local_path=str(Path(tmp) / "b.jpg"),
                sha256="hash-b",
            )
            active = storage.list_sticker_assets("100", enabled_only=False)

            self.assertIsNotNone(first_asset)
            self.assertIsNotNone(second_asset)
            self.assertEqual(second_asset.id, first_asset.id)
            self.assertEqual(len(active), 1)
            self.assertEqual(active[0].local_path, str(Path(tmp) / "a.png"))
            self.assertEqual(active[0].sha256, "hash-a")
            self.assertEqual(active[0].url, "https://example.test/b.jpg")
            self.assertEqual(active[0].hit_count, 2)

    def test_sticker_file_refs_prefer_url_then_base64_without_local_path_fallback(self) -> None:
        with project_temp_directory() as tmp:
            sticker_path = Path(tmp) / "meme.gif"
            sticker_path.write_bytes(b"fake image")
            asset = StickerAssetRecord(
                id=7,
                group_id="100",
                source_user_id="42",
                source_message_id="m-sticker",
                url="https://example.test/meme.gif",
                file="meme.gif",
                local_path=str(sticker_path),
                sha256="abc",
                description="reaction sticker",
                ocr_text="",
                mood="funny",
                usage="use for a light reaction",
                tags=("funny",),
                confidence=0.86,
                enabled=True,
                created_at=1,
                updated_at=1,
                last_seen_at=1,
            )

            refs = sticker_file_refs(asset)
            file_ref = sticker_file_ref(asset)
            base64_ref = "base64://" + base64.b64encode(b"fake image").decode("ascii")

            self.assertEqual(file_ref, "https://example.test/meme.gif")
            self.assertEqual(refs, ("https://example.test/meme.gif", base64_ref))
            self.assertNotIn(str(sticker_path.resolve()), refs)

    def test_delete_sticker_asset_and_local_file(self) -> None:
        with project_temp_directory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            store = StickerLocalStore(config)
            sticker_dir = config.resolve_path(config.stickers.storage_dir) / "100"
            sticker_dir.mkdir(parents=True)
            sticker_file = sticker_dir / "meme.png"
            sticker_file.write_bytes(b"fake image")

            storage = BotStorage.from_config(config)
            storage.setup()
            context = MessageContext(
                group_id="100",
                user_id="42",
                message_id="m-sticker",
                plain_text="",
                raw_message="[CQ:image]",
            )
            candidate = StickerCandidate(
                url="https://example.test/meme.png",
                file="meme.png",
                description="一张猫猫下班表情包",
                mood="疲惫",
                usage="适合聊到下班、犯困或想摆一下时使用",
                tags=("下班",),
                confidence=0.86,
            )
            asset = storage.upsert_sticker_asset(
                context,
                candidate,
                local_path=str(sticker_file),
                sha256="abc",
            )

            deleted_asset = storage.delete_sticker_asset(asset.id if asset else 0)
            deleted_file = store.delete_saved_file(deleted_asset.local_path if deleted_asset else "")

            self.assertIsNotNone(deleted_asset)
            self.assertTrue(deleted_file)
            self.assertFalse(sticker_file.exists())
            self.assertEqual(storage.list_sticker_assets("100", enabled_only=False), [])

    def test_sticker_usage_is_counted_by_day(self) -> None:
        with project_temp_directory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()
            context = MessageContext(
                group_id="100",
                user_id="42",
                message_id="m-sticker",
                plain_text="",
                raw_message="[CQ:image]",
            )
            candidate = StickerCandidate(
                url="https://example.test/meme.png",
                file="meme.png",
                description="reaction sticker",
                mood="funny",
                usage="use for a quick reaction",
                tags=("funny",),
                confidence=0.86,
            )
            asset = storage.upsert_sticker_asset(
                context,
                candidate,
                local_path=str(Path(tmp) / "meme.png"),
                sha256="abc",
            )

            self.assertIsNotNone(asset)
            storage.record_sticker_sent(asset.id, usage_date="2026-07-05", sent_at=100)
            storage.record_sticker_sent(asset.id, usage_date="2026-07-05", sent_at=200)
            storage.record_sticker_sent(asset.id, usage_date="2026-07-06", sent_at=300)
            refreshed = storage.get_sticker_asset(asset.id)
            daily = storage.list_sticker_usage_daily("100", "2026-07-05")

            self.assertIsNotNone(refreshed)
            self.assertEqual(refreshed.send_count, 3)
            self.assertEqual(refreshed.last_sent_at, 300)
            self.assertEqual(storage.count_sticker_usage(asset.id, "2026-07-05"), 2)
            self.assertEqual(storage.count_sticker_usage(asset.id, "2026-07-06"), 1)
            self.assertEqual(storage.count_sticker_usage(asset.id), 3)
            self.assertEqual(daily[0]["send_count"], 2)

    def test_unused_sticker_cleanup_removes_assets_after_72_hours(self) -> None:
        with project_temp_directory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            store = StickerLocalStore(config)
            sticker_dir = config.resolve_path(config.stickers.storage_dir) / "100"
            sticker_dir.mkdir(parents=True)
            old_file = sticker_dir / "old.png"
            stale_file = sticker_dir / "stale.png"
            recent_file = sticker_dir / "recent.png"
            for path in (old_file, stale_file, recent_file):
                path.write_bytes(b"fake image")

            storage = BotStorage.from_config(config)
            storage.setup()
            context = MessageContext(
                group_id="100",
                user_id="42",
                message_id="m-sticker",
                plain_text="",
                raw_message="[CQ:image]",
            )

            def save_asset(name: str, path: Path) -> StickerAssetRecord:
                asset = storage.upsert_sticker_asset(
                    replace(context, message_id=f"m-{name}"),
                    StickerCandidate(
                        url=f"https://example.test/{name}.png",
                        file=f"{name}.png",
                        description=f"{name} sticker",
                        mood="funny",
                        usage=f"use {name}",
                        tags=(name,),
                        confidence=0.86,
                    ),
                    local_path=str(path),
                    sha256=f"hash-{name}",
                )
                self.assertIsNotNone(asset)
                return asset

            old_asset = save_asset("old", old_file)
            stale_asset = save_asset("stale", stale_file)
            recent_asset = save_asset("recent", recent_file)
            now = 1_800_000_000
            ttl_seconds = 72 * 60 * 60
            old_at = now - ttl_seconds - 10
            recent_at = now - ttl_seconds + 10
            storage.record_sticker_sent(stale_asset.id, usage_date="2026-07-01", sent_at=old_at)
            storage.record_sticker_sent(recent_asset.id, usage_date="2026-07-02", sent_at=recent_at)
            with storage._connect() as conn:
                conn.execute(
                    "UPDATE sticker_assets SET created_at = ?, updated_at = ?, last_seen_at = ? WHERE id = ?",
                    (old_at, old_at, old_at, old_asset.id),
                )
                conn.execute(
                    "UPDATE sticker_assets SET created_at = ?, updated_at = ?, last_seen_at = ? WHERE id = ?",
                    (old_at, old_at, old_at, stale_asset.id),
                )
                conn.execute(
                    "UPDATE sticker_assets SET created_at = ?, updated_at = ?, last_seen_at = ? WHERE id = ?",
                    (old_at, old_at, old_at, recent_asset.id),
                )

            deleted = storage.delete_unused_sticker_assets(ttl_seconds, now=now)
            deleted_ids = {asset.id for asset in deleted}
            deleted_files = {asset.id for asset in deleted if store.delete_saved_file(asset.local_path)}

            self.assertEqual(deleted_ids, {old_asset.id, stale_asset.id})
            self.assertEqual(deleted_files, deleted_ids)
            self.assertIsNone(storage.get_sticker_asset(old_asset.id))
            self.assertIsNone(storage.get_sticker_asset(stale_asset.id))
            self.assertIsNotNone(storage.get_sticker_asset(recent_asset.id))
            self.assertEqual(storage.count_sticker_usage(stale_asset.id), 0)
            self.assertFalse(old_file.exists())
            self.assertFalse(stale_file.exists())
            self.assertTrue(recent_file.exists())

    def test_sticker_cleanup_claim_runs_once_per_interval(self) -> None:
        storage = InMemoryBotStorage()
        try:
            storage.setup()

            self.assertTrue(storage.claim_sticker_cleanup(24 * 60 * 60, now=1000))
            self.assertFalse(storage.claim_sticker_cleanup(24 * 60 * 60, now=1000 + 60))
            self.assertTrue(storage.claim_sticker_cleanup(24 * 60 * 60, now=1000 + 24 * 60 * 60))
        finally:
            storage.connection.close()


if __name__ == "__main__":
    unittest.main()


