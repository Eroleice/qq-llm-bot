from __future__ import annotations

import re


def _dashboard_user_id(value: str) -> str:
    user_id = str(value or "").strip()
    match = re.fullmatch(r"(?i)qq[:：]\s*(\d+)", user_id)
    return match.group(1) if match else user_id

def _dashboard_user_id_variants(value: str) -> list[str]:
    canonical = _dashboard_user_id(value)
    variants = [canonical]
    if canonical.isdigit():
        variants.extend([f"QQ:{canonical}", f"qq:{canonical}", f"QQ：{canonical}"])
    return list(dict.fromkeys(item for item in variants if item))
