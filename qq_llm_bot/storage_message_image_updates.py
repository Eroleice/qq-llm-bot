from __future__ import annotations


def update_image_descriptions(
    storage: object,
    group_id: str,
    message_id: str,
    descriptions: list[str],
) -> None:
    if not descriptions:
        return
    with storage._connect() as conn:
        rows = conn.execute(
            """
            SELECT id
            FROM message_attachments
            WHERE group_id = ?
              AND message_id = ?
              AND attachment_type = 'image'
            ORDER BY id
            """,
            (str(group_id), str(message_id)),
        ).fetchall()
        for row, description in zip(rows, descriptions):
            if not str(description).strip():
                continue
            conn.execute(
                """
                UPDATE message_attachments
                SET summary = ?
                WHERE id = ?
                """,
                (description[:500], int(row["id"])),
            )
