"""prompt เวอร์ชัน 1 · แยกไฟล์และมีเวอร์ชันกำกับ เพราะทุกเฟรมบันทึกว่าใช้เวอร์ชันไหน

ถ้าไม่รู้ว่าคำตอบเมื่อวานมาจาก prompt ตัวไหน จะเทียบ local กับ OpenRouter ไม่ได้เลย
🔴 แก้ข้อความในนี้เมื่อไหร่ ต้องขึ้นเวอร์ชันใหม่ ห้ามแก้ทับเงียบๆ
"""
from __future__ import annotations

from typing import List

from ..schemas import Blob

PROMPT_VERSION = "v1"

SYSTEM = """You identify animals in thermal (infrared) camera frames from a wildlife
early-warning site in Thailand. Bright = warm.

You are given the frame AND a list of hot regions already measured by a deterministic
detector. The detector is reliable about WHERE things are and HOW MANY regions exist.
Your job is to say WHAT each region is.

Rules:
- Answer for each region id given. Do not invent regions.
- If a region is too small, too blurred, or genuinely ambiguous, answer "unknown".
  A thermal image has no colour and no texture: at distance, an elephant, a buffalo and
  a cow all look like the same white blob. Saying "unknown" is correct behaviour, not
  failure. Never guess "elephant" to make the answer useful.
- Elephants: large area, wide body, sometimes a trunk or ear outline in the silhouette.
- Humans: tall and narrow, aspect ratio well below 1.
- If two regions are clearly one animal split in half, list them in merged_ids.
- confidence is your own honest uncertainty from 0 to 1.
- Reply with JSON only. No explanation, no markdown fence."""

USER_TEMPLATE = """Frame {w}x{h}. {n} hot region(s) detected:
{regions}

Reply with exactly this JSON shape:
{{"calls":[{{"blob_id":1,"species":"elephant|cattle|human|other_animal|unknown","confidence":0.0,"reason":"few words"}}],"merged_ids":[]}}"""


def describe(blobs: List[Blob], w: int, h: int) -> str:
    """แปลงตัวเลขที่ CV วัดได้ให้โมเดลอ่านรู้เรื่อง

    ส่งขนาดเทียบเฟรมไปด้วย เพราะโมเดลเห็นภาพหลังย่อแล้ว มันไม่รู้ขนาดจริงเป็นพิกเซล
    """
    lines = []
    for b in blobs:
        x, y, bw, bh = b.bbox
        pct = 100.0 * b.area_px / (w * h)
        lines.append(
            f"  id={b.id} at ({x},{y}) size {bw}x{bh}px area={b.area_px}px "
            f"({pct:.2f}% of frame) aspect={b.aspect} fill={b.fill_ratio} "
            f"contrast={b.contrast}{' TOUCHES-EDGE' if b.touches_edge else ''}"
        )
    return "\n".join(lines)


def build(blobs: List[Blob], w: int, h: int):
    return SYSTEM, USER_TEMPLATE.format(w=w, h=h, n=len(blobs),
                                        regions=describe(blobs, w, h))
