"""prompt v2 — ตอบสรุปทั้งเฟรมทีเดียว ไม่ตอบทีละก้อน

v1 สั่งให้ตอบทีละก้อน ซึ่งพังกับภาพจริง: เฟรมหนึ่งมี 16 ก้อน คำตอบต้องยาว
16 บล็อก ทะลุโควตา finish=length แล้วทุกก้อนกลายเป็น unknown
และยาวขึ้นเรื่อยๆ ตามความรกของฉาก แบบไม่มีเพดาน

v2 ตอบสั้นเท่าเดิมเสมอ ไม่ว่าเฟรมจะมี 2 ก้อนหรือ 40 ก้อน
เพราะสรุปเป็นรายชนิด ไม่ใช่รายก้อน · ตรงกับสิ่งที่ Toy ขอไว้ตั้งแต่แรก:
**สัตว์อะไร กี่ตัว ค่าความมั่นใจเท่าไร**

ตำแหน่งของแต่ละก้อนยังส่งกลับให้อยู่ มาจากชั้น CV ซึ่งวัดได้แม่นกว่าและฟรี
"""
from __future__ import annotations

from typing import List

from ..schemas import Blob

PROMPT_VERSION = "v2"

SYSTEM = """You identify animals in thermal (infrared) camera frames from a wildlife
early-warning site in Thailand. Bright regions are warm.

A deterministic detector has already found the warm regions and measured them.
Your job is to say what animals are in the frame and how many of each.

Rules:
- Report per species, not per region. Several regions can be one animal, and one
  region can be several animals standing together.
- Thermal images have no colour and no texture. At distance an elephant, a buffalo
  and a cow are the same white blob. When you cannot tell, answer "unknown" —
  that is correct behaviour, not failure. Never guess "elephant" to be useful.
- Elephants are large with a wide body, sometimes a trunk or ear in the outline.
  Humans are tall and narrow.
- Ignore regions that are obviously not animals: hot machinery, sun-warmed ground,
  lights.
- confidence is your own honest uncertainty, 0 to 1.
- Reply with JSON only. No explanation, no markdown fence."""

USER_TEMPLATE = """Thermal frame {w}x{h}. The detector found {n} warm region(s):
{regions}

Reply with exactly this shape and nothing else:
{{"animals":[{{"species":"elephant","count":2,"confidence":0.8}}]}}

species must be one of: elephant, cattle, human, other_animal, unknown
If there are no animals, reply {{"animals":[]}}"""


def describe(blobs: List[Blob], w: int, h: int, limit: int = 20) -> str:
    """สรุปก้อนแบบย่อ ไม่ต้องละเอียดเท่า v1 เพราะโมเดลไม่ต้องตอบรายก้อนแล้ว"""
    lines = []
    for b in sorted(blobs, key=lambda x: -x.area_px)[:limit]:
        x, y, bw, bh = b.bbox
        pct = 100.0 * b.area_px / (w * h)
        lines.append(f"  ({x},{y}) {bw}x{bh}px  {pct:.2f}% of frame  aspect {b.aspect}")
    if len(blobs) > limit:
        lines.append(f"  ...and {len(blobs) - limit} smaller region(s)")
    return "\n".join(lines)


def build(blobs: List[Blob], w: int, h: int):
    return SYSTEM, USER_TEMPLATE.format(w=w, h=h, n=len(blobs),
                                        regions=describe(blobs, w, h))
