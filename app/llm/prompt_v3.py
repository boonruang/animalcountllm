"""prompt v3 — ถามโมเดลตรงๆ ไม่มีชั้น CV มาบอกใบ้ก่อน

ตัดสินโดย Toy 2026-08-17: **API เส้นนี้วางตำแหน่งไว้ว่าเป็น LLM ไม่ใช่ CV**
ไซต์จะส่งมาทั้งภาพความร้อน greyscale และภาพสีปกติ จากกล้องหลายตัว
เป้าหมายเดียวคือ **จับช้างให้ได้** ไม่ว่าภาพจะมาแบบไหน

🔴 ทำไม v2 ถึงตอบว่าไม่มีสัตว์ทั้งที่ช้างเต็มเฟรม (วัดจริง 2026-08-17)

v2 ส่งข้อความนี้ไปให้โมเดล เมื่อชั้น CV หาก้อนร้อนไม่เจอ:

    Thermal frame 1041x628. The detector found 0 warm region(s):

แล้วปิดท้ายว่า 'If there are no animals, reply {"animals":[]}'
**เราเดินไปบอกมันเองว่าไม่มีอะไรให้ดู** โมเดลก็เชื่อ แล้วตอบตามที่สั่ง
พอถามภาพใบเดียวกันแบบเปล่าๆ โมเดลตัวเดิม endpoint เดิม ตอบว่า
"There are 2 elephants in the image" ทั้ง qwen3-vl-32b และ 30b-a3b

บทเรียน: **prompt ที่ยัดข้อสรุปผิดๆ เข้าไป แย่กว่าไม่มี prompt เลย**
โมเดลไม่ได้ตาบอด มันเชื่อสิ่งที่เราบอก

วัดจริงกับภาพช้างสองตัว (tests/1786846183763.jpg) บน qwen3-vl-32b-instruct:
| ยิงแบบไหน                       | ตอบ                          |
|---------------------------------|------------------------------|
| ภาพสีเต็มใบ                     | elephant count=2 conf 0.95   |
| แปลง greyscale จำลอง IR         | elephant count=2 conf 0.95   |
| ย่อช้างเหลือ ~200x125 กลางเฟรม  | elephant count=2 conf 0.90   |
| ผ่าน prompt v2 (blobs=0)        | {"animals":[]}  ❌           |

🔴 bbox เชื่อไม่ได้ **จึงไม่ขอ** โมเดลคืน [64,57,519,998] บนภาพสูง 628 px
คือทั้งเกินขอบภาพและสับสนระหว่าง xywh กับ x1y1x2y2 · ขอแต่สิ่งที่มันทำได้จริง
คือ **ชนิด จำนวน ความมั่นใจ** ตรงกับที่ Toy สั่งไว้ตั้งแต่วันแรกพอดี

`image_type` ให้โมเดลบอกเอง ไม่ใช่คำนวณฝั่งเรา เพราะเส้นนี้ไม่มี numpy แล้ว
มันเป็นข้อมูลสำหรับตรวจย้อนหลังว่าเฟรมนั้นเป็นภาพแบบไหน ไม่ได้เอาไป gate อะไร
"""
from __future__ import annotations

PROMPT_VERSION = "v3"

SYSTEM = """You are the animal detector for a wildlife early-warning site in Thailand.
The cameras there send two kinds of picture and you must handle both:

- thermal / infrared frames, greyscale, where bright usually means warm
- ordinary colour photographs, day or night

Look at the image yourself and report what animals are in it. Nothing has been
detected for you beforehand, so an empty answer means you genuinely see no animal,
not that some earlier stage found nothing.

Rules:
- Report per species with a total count for the frame.
- The site cares about elephants above everything else. Report an elephant whenever
  you see one, however far away or partly out of frame.
- Do not invent one either. In a thermal frame at distance an elephant, a buffalo and
  a cow are the same pale blob with no colour and no texture. When you cannot tell,
  answer "unknown" with a low confidence. That is the correct answer, not a failure.
- Ignore things that are warm but not alive: machinery, sun-warmed ground, lights,
  vehicles.
- confidence is your own honest uncertainty from 0 to 1. An elephant filling the frame
  in daylight deserves 0.95. A pale smudge at the treeline deserves 0.3.
- Reply with JSON only. No explanation, no markdown fence."""

USER_TEMPLATE = """Image {w}x{h} from camera {cam}. What animals are in it?

Reply with exactly this shape and nothing else:
{{"image_type":"thermal","animals":[{{"species":"elephant","count":2,"confidence":0.9}}]}}

image_type must be one of: thermal, colour, unclear
species must be one of: elephant, cattle, human, other_animal, unknown
If you see no animals, reply {{"image_type":"...","animals":[]}}"""


def build(w: int, h: int, camera_id: str = "unknown"):
    return SYSTEM, USER_TEMPLATE.format(w=w, h=h, cam=camera_id)
