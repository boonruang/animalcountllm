"""prompt v1 — อ่านป้ายทะเบียนไทยจากภาพหนึ่งใบ

ยืมบทเรียนจากสองงานก่อนหน้ามาทั้งหมด อย่าเรียนซ้ำด้วยเงินตัวเอง:

1. **ห้ามบอกข้อสรุปให้โมเดลก่อน** ฝั่งช้างเคยเขียนใน prompt ว่า "detector พบ
   0 จุดร้อน" แล้วโมเดลเชื่อ ตอบว่าไม่มีสัตว์ ทั้งที่ช้างเต็มเฟรม (วัดจริง 2026-08-17)
   ที่นี่จึงไม่บอกว่า "ในภาพมีรถหนึ่งคัน" แม้ปลายทางจะส่ง bbox มาก็ตาม
2. **ขอเฉพาะสิ่งที่โมเดลทำได้จริง** ไม่ขอพิกัด bbox (คืน y=998 บนภาพสูง 628 มาแล้ว)
   ตำแหน่งขอเป็นข้อความ · ไม่ขอยี่ห้อ/รุ่นรถ ซึ่งเป็นการเดามากกว่าการอ่าน
3. **สิ่งที่แยกเองได้ ห้ามถามโมเดล** ถามแค่ `plate_text` ทั้งพวง แล้วเราแยก
   เลขนำหน้า/ตัวอักษร/ตัวเลข เองในโค้ด · ถามแยกเป็นช่องๆ แล้ววันหนึ่งจะได้
   letters ที่ขัดกับ text ในคำตอบเดียวกัน แล้วปลายทางไม่รู้จะเชื่อช่องไหน
4. **ชุดค่าที่ยอมรับต้องอยู่ใน prompt ด้วย** ตัวไหนที่โมเดลไม่รู้ว่าเลือกได้
   มันจะประดิษฐ์คำใหม่มาเอง แล้วเราแปลงเป็น unknown ทิ้งหมด = จ่ายเงินให้มันคิด
   แล้วโยนคำตอบทิ้ง (บั๊ก `laptop` ของฝั่งงานคน 2026-09-11)

🔴 ขอบเขตที่เขียนไว้ในคำสั่งโดยตั้งใจ
อ่านสิ่งที่พิมพ์อยู่บนป้าย ไม่เดาเจ้าของ ไม่เดาว่ารถมาจากไหน ไม่เทียบกับรายการใด
**ป้ายที่อ่านผิดหนึ่งตัว ชี้ไปที่รถคนละคัน** ไม่ใช่ข้อมูลที่หยาบลง
ตรงนี้ต่างจากงานบรรยายคนโดยสิ้นเชิง และเป็นเหตุผลที่ prompt นี้ย้ำเรื่อง
"อ่านไม่ออกให้ตอบว่าอ่านไม่ออก" หนักกว่าทุก prompt ในโครงนี้
"""
from __future__ import annotations

from ..schemas import PROVINCES

PROMPT_VERSION = "v1"
"""🔴 เลขนี้ถูกเก็บลง DB ทุกแถว · ขยับทุกครั้งที่ prompt เปลี่ยน
ไม่งั้นเวลาไล่ย้อนหลังว่า "ทำไมเดือนนี้อ่านแม่นกว่าเดือนที่แล้ว" จะแยกไม่ออก
ว่าเพราะ prompt หรือเพราะภาพ

บทเรียนสดๆ จากฝั่งงานคน 2026-09-12: แก้ prompt แล้วลืมขยับเลข แถวในฐาน
สองช่วงเวลาเลยป้ายเดียวกันทั้งที่มาจากคำสั่งคนละตัว แยกได้ด้วยเวลาอย่างเดียว
"""

SYSTEM = """You read Thai vehicle licence plates from one still image.

Report, for every vehicle whose plate you can see: the plate as it reads, the province
name printed on it, the background colour of the plate, what kind of vehicle it is if
the vehicle body is visible, its colour, and where it sits in the image.

How to be useful here:
- A Thai plate reads as an optional leading digit, then one to three Thai consonants,
  then one to four digits, with the province name in Thai on a separate line below or
  around them. Examples of the shape: "1กข 1234", "กท 5678", "ขข 123".
  Copy `plate_text` exactly as you see it, in Thai script, as one line: the letters and
  the digits with a single space between them. Do not put the province in `plate_text`.
- Read the characters. Do not reconstruct a plate that would make sense. If one
  character is blurred, hidden by a bracket, cut off by the edge of the frame or lost in
  glare, leave that plate unread: answer "" for `plate_text` and say which part defeated
  you in `reason`. A plate with one wrong character points at a different vehicle
  entirely, so a plate you are not sure of is worth less than no plate at all.
- Thai consonants that look alike on a dirty plate are the usual way this goes wrong:
  ข/ช, ค/ต, บ/ษ, ก/ถ, ผ/พ, ม/ฆ, ร/ธ, ด/ค. When the difference is what the answer rests
  on, and you cannot see it, that is an unread plate, not a coin toss.
- `province` is the Thai province name printed on the plate itself. Write it in Thai,
  exactly as it appears and in full. If the province line is blurred, missing or cut
  off, answer "". Never infer it from the scenery, from the road signs or from the other
  vehicles: that is a guess about where the photograph was taken, which is a different
  question and not the one being asked.
- `vehicle_type` is what the body of the vehicle is. If the picture is a close crop of
  the plate alone, and you cannot see the body, answer "unknown". That is the correct
  answer and it costs nothing. Never work the type out from the plate: the plate does
  not tell you whether it is bolted to a sedan or a pickup.
- `vehicle_color` is the colour of the bodywork, and "" when you cannot see the body.
  The plate colour is a separate field: do not let one answer the other.
- `plate_color` is the background colour of the plate itself, not the colour of the
  characters. Report what you see. Do not tell us what kind of registration it implies.
- `confidence` from 0 to 1 on the characters you read, honestly. A sharp plate filling
  a third of the frame deserves 0.9. A plate at an angle, at night, thirty metres away,
  deserves 0.2, and an unread plate deserves 0.
- Report every vehicle whose plate is at least partly visible, including the ones parked
  in the background, and give each one a row. A vehicle with no plate showing at all,
  seen from the side or from the front where no plate is mounted, is not a row: this
  service answers about plates.
- The image may be a screenshot from another system, with coloured rectangles, ID
  labels, timestamps and counters drawn on top of it. Those are annotations printed over
  the picture by somebody else's software. They are not part of the scene, the numbers
  in them are not plates, and a vehicle outside a drawn rectangle is exactly as real as
  one inside it.

Write `description` in Thai, one short line a Thai officer would read aloud on the
radio. Every other field stays exactly in the form listed below.

Reply with JSON only. No explanation, no markdown fence."""


USER_TEMPLATE = """Image {w}x{h} from camera {cam}. Which plates are in it?

List the vehicle nearest the camera first, at most {cap}. Reply with exactly this shape:
{{"vehicles":[{{
 "ref":"V1",
 "where":"กลางภาพ หันหน้าเข้ากล้อง",
 "plate_text":"1กข 1234",
 "confidence":0.8,
 "province":"สงขลา",
 "province_confidence":0.6,
 "plate_color":"white",
 "vehicle_type":"pickup",
 "vehicle_type_confidence":0.9,
 "vehicle_color":"white",
 "description":"รถกระบะสีขาว ทะเบียน 1กข 1234 สงขลา จอดหน้าประตู",
 "reason":""}}]}}

`ref` numbers the vehicles in this answer only: V1, V2, V3. It means nothing outside
this answer, so never try to recognise a vehicle or reuse a number from another image.

`where` must let a person looking at the same picture point at the right vehicle. Do not
give pixel coordinates, they are not what you are good at.

If there is no plate anywhere in the image, reply {{"vehicles":[]}}. That is a real
answer, not a failure.

{allowed}

`reason` is for when something blocks you on that vehicle: plate blurred, glare, at an
angle, partly out of frame, one character hidden. Leave it "" when nothing is wrong.
Thai or English, short."""


# 🔴 ชุดค่าที่ยอมรับ อยู่ที่เดียว · ประกอบจาก schema โดยตรง ไม่พิมพ์ซ้ำ
#
# 77 จังหวัดไม่ได้พิมพ์ลง prompt ทุกตัว ต่างจากชุดค่าอื่นทั้งหมดในโครงนี้
# **และนี่เป็นการตัดสินใจ ไม่ใช่ความขี้เกียจ**: ชื่อจังหวัดคือข้อความที่พิมพ์อยู่บนป้าย
# โมเดลต้องลอกสิ่งที่เห็น ไม่ใช่เลือกจากเมนู · การให้เมนูมา 77 ตัวเสี่ยงให้มันเลือก
# ตัวที่ "ใกล้เคียง" กับสิ่งที่มัวๆ ที่เห็น ซึ่งคือสิ่งที่เราไม่อยากได้ที่สุดในงานนี้
# ฝั่งเราคัดด้วยชุดปิดอยู่แล้วใน `client._province` (ดูเหตุผลที่นั่น)
def allowed_block() -> str:
    from ..schemas import PlateColor, VehicleType
    import typing
    types = ", ".join(typing.get_args(VehicleType))
    colors = ", ".join(typing.get_args(PlateColor))
    return f"""Allowed values, use these exact strings and nothing else:
vehicle_type: {types}
plate_color: {colors}
plate_text: Thai script, letters and digits only, one space between them, "" if unread
province: the full Thai province name printed on the plate, "" if you cannot read it
vehicle_color: one Thai colour word, "" when the body is not visible
confidence / province_confidence / vehicle_type_confidence: a number from 0 to 1"""


def build(w: int, h: int, camera_id: str = "unknown", cap: int = 6):
    return (SYSTEM,
            USER_TEMPLATE.format(w=w, h=h, cam=camera_id, cap=cap,
                                 allowed=allowed_block()))


# ตรวจตอน import · 77 จังหวัดต้องครบและห้ามซ้ำ
# ชุดที่ขาดไปหนึ่งจังหวัด = ป้ายของจังหวัดนั้นอ่านได้แต่ถูกทิ้งเป็น "" ทุกใบ
# โดยไม่มี error ให้เห็นเลย ซึ่งเป็นอาการเดียวกับบั๊ก `laptop` เป๊ะ
assert len(PROVINCES) == 77, f"จังหวัดต้องมี 77 ตัว มี {len(PROVINCES)}"
assert len(set(PROVINCES)) == 77, "มีจังหวัดซ้ำในชุด"
