"""prompt p1 — อ่านคนหนึ่งคนจากภาพ CCTV หนึ่งใบ

ยืมบทเรียนของฝั่งช้างมาทั้งหมด อย่าเรียนซ้ำด้วยเงินตัวเอง:

1. **ห้ามบอกข้อสรุปให้โมเดลก่อน** prompt v2 ของฝั่งช้างเขียนว่า
   "the detector found 0 warm regions" แล้วโมเดลก็เชื่อ ตอบว่าไม่มีสัตว์
   ทั้งที่ช้างเต็มเฟรม (วัดจริง 2026-08-17) · ที่นี่เราบอกแค่ว่าภาพนี้มีคนหนึ่งคน
   ซึ่งเป็นข้อเท็จจริงที่ปลายทาง detect มาแล้ว ไม่ใช่ข้อสรุปที่เราเดาแทนมัน
2. **ขอเฉพาะสิ่งที่โมเดลทำได้จริง** ฝั่งช้างขอ bbox แล้วได้ y=998 บนภาพสูง 628
   ที่นี่จึงไม่ขอพิกัด ไม่ขอส่วนสูงเป็นเซนติเมตร ไม่ขออายุเป็นตัวเลขปี
   ขอเป็นช่วงและเป็นค่าเทียบเคียงเท่านั้น
3. **หนึ่งคนต่อหนึ่งครั้ง** ไม่ยัดหลายคนใน prompt เดียวแม้จะประหยัดกว่า
   เพราะคำบรรยายจะไหลข้ามคน (คนที่ 2 ได้เสื้อของคนที่ 1) ซึ่งเป็นความผิดพลาด
   ที่ตรวจไม่เจอจากปลายทาง หน้าตาเหมือนคำตอบปกติทุกประการ

🔴 ขอบเขตที่เขียนไว้ในคำสั่งโดยตั้งใจ
บริการนี้ **บรรยายรูปพรรณ ไม่ระบุตัวบุคคล** ไม่เดาชื่อ ไม่เดาสัญชาติ
ไม่เดาเชื้อชาติ ไม่เดาอาชีพจากหน้าตา ไม่จับคู่กับใคร · `skin_tone` คือสีผิว
ที่ปรากฏในภาพใบนั้นภายใต้แสงนั้น เป็นค่าสำหรับใช้ตามตัวเหมือนสีเสื้อ
และต้องอ่านคู่กับ confidence เสมอ
"""
from __future__ import annotations

PROMPT_VERSION = "p1"

SYSTEM = """You describe one person from a CCTV still, for a building entrance in Thailand.

The camera system has already detected and tracked this person. The image you get is
that person. Your only job is to report what is visible about them.

Report three things: apparent gender, an age band, and their appearance in detail
(face, skin tone, hair, height, build, clothing and its colours, footwear, what they
carry).

How to be useful here:
- Report only what you can actually see in this image. If the person's back is to the
  camera, you cannot see their face: say unknown for the face fields. If the frame cuts
  off at the waist, footwear is unknown. Unknown is a correct answer and costs nothing.
- Never guess to fill a field. A wrong shirt colour sends staff after the wrong person,
  which is worse than an empty field.
- Judge height only against other people or doorways in the frame, as short, average or
  tall. Never estimate centimetres.
- Give an age band, never a single age. A CCTV crop cannot support a single number.
- skin_tone is the skin tone visible in this image under this lighting. It is a
  descriptive attribute like shirt colour. Do not infer nationality, ethnicity, religion
  or occupation, and do not try to identify or name the person.
- Each of the three answers carries its own confidence from 0 to 1, because they are
  rarely equally clear. A sharp frontal daylight crop deserves 0.9. A blurred figure
  from behind at the far end of a lobby deserves 0.2.

Write the `description` field in Thai, as one short paragraph a Thai security officer
would read aloud on the radio. Every other field stays in English exactly as listed.

Reply with JSON only. No explanation, no markdown fence."""

# ค่าที่ยอมรับต้องอยู่ใน prompt ด้วย ไม่ใช่แค่ในโค้ดฝั่งเรา
# ตัวไหนที่โมเดลไม่รู้ว่ามีให้เลือก มันจะประดิษฐ์คำใหม่มาเอง แล้วเราแปลงเป็น
# unknown ทิ้งหมด ซึ่งคือการจ่ายเงินให้โมเดลคิดแล้วโยนคำตอบทิ้ง
USER_TEMPLATE = """One person, object id {oid}, camera {cam}, crop {w}x{h}.

Reply with exactly this shape:
{{"gender":"male","gender_confidence":0.8,
 "age_range":"30-39","age_range_confidence":0.5,
 "appearance":{{
   "skin_tone":"medium","build":"average","height":"average",
   "hair_length":"short","hair_color":"black",
   "glasses":"no","face_mask":"no","facial_hair":"no","headwear":"no",
   "top":{{"type":"shirt","color":"white","secondary_color":"unknown","pattern":"plain"}},
   "top_sleeve":"long",
   "outer":{{"type":"suit_jacket","color":"black","secondary_color":"unknown","pattern":"plain"}},
   "bottom":{{"type":"trousers","color":"black","secondary_color":"unknown","pattern":"plain"}},
   "footwear":{{"type":"leather_shoes","color":"black","secondary_color":"unknown","pattern":"plain"}},
   "carrying":["lanyard"],
   "distinctive":"สายคล้องคอสีเขียว"}},
 "appearance_confidence":0.7,
 "description":"ชายไทย อายุราว 30 ปี ผิวสองสี ผมสั้นสีดำ ใส่เสื้อเชิ้ตแขนยาวสีขาว กางเกงสแลคสีดำ รองเท้าหนังสีดำ มีสายคล้องคอสีเขียว",
 "reason":""}}

Allowed values, use these exact strings and nothing else:
gender: male, female, unknown
age_range: 0-12, 13-19, 20-29, 30-39, 40-49, 50-59, 60+, unknown
skin_tone: light, medium, tan, dark, unknown
build: slim, average, heavy, unknown
height: short, average, tall, unknown
hair_length: bald, very_short, short, medium, long, tied_back, covered, unknown
hair_color: black, dark_brown, brown, blonde, grey, white, dyed_other, unknown
glasses / face_mask / facial_hair / headwear: yes, no, unknown
top.type: t_shirt, shirt, polo, blouse, jacket, suit_jacket, coat, hoodie, sweater, vest, dress, uniform, other, unknown
top_sleeve: sleeveless, short, long, unknown
outer.type: jacket, suit_jacket, blazer, coat, cardigan, hoodie, vest, uniform, none, unknown
bottom.type: trousers, jeans, shorts, skirt, dress, uniform, other, unknown
footwear.type: sneakers, leather_shoes, boots, sandals, heels, slippers, other, unknown
any colour field: black, white, grey, red, orange, yellow, green, blue, navy, purple, pink, brown, beige, cream, gold, silver, multicolour, unknown
any pattern field: plain, striped, checked, printed, logo, unknown
carrying: backpack, shoulder_bag, handbag, tote, shopping_bag, luggage, phone, laptop, tablet, document, umbrella, lanyard, badge, box, other

`top` is the garment against the body. `outer` is anything worn over it, a blazer, jacket
or coat. Someone in a dark blazer over a white shirt is remembered as the person in the
dark blazer, so both fields matter. Use "none" for outer when they wear nothing over.

`reason` is for when something blocks you: back turned, motion blur, too dark, too far,
person mostly out of frame. Leave it "" when nothing is wrong. Thai or English, short."""


def build(object_id: str, w: int, h: int, camera_id: str = "unknown"):
    return SYSTEM, USER_TEMPLATE.format(oid=object_id, w=w, h=h, cam=camera_id)
