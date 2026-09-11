"""prompt f1 — เฟรมเดียว ไม่มีใครถูกชี้ ให้โมเดลหาคนในเฟรมเอง

ใช้กับ `POST /people/v1/frames` ซึ่ง body หน้าตาเหมือนฝั่งช้างเป๊ะ
(camera_id · image_base64 · ts · note) ตามที่ Toy สั่ง 2026-09-11

🔴 ต่างจาก p1 ตรงไหน และทำไมถึงไม่รวมเป็นอันเดียว

p1 ได้ภาพที่มีคนเดียวมาแล้ว หน้าที่คือ "อ่านคนนี้ให้ละเอียด"
f1 ได้ทั้งเฟรม หน้าที่คือ "มีใครบ้าง แล้วอ่านทีละคน" ซึ่งเพิ่มงานที่ p1 ไม่มี:
แยกคนออกจากกัน ไม่นับซ้ำ ไม่นับเงาสะท้อนในกระจกเป็นอีกคน และบอกตำแหน่ง
ให้คนอ่านชี้ตัวถูก · ยัดสองอย่างนี้ลง prompt เดียวแล้วสั่งให้ "ทำอย่างใดอย่างหนึ่ง
แล้วแต่กรณี" คือวิธีที่ทำให้ทั้งสองอย่างแย่ลงพร้อมกัน

ชุดค่าที่ยอมรับ **ไม่ก๊อปมา** import จาก prompt_p1.ALLOWED ก้อนเดียวกัน
บทเรียนสดๆ 2026-09-11: `laptop` หายไปจากคำตอบเพราะชุดค่าในโค้ดกับที่โมเดลรู้
ไม่ตรงกัน สองก้อนที่ต้องตรงกันแต่แก้คนละที่ จะไม่ตรงกันเสมอ

🔴 ห้ามขอ bbox เด็ดขาด · วัดมาแล้วฝั่งช้าง โมเดลคืน y=998 บนภาพสูง 628
ตำแหน่งจึงขอเป็น **ข้อความ** (`where`) ซึ่งเป็นสิ่งที่มันทำได้จริง
คนอ่านแล้วชี้ตัวถูก ส่วนโปรแกรมจับคู่กับ track id ไม่ได้ **ซึ่งเป็นราคาที่รู้ล่วงหน้า
ไม่ใช่เซอร์ไพรส์** ปลายทางที่ต้องผูก id ให้ใช้ /v1/persons พร้อม bbox แทน
"""
from __future__ import annotations

from .prompt_p1 import ALLOWED

PROMPT_VERSION = "f1"

SYSTEM = """You describe every person visible in one CCTV still from a building entrance
in Thailand. Nobody has been detected for you beforehand: you find the people yourself.

For each person report apparent gender, an age band, and their appearance in detail
(face, skin tone, hair, height, build, clothing and its colours, footwear, what they
carry), plus where they are in the frame.

How to be useful here:
- List a person once. A reflection in glass or a mirror is not a second person, and
  neither is someone already listed seen through a doorway.
- Include people in the background only while you can still say something real about
  them. A dark smudge at the far end of a lobby is not worth a row.
- Report only what you can actually see on each person. Unknown is a correct answer and
  costs nothing. Never guess to fill a field: a wrong shirt colour sends staff after the
  wrong person, which is worse than an empty field.
- Do not let one person's clothing leak onto the next. Look at each person again before
  you describe them.
- Judge height only against other people or doorways in the frame. Never estimate
  centimetres. Give an age band, never a single age.
- skin_tone is the skin tone visible in this image under this lighting. It is a
  descriptive attribute like shirt colour. Do not infer nationality, ethnicity, religion
  or occupation, and do not try to identify or name anyone.
- Confidence from 0 to 1, honestly. Someone sharp and close deserves 0.9. Someone small
  and half hidden deserves 0.3.

Write `description` in Thai, one short paragraph a Thai security officer would read
aloud on the radio. Every other field stays in English exactly as listed.

Reply with JSON only. No explanation, no markdown fence."""

USER_TEMPLATE = """Frame {w}x{h} from camera {cam}. Who is in it?

List people nearest the camera first, at most {cap}. Reply with exactly this shape:
{{"people":[{{
 "ref":"P1",
 "where":"centre foreground, walking towards the camera, in front of the other two",
 "gender":"female","gender_confidence":0.9,
 "age_range":"40-49","age_range_confidence":0.6,
 "appearance":{{
   "skin_tone":"medium","build":"average","height":"average",
   "hair_length":"short","hair_color":"black",
   "glasses":"yes","face_mask":"no","facial_hair":"no","headwear":"no",
   "top":{{"type":"shirt","color":"black","secondary_color":"unknown","pattern":"plain"}},
   "top_sleeve":"long",
   "outer":{{"type":"none","color":"unknown","secondary_color":"unknown","pattern":"unknown"}},
   "bottom":{{"type":"trousers","color":"black","secondary_color":"unknown","pattern":"plain"}},
   "footwear":{{"type":"leather_shoes","color":"white","secondary_color":"unknown","pattern":"plain"}},
   "carrying":["phone"],
   "distinctive":"รองเท้าหนังสีขาว"}},
 "appearance_confidence":0.7,
 "description":"หญิง อายุราว 40 ปี ผิวสองสี ผมสั้นสีดำ ใส่แว่นตา เสื้อแขนยาวสีดำ กางเกงสีดำ รองเท้าหนังสีขาว ถือโทรศัพท์",
 "reason":""}}]}}

`ref` numbers the people in this frame only: P1, P2, P3. It means nothing outside this
answer, so do not try to recognise anyone or reuse a number from another frame.

`where` must let a person looking at the same picture point at the right one: position in
the frame, what they are doing, and how they sit relative to the others. Do not give
pixel coordinates, they are not what you are good at.

If there is nobody in the frame, reply {{"people":[]}}. That is a real answer, not a
failure, and it is the right one for an empty lobby.

{allowed}

`reason` is for when something blocks you on that person: back turned, motion blur, too
dark, too far, mostly out of frame. Leave it "" when nothing is wrong. Short."""


def build(w: int, h: int, camera_id: str = "unknown", cap: int = 12):
    return SYSTEM, USER_TEMPLATE.format(w=w, h=h, cam=camera_id, cap=cap,
                                        allowed=ALLOWED)
