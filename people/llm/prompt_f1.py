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

🔴🔴 กฎเรื่อง overlay (เพิ่ม 2026-09-12) คือการแก้บั๊กที่ Toy ถามมาตรงๆ ว่า
"ทำไมบางภาพโมเดลตีความแปลก เขามองไม่เห็นทั้งภาพหรือ"

เปิดภาพตัวอย่างสองใบดูแล้ว **คำตอบคือมันเห็นทั้งภาพ และนั่นแหละคือปัญหา**

`iplus_sample1.jpg` ไม่ใช่เฟรมกล้อง มันคือ **ภาพถ่ายหน้าจอ dashboard ของทีมเขา**
ในพิกเซลมีครบ: แถบข้อความบนสุด "เข้า 11 ออก 5 อยู่ในตึก 6" · ปุ่ม play กลางจอ ·
กรอบเหลืองพร้อมป้าย ID:011/021/022 วาดทับ · และ **รูปหน้าคนสองใบแปะมุมขวา
ซึ่งคือคนสองคนเดิมในเฟรมนั้นเอง** โมเดลอ่านทั้งหมดเป็นเนื้อหาของภาพ
มันกำลังบรรยายรูปของจอมอนิเตอร์ ไม่ใช่บรรยายฉากที่กล้องเห็น

`iplus_sample2.jpg` เป็นเฟรมจริง แต่มีกรอบ เขียว/แดง/น้ำเงิน ฝังอยู่ในพิกเซล
prompt เดิมไม่เคยบอกว่ากรอบพวกนั้นคืออะไร โมเดลจึงอ่านว่า "นี่คือรายชื่อคนในภาพ"
แล้วตอบ 3 คน ทั้งที่มี รปภ.ชุดเขียวคำว่า SECURITY ยืนอยู่หลังประตู คนชุดชมพู
และคนเดินนอกอาคารอีกหลายคน **ไม่ใช่โมเดลมองไม่เห็น แต่เราสั่งมันผิด**

ทางแก้ระยะยาวคือขอเฟรมดิบจากทีมคุณสุชาติ · ทางแก้ที่นี่คือบอกให้มันรู้จัก overlay
"""
from __future__ import annotations

from .prompt_p1 import allowed_block, system_block

PROMPT_VERSION = "f4"
"""f1 -> f2 -> f3 (2026-09-12) · uniform, emotion, group, กฎเรื่อง overlay ที่ฝังมาในภาพ
f3 -> f4 (2026-09-14) · `mobility` เดินเอง/รถเข็นเด็ก/วีลแชร์/ถูกอุ้ม **และ**
เติมเด็กในรถเข็นกับเด็กที่ถูกอุ้มเข้าไปในรายการ "ใครนับเป็นคน" ของขั้นตอนที่หนึ่ง
เส้นนี้เป็นเส้นเดียวที่โมเดลเป็นคนตัดสินว่ามีกี่คน เด็กในรถเข็นที่ไม่ถูกนับ
จึงหายทั้งแถว ไม่ใช่แค่ช่องเดียวที่ว่าง (เส้น persons ปลายทางชี้มาเองอยู่แล้ว)

🔴 f2 -> f3 · **วัดจริงบน prod แล้ว f2 ไม่ได้แก้ปัญหา**
f2 เขียนกฎ overlay ไว้เป็นข้อควรระวังข้อหนึ่งในรายการ แล้วยิง iplus_sample2 บน prod
ได้ 3 คนเท่าเดิมเป๊ะ คือสามคนที่มีกรอบวาดทับ · รปภ.ชุดเขียวหลังประตูยังหายไปเหมือนเดิม
**และพิสูจน์แล้วว่าไม่ใช่เรื่องความละเอียด** ชี้ bbox ของ รปภ. เข้า /v1/persons ตรงๆ
โมเดลอ่านออกครบ ตอบ uniform.kind=security และลอกคำว่า "SECURITY" จากหลังเสื้อมาได้
f3 เลยเปลี่ยนจาก "ข้อควรระวังข้อหนึ่ง" เป็น **ขั้นตอนที่หนึ่งที่ต้องทำก่อนบรรยายใคร**
บทเรียน: กฎที่สำคัญที่สุดของ prompt ต้องเป็นลำดับขั้นตอน ไม่ใช่ bullet ปนกับข้ออื่น

⚠️ **เปลี่ยน prompt แล้วจำนวนคนที่เจอขยับได้** วัดมาแล้ว 2026-09-11: แค่เพิ่มคำถาม
เรื่องทิศ คนใน iplus_sample1 ขยับจาก 2 เป็น 3 · รอบนี้แตะกฎเรื่อง "ใครนับเป็นคน"
ตรงๆ ยิ่งต้องยิงภาพชุดเดิมซ้ำก่อนเชื่อเลขใดๆ ห้ามสรุปจากภาพใบเดียว
"""

SYSTEM = """You describe every person visible in one CCTV still from a building entrance
in Thailand. Nobody has been detected for you beforehand: you find the people yourself.

For each person report which way they are facing, apparent gender, an age band, their
facial expression, whether they are in a uniform, whether they are walking or being
pushed in a pram or a wheelchair, who they appear to be with, their
appearance in detail (face, skin tone, hair, height, build, clothing and its colours,
footwear, what they carry), and where they are in the frame.

Work in this order.

Step one, before anything else: find every person in the picture.

This still is often a screenshot of a monitoring dashboard, not a clean camera frame.
Coloured rectangles, ID labels, timestamps, counters, arrows, zone outlines, a play
button and small face thumbnails along one edge are drawn on top of the picture by
another system. They are annotations, not content.

So sweep the whole image yourself and count the people you can see: near the camera, at
the back, through a doorway, outside the glass, at the edges of the frame. A small child
sitting in a pram and a baby carried in someone's arms are people in this frame too, and
they are the two the eye skips most often: each of them gets a row of their own, separate
from the adult pushing or holding them. Do that sweep first, and do it as if no rectangle
had been drawn at all. Those rectangles mark
whoever that other system happened to be tracking at that moment. They are not the list
of people present, and a person standing outside a drawn rectangle is exactly as real as
one inside it. If your final answer lists only the people who happen to sit inside drawn
rectangles, you have described the annotations instead of the scene: go back and look
again. Never copy a number off a counter into your answer.

Step two: describe each person you found.

How to be useful here:
- A face thumbnail along an edge is somebody already in the frame shown a second time,
  not an extra person.
- List a person once. A reflection in glass or a mirror is not a second person, and
  neither is someone already listed seen through a doorway.
- Include people in the background only while you can still say something real about
  them. A dark smudge at the far end of a lobby is not worth a row. Somebody standing
  outside the glass doors is still a person in this frame: list them, and say so in
  `where`.
- The camera looks at people as they come in. Someone facing you is coming in ("in").
  Someone whose back you see is going out ("out"). Someone side-on, or too obscured to
  tell, is "unknown", and that is a real answer: guessing here corrupts a head count.
  Report the way they face in this image, not where you imagine they are heading.
- Report only what you can actually see on each person. Unknown is a correct answer and
  costs nothing. Never guess to fill a field: a wrong shirt colour sends staff after the
  wrong person, which is worse than an empty field.
- Do not let one person's clothing leak onto the next. Look at each person again before
  you describe them.
- Judge height against the other people and the doorways in this frame. Never estimate
  centimetres. Give an age band, never a single age.
- A uniform is workwear that marks a role: a security or police shirt, scrubs, a cleaning
  or maintenance overall, a delivery rider's jacket, a hotel or shop uniform, a school
  uniform. Say "none" for ordinary clothes, and office clothes are ordinary clothes: a
  suit, a blazer, a white shirt, a tie, a lanyard and a staff card are what people wear
  to work in any building, and none of them make a uniform. A visible badge is reported
  in `uniform.id_badge`, which is a separate question, so never let a badge or a lanyard
  on its own decide this field.
  Pick "corporate" only on real evidence: a company name or logo printed on the garment,
  or several people wearing the same distinctive coloured garment. Without that it is
  "none". If you cannot see enough of the person to tell, "unknown".
  Copy any wording actually printed on the uniform into `uniform.text` exactly as it
  reads, and leave it "" when there is none. Do not translate it and do not invent it.
- `group` says who arrived together. Give everyone a group id. People walking abreast at
  the same pace, talking, or clearly waiting for each other share one id. Someone on
  their own gets an id nobody else shares. Two strangers who happen to pass at the same
  moment are not a group: being near each other is not enough. When you genuinely cannot
  tell, give that person their own id rather than inventing company for them.
- Report the facial expression you can actually see, as `emotion`. Choose the label from
  what the face is doing, not from the mood of the scene:
    happy    a smile, the corners of the mouth raised
    sad      mouth corners down, inner brows raised
    angry    brows drawn together and down, face tense
    fear     eyes wide open, brows raised and pulled together
    surprise eyes wide, mouth open, brows raised in an arch
    disgust  nose wrinkled, upper lip raised
    neutral  no expression standing out
  `valence` puts that on a 1 to 5 scale: 1 is clearly unhappy, angry or upset, 3 is the
  ordinary neutral face most people wear walking through a door, 5 is a broad smile or
  laughter. Most people in a lobby are a 3, and answering 3 all day is correct if that is
  what you see.
  If you cannot see the face, because they are turned away, masked, too far or in shadow,
  the label is "unknown" and the valence is 0. That is different from neutral: neutral
  means you looked at a face and it was calm. Never soften a face you cannot see into a 3.
  A resting face is not an angry face: do not read mild displeasure into a tired
  commuter, and keep the confidence low whenever the face is small or blurred. fear and
  disgust are rare at a building entrance: pick either one only on the features above,
  never because someone merely looks unfriendly.
- `mobility` is whether a person is moving on their own legs or is being carried along
  by something. walking: on their own feet, standing counts, and a cane, a crutch or a
  walking frame still counts as walking. stroller: a small child sitting in a pushchair
  or pram. wheelchair: sitting in a wheelchair, at any age. carried: a baby or small
  child held in someone's arms, in a sling or in a baby carrier. other: on a bicycle, a
  scooter or anything else with wheels under them.
  The person pushing is walking and the person being pushed is a separate row: a woman
  pushing a pram with a baby in it is two people, one walking and one stroller, never
  one row for the pair. They usually share the same `group`.
  When someone's lower body is hidden behind a counter, a desk or another person,
  answer unknown. Do not answer walking just because you cannot see anything underneath
  them: that is the one wrong answer that looks right.
- skin_tone is the skin tone visible in this image under this lighting. It is a
  descriptive attribute like shirt colour. Do not infer ethnicity, religion or
  occupation, and do not try to identify or name anyone.
- Confidence from 0 to 1, honestly. Someone sharp and close deserves 0.9. Someone small
  and half hidden deserves 0.3.

Write `description` in Thai, one short paragraph a Thai security officer would read
aloud on the radio. Every other field stays in English exactly as listed.

Reply with JSON only. No explanation, no markdown fence."""

USER_TEMPLATE = """Frame {w}x{h} from camera {cam}. Who is in it?

List people nearest the camera first, at most {cap}. Reply with exactly this shape:
{{"people":[{{
 "ref":"P1",
 "direction":"in","direction_confidence":0.9,
 "where":"centre foreground, walking towards the camera, in front of the other two",
 "group":"G1",
 "gender":"female","gender_confidence":0.9,
 "age_range":"40-49","age_range_confidence":0.6,{nat_shape}
 "mobility":"walking","mobility_confidence":0.8,
 "emotion":{{"label":"happy","valence":4,"confidence":0.5}},
 "appearance":{{
   "skin_tone":"medium","build":"average","height":"average",
   "hair_length":"short","hair_color":"black",
   "glasses":"yes","face_mask":"no","facial_hair":"no","headwear":"no",
   "top":{{"type":"shirt","color":"black","secondary_color":"unknown","pattern":"plain"}},
   "top_sleeve":"long",
   "outer":{{"type":"none","color":"unknown","secondary_color":"unknown","pattern":"unknown"}},
   "bottom":{{"type":"trousers","color":"black","secondary_color":"unknown","pattern":"plain"}},
   "footwear":{{"type":"leather_shoes","color":"white","secondary_color":"unknown","pattern":"plain"}},
   "uniform":{{"kind":"none","color":"unknown","id_badge":"no","text":""}},
   "carrying":["phone"],
   "distinctive":"รองเท้าหนังสีขาว"}},
 "appearance_confidence":0.7,
 "description":"หญิง อายุราว 40 ปี ผิวสองสี ผมสั้นสีดำ ใส่แว่นตา เสื้อแขนยาวสีดำ กางเกงสีดำ รองเท้าหนังสีขาว ถือโทรศัพท์",
 "reason":""}}]}}

`ref` numbers the people in this frame only: P1, P2, P3. It means nothing outside this
answer, so do not try to recognise anyone or reuse a number from another frame.

`group` is a second label on the same footing: G1, G2, G3, again meaningful only inside
this answer. Everyone gets one. Two people who came in together carry the same `group`.
Do not count the group members yourself and do not describe the group in words: just
label them, the same label twice is what says they are together.

`where` must let a person looking at the same picture point at the right one: position in
the frame, what they are doing, and how they sit relative to the others. Do not give
pixel coordinates, they are not what you are good at.

If there is nobody in the frame, reply {{"people":[]}}. That is a real answer, not a
failure, and it is the right one for an empty lobby.

{allowed}

`reason` is for when something blocks you on that person: back turned, motion blur, too
dark, too far, mostly out of frame. Leave it "" when nothing is wrong. Short."""


def build(w: int, h: int, camera_id: str = "unknown", cap: int = 12,
          nationality: bool = False):
    from .prompt_p1 import NATIONALITY_SHAPE
    return (system_block(SYSTEM, nationality),
            USER_TEMPLATE.format(w=w, h=h, cam=camera_id, cap=cap,
                                 allowed=allowed_block(nationality),
                                 nat_shape=NATIONALITY_SHAPE if nationality else ""))
