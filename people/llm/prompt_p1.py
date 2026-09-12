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
4. **สิ่งที่คำนวณเองได้ ห้ามถามโมเดล** `age_group` มาจาก `age_range` ในโค้ด
   ถามสองครั้งเรื่องเดียวกันแล้ววันหนึ่งจะได้คำตอบที่ขัดกันเอง แล้วปลายทาง
   ไม่มีทางรู้ว่าเชื่อช่องไหน · เรื่องเดียวกับ `group.size` ที่เรานับจาก ref เอง

🔴 ขอบเขตที่เขียนไว้ในคำสั่งโดยตั้งใจ
บริการนี้ **บรรยายรูปพรรณ ไม่ระบุตัวบุคคล** ไม่เดาชื่อ ไม่เดาเชื้อชาติ
ไม่เดาอาชีพจากหน้าตา ไม่จับคู่กับใคร · `skin_tone` คือสีผิวที่ปรากฏในภาพใบนั้น
ภายใต้แสงนั้น เป็นค่าสำหรับใช้ตามตัวเหมือนสีเสื้อ และต้องอ่านคู่กับ confidence เสมอ
`uniform` คือ **ชุดที่ใส่** ไม่ใช่อาชีพ ปลายทางเป็นคนแปลเอง

🔴 สัญชาติ (PEOPLE_ALLOW_NATIONALITY) **ค่าเริ่มต้นคือไม่ถาม** และตอนไม่ถาม
บล็อกคำถามนี้จะไม่อยู่ใน prompt เลย ไม่ใช่ใส่แล้วทิ้งคำตอบ · จ่าย token
ให้โมเดลคิดแล้วโยนทิ้ง คือการจ่ายเงินซื้อความเสี่ยงเปล่าๆ
เหตุผลที่ปิดไว้อยู่ใน `schemas.Nationality` ยาวกว่านี้ อ่านที่นั่น
"""
from __future__ import annotations

PROMPT_VERSION = "p2"
"""p1 -> p2 (2026-09-12) · เพิ่ม uniform, emotion, กฎเรื่อง overlay ที่ฝังมาในภาพ
(group ไม่มีในเส้นนี้ ดู schemas.GroupType)

🔴 เลขนี้ถูกเก็บลง DB ทุกแถว · ขยับทุกครั้งที่ prompt เปลี่ยน ไม่งั้นเวลาไล่ย้อนหลัง
ว่า "ทำไมผลเดือนที่แล้วกับเดือนนี้ไม่เหมือนกัน" จะแยกไม่ออกว่าเพราะ prompt หรือเพราะภาพ
"""

SYSTEM = """You describe one person from a CCTV still, for a building entrance in Thailand.

The camera system has already detected and tracked this person. The image you get is
that person. Your only job is to report what is visible about them.

Report which way they are facing, apparent gender, an age band, their facial expression,
whether they are in a uniform, and their appearance in detail (face, skin tone, hair,
height, build, clothing and its colours, footwear, what they carry).

How to be useful here:
- The camera looks at people as they come in. Someone facing you is coming in ("in").
  Someone whose back you see is going out ("out"). Someone side-on, or too obscured to
  tell, is "unknown", and that is a real answer: guessing here corrupts a head count.
  Report the way they face in this image. You cannot see motion in a still, so do not
  pretend to know where they are actually heading.
- Report only what you can actually see in this image. If the person's back is to the
  camera, you cannot see their face: say unknown for the face fields. If the frame cuts
  off at the waist, footwear is unknown. Unknown is a correct answer and costs nothing.
- Never guess to fill a field. A wrong shirt colour sends staff after the wrong person,
  which is worse than an empty field.
- This crop may have a coloured rectangle, an ID label or a timestamp drawn on top of it
  by the tracking system. Those are annotations printed over the picture, not clothing
  and not objects the person carries. Describe the person underneath them and ignore
  them completely.
- Height only means something against a reference. A tight crop of one person usually
  has nothing to compare with, so height is normally "unknown" here. Answer short or
  tall only when a doorway, a counter, a handrail or another person is visible in the
  crop to judge against. Never estimate centimetres.
- Give an age band, never a single age. A CCTV crop cannot support a single number.
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
- skin_tone is the skin tone visible in this image under this lighting. It is a
  descriptive attribute like shirt colour. Do not infer ethnicity, religion or
  occupation, and do not try to identify or name the person.
- Each headline answer carries its own confidence from 0 to 1, because they are rarely
  equally clear. A sharp frontal daylight crop deserves 0.9. A blurred figure from
  behind at the far end of a lobby deserves 0.2.

Write the `description` field in Thai, as one short paragraph a Thai security officer
would read aloud on the radio. Every other field stays in English exactly as listed.

Reply with JSON only. No explanation, no markdown fence."""

# ค่าที่ยอมรับต้องอยู่ใน prompt ด้วย ไม่ใช่แค่ในโค้ดฝั่งเรา
# ตัวไหนที่โมเดลไม่รู้ว่ามีให้เลือก มันจะประดิษฐ์คำใหม่มาเอง แล้วเราแปลงเป็น
# unknown ทิ้งหมด ซึ่งคือการจ่ายเงินให้โมเดลคิดแล้วโยนคำตอบทิ้ง
USER_TEMPLATE = """One person, object id {oid}, camera {cam}, crop {w}x{h}.

Reply with exactly this shape:
{{"direction":"in","direction_confidence":0.9,
 "gender":"male","gender_confidence":0.8,
 "age_range":"30-39","age_range_confidence":0.5,{nat_shape}
 "emotion":{{"label":"neutral","valence":3,"confidence":0.6}},
 "appearance":{{
   "skin_tone":"medium","build":"average","height":"unknown",
   "hair_length":"short","hair_color":"black",
   "glasses":"no","face_mask":"no","facial_hair":"no","headwear":"no",
   "top":{{"type":"shirt","color":"white","secondary_color":"unknown","pattern":"plain"}},
   "top_sleeve":"long",
   "outer":{{"type":"suit_jacket","color":"black","secondary_color":"unknown","pattern":"plain"}},
   "bottom":{{"type":"trousers","color":"black","secondary_color":"unknown","pattern":"plain"}},
   "footwear":{{"type":"leather_shoes","color":"black","secondary_color":"unknown","pattern":"plain"}},
   "uniform":{{"kind":"none","color":"unknown","id_badge":"yes","text":""}},
   "carrying":["lanyard"],
   "distinctive":"สายคล้องคอสีเขียว"}},
 "appearance_confidence":0.7,
 "description":"ชายไทย อายุราว 30 ปี ผิวสองสี ผมสั้นสีดำ ใส่เสื้อเชิ้ตแขนยาวสีขาว กางเกงสแลคสีดำ รองเท้าหนังสีดำ มีสายคล้องคอสีเขียว",
 "reason":""}}

{allowed}

`reason` is for when something blocks you: back turned, motion blur, too dark, too far,
person mostly out of frame. Leave it "" when nothing is wrong. Thai or English, short."""


# 🔴 ชุดค่าที่ยอมรับ อยู่ที่เดียว · prompt_f1 ใช้ก้อนนี้ร่วมกัน
#
# เคยจะก๊อปไปวางในไฟล์ที่สอง แล้วนึกได้ว่านั่นคือบั๊กที่เพิ่งเจอไปเมื่อเช้า:
# `laptop` หายเพราะชุดค่าในโค้ดกับที่โมเดลรู้ ไม่ตรงกัน · สองก้อนที่ต้องตรงกัน
# แต่แก้คนละที่ จะไม่ตรงกันภายในหนึ่งเดือน ไม่ว่าจะเขียนคอมเมนต์เตือนไว้ดีแค่ไหน
ALLOWED = """Allowed values, use these exact strings and nothing else:
direction: in, out, unknown
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
uniform.kind: security, police, military, medical, cleaning, maintenance, construction, delivery, retail_staff, hospitality, school, corporate, other, none, unknown
emotion.label: neutral, happy, sad, angry, surprise, fear, disgust, unknown
emotion.valence: whole number 1 to 5, or 0 when the label is unknown
uniform.id_badge: yes, no, unknown
uniform.text: free text, at most 40 characters, copied off the uniform exactly, "" if none
any colour field: black, white, grey, red, orange, yellow, green, blue, navy, purple, pink, brown, beige, cream, gold, silver, multicolour, unknown
any pattern field: plain, striped, checked, printed, logo, unknown
carrying: backpack, shoulder_bag, handbag, tote, shopping_bag, luggage, phone, laptop, tablet, document, umbrella, lanyard, badge, box, other

`top` is the garment against the body. `outer` is anything worn over it, a blazer, jacket
or coat. Someone in a dark blazer over a white shirt is remembered as the person in the
dark blazer, so both fields matter. Use "none" for outer when they wear nothing over."""


# ---------------------------------------------------------------- nationality
# 🔴 บล็อกนี้เข้า prompt เฉพาะตอน PEOPLE_ALLOW_NATIONALITY=true เท่านั้น
# ทั้ง p1 และ f1 ใช้ก้อนนี้ร่วมกัน เหมือน ALLOWED · ห้ามก๊อป
#
# ถ้อยคำเลือกมาเพื่อ **กดความมั่นใจลง** ไม่ใช่เพื่อให้ได้คำตอบสวยๆ
# โมเดลปกติจะตอบทุกใบด้วยความมั่นใจเท่ากันหมด ซึ่งเป็นสิ่งที่ทำให้ค่านี้อันตราย
NATIONALITY_RULE = """
- One extra field, `nationality`, is requested and it needs unusual care. You cannot see
  a passport, so you are not reporting nationality: you are reporting which broad region
  this person's appearance and dress most resemble, which is a weaker and often wrong
  signal. Thai, Lao, Burmese, Cambodian and Malaysian people are not distinguishable
  from a CCTV still, so all of them read as Thai here if they read as anything. Answer
  "unknown" whenever the face is unclear, turned away, masked or far, and keep the
  confidence low unless something concrete supports you. Never let this field change any
  other answer."""

NATIONALITY_SHAPE = '\n "nationality":"thai","nationality_confidence":0.4,'

NATIONALITY_ALLOWED = "\nnationality: thai, asian_other, western, other, unknown"


def allowed_block(nationality: bool = False) -> str:
    return ALLOWED + (NATIONALITY_ALLOWED if nationality else "")


def system_block(base: str, nationality: bool = False) -> str:
    """แทรกกฎเรื่องสัญชาติก่อนย่อหน้าสุดท้าย ไม่ใช่ต่อท้ายสุด

    ต่อท้ายหลัง "Reply with JSON only" แล้วโมเดลบางตัวจะถือว่าบรรทัดนั้นจบคำสั่ง
    แล้วอ่านของที่ตามมาเป็นข้อมูล ไม่ใช่คำสั่ง
    """
    if not nationality:
        return base
    marker = "\n\nWrite the `description` field"
    if marker in base:
        return base.replace(marker, NATIONALITY_RULE + marker, 1)
    marker = "\n\nWrite `description`"
    return base.replace(marker, NATIONALITY_RULE + marker, 1)


def build(object_id: str, w: int, h: int, camera_id: str = "unknown",
          nationality: bool = False):
    return (system_block(SYSTEM, nationality),
            USER_TEMPLATE.format(oid=object_id, w=w, h=h, cam=camera_id,
                                 allowed=allowed_block(nationality),
                                 nat_shape=NATIONALITY_SHAPE if nationality else ""))
