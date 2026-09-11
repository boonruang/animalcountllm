"""สัญญาข้อมูลของงาน smart people counting อยู่ไฟล์เดียว

คนละไฟล์ คนละแพ็กเกจกับ `app/schemas.py` โดยตั้งใจ · งานช้างกับงานคนใช้คำว่า
"confidence" เหมือนกันแต่คนละความหมาย และ response ของงานช้าง **ห้ามขยับ**
เพราะ confirm กับทีมคุณสุชาติไปแล้ว การ import ข้ามกันคือทางที่ทำให้วันหนึ่ง
แก้ฝั่งนี้แล้วฝั่งโน้นพัง เลยไม่ทำตั้งแต่แรก

🔴 ทุกฟิลด์ที่ "ตีความ" ต้องมีค่า unknown ได้เสมอ
งานนี้คือบรรยายรูปพรรณคนเพื่อให้เจ้าหน้าที่ตามหาต่อ · "เสื้อขาว" ที่เดาผิด
แพงกว่า "ไม่รู้" มาก เพราะมันส่งคนไปตามผิดคน ไม่ใช่แค่ข้อมูลขาด
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------- enums
# ชุดค่าทั้งหมดปิดท้ายด้วย unknown ทุกตัว · โมเดลตอบนอกชุด = แปลงเป็น unknown
# ไม่ใช่โยน 500 (ดู people/llm/client.py:normalize)

Gender = Literal["male", "female", "unknown"]

# ช่วงอายุ ไม่ใช่ตัวเลขเดี่ยว ตามที่ Toy สั่ง · ภาพ CCTV ที่ 80-200 px
# บอกอายุเป็นปีไม่ได้จริง การให้ตัวเลขเดี่ยวคือการแกล้งทำเป็นแม่นกว่าที่เป็น
AgeRange = Literal["0-12", "13-19", "20-29", "30-39", "40-49", "50-59", "60+",
                   "unknown"]

# สีผิวที่ "มองเห็น" ในภาพนั้น ไม่ใช่เชื้อชาติ · แสงในเฟรมเปลี่ยนค่านี้ได้ง่าย
# จึงต้องอ่านคู่กับ confidence เสมอ
SkinTone = Literal["light", "medium", "tan", "dark", "unknown"]

Build = Literal["slim", "average", "heavy", "unknown"]

# ส่วนสูงจากกล้องมุมกดบอกเป็นเซนติเมตรไม่ได้ ต้องเทียบกับคนอื่นในเฟรมเท่านั้น
Height = Literal["short", "average", "tall", "unknown"]

HairLength = Literal["bald", "very_short", "short", "medium", "long",
                     "tied_back", "covered", "unknown"]
HairColor = Literal["black", "dark_brown", "brown", "blonde", "grey", "white",
                    "dyed_other", "unknown"]

YesNo = Literal["yes", "no", "unknown"]

# ชุดสีปิด ไม่ใช่ string อิสระ · ปลายทางต้องกรองได้ว่า "หาคนเสื้อแดง"
# ถ้าปล่อยอิสระจะได้ crimson/maroon/แดงเลือดหมู ปนกันจนกรองไม่ได้
Color = Literal["black", "white", "grey", "red", "orange", "yellow", "green",
                "blue", "navy", "purple", "pink", "brown", "beige", "cream",
                "gold", "silver", "multicolour", "unknown"]

Pattern = Literal["plain", "striped", "checked", "printed", "logo", "unknown"]

TopType = Literal["t_shirt", "shirt", "polo", "blouse", "jacket", "suit_jacket",
                  "coat", "hoodie", "sweater", "vest", "dress", "uniform",
                  "other", "unknown"]
BottomType = Literal["trousers", "jeans", "shorts", "skirt", "dress", "uniform",
                     "other", "unknown"]
FootwearType = Literal["sneakers", "leather_shoes", "boots", "sandals", "heels",
                       "slippers", "other", "unknown"]

# 🔴 laptop / tablet / document เพิ่ม 2026-09-11 หลังยิงภาพจริงใบแรก
# โมเดลเห็น "ถือแล็ปท็อปสีเทา" แล้วเขียนไว้ใน distinctive ถูกต้อง แต่ carrying กลับว่าง
# เพราะ laptop ไม่อยู่ในชุดค่าที่ยอมรับ เราเลยทิ้งข้อมูลที่โมเดลอ่านถูกแล้ว
# บทเรียน: ชุดค่าปิดที่แคบเกินจริง = ทิ้งของดีเงียบๆ ไม่มี error ให้เห็น
Carrying = Literal["backpack", "shoulder_bag", "handbag", "tote", "shopping_bag",
                   "luggage", "phone", "laptop", "tablet", "document", "umbrella",
                   "lanyard", "badge", "box", "other"]

Sleeve = Literal["sleeveless", "short", "long", "unknown"]

# 🔴 ทิศ เข้า/ออก · Toy สั่ง 2026-09-11
#
# มาจาก **คนคนนั้นหันหน้าไปทางไหนในเฟรมนี้เฟรมเดียว** เท่านั้น
#   in  = หันหน้าเข้าหากล้อง (เดินเข้าอาคาร)
#   out = เห็นด้านหลัง (เดินออก)
#   unknown = หันข้าง โดนบังครึ่งตัว หรือดูไม่ออก
#
# ⚠️ **หันหน้า ไม่เท่ากับ กำลังเดินไปทางนั้น** คนยืนคุยโทรศัพท์หันหน้าเข้ากล้อง
# ได้โดยไม่ได้เข้าไปไหน คนเดินออกแล้วหันกลับมามองข้างหลังก็ได้ ภาพนิ่งใบเดียว
# แยกสองอย่างนี้ไม่ได้ · ถ้าปลายทางต้องการ "เข้ากี่คน ออกกี่คน" ที่เชื่อถือได้จริง
# ต้องนับจาก track ที่ข้ามเส้นในวิดีโอ ซึ่งระบบของทีมคุณสุชาติทำอยู่แล้ว (เห็น
# "เข้า 11 ออก 5" บนหน้าจอเขาใน iplus_sample1) ค่าที่เราให้เป็นการอ่านจากท่าทาง
# ในภาพนิ่ง ใช้ประกอบได้ ใช้แทนกันไม่ได้
Direction = Literal["in", "out", "unknown"]

PersonStatus = Literal["ok", "degraded", "error"]
RequestStatus = Literal["ok", "partial", "degraded", "error"]


# ---------------------------------------------------------------- request
class ObjectIn(BaseModel):
    """คนหนึ่งคนที่ปลายทางชี้มา

    ระบบของทีมคุณสุชาติ detect + track เองอยู่แล้ว (เห็นจาก iplus_sample1:
    มี ID:011 / ID:021 / ID:022 พร้อม crop ใบหน้าแปะข้างจอ) เราไม่หาคนซ้ำ
    เราอ่านเฉพาะคนที่เขาชี้มาและตอบกลับผูกกับ `id` เดิมของเขา

    รับได้สองแบบ เพราะภาพตัวอย่างสองใบมาคนละแบบ:
      1. `image_base64` = ภาพที่ครอปมาแล้ว (แบบ iplus_sample1)
      2. `bbox` ชี้เข้าไปใน `frame_base64` ระดับ request (แบบ iplus_sample2
         ที่เป็นเฟรมเต็ม 1920x1080 มีกรอบสามกรอบ) แล้วเราครอปให้เอง
    ส่งมาทั้งคู่ได้ `image_base64` ชนะ เพราะเป็นภาพที่ปลายทางเลือกมาเองแล้ว
    """

    id: str = Field(..., min_length=1, max_length=64)
    image_base64: Optional[str] = None
    bbox: Optional[List[int]] = Field(None, min_length=4, max_length=4)
    """[x, y, w, h] ในพิกัดของ frame_base64 · หน่วยพิกเซล ไม่ใช่สัดส่วน"""
    note: Optional[str] = None


class PersonsIn(BaseModel):
    camera_id: str = Field(..., min_length=1, max_length=64)
    objects: List[ObjectIn] = Field(..., min_length=1, max_length=16)
    """🔴 เพดาน 16 คนต่อ request · หนึ่งคน = หนึ่งครั้งที่ยิงโมเดล
    ไม่มีเพดาน = ปลายทางส่งมา 200 คนแล้วเราจ่ายค่าโมเดล 200 ครั้งในนัดเดียว
    โดยไม่มีใครเห็นจนบิลมา · เกินเพดานตอบ 400 พร้อมบอกว่าเกินเท่าไร"""

    frame_base64: Optional[str] = None
    """เฟรมเต็ม ใช้คู่กับ bbox · ไม่ส่งมาก็ได้ถ้าทุก object มีภาพของตัวเอง"""

    ts: Optional[str] = None  # ISO 8601; ไม่ส่งมา = ใช้เวลาที่เซิร์ฟเวอร์รับ
    note: Optional[str] = None
    client_request_id: Optional[str] = Field(None, max_length=128)
    """เลขอ้างอิงของ**ฝั่งปลายทางเอง** เราไม่ตีความ ไม่ตรวจ ไม่บังคับให้ไม่ซ้ำ
    ส่งมาอย่างไรตอบกลับอย่างนั้นเป๊ะ และใช้ยิง GET กลับมาดูทีหลังได้

    มีไว้เพราะ `request_id` ที่เราออกให้ ปลายทางจะรู้ก็ต่อเมื่อได้ response แล้ว
    **ถ้า timeout หรือเน็ตหลุดกลางทาง เขาจับคู่อะไรไม่ได้เลย** ตัวนี้เขารู้ตั้งแต่
    ก่อนยิง เลยใช้ตามของที่ส่งไปแล้วไม่รู้ผลได้"""


class PeopleFrameIn(BaseModel):
    """เฟรมเดียว ไม่มีใครถูกชี้ · ให้โมเดลหาคนในเฟรมเอง

    🔴 รูป body **เหมือน `FrameIn` ของฝั่งช้างเป๊ะทุกฟิลด์** ตามที่ Toy สั่ง 2026-09-11
    ปลายทางที่ยิงฝั่งช้างเป็นอยู่แล้ว ยิงเส้นนี้ได้โดยไม่ต้องเขียน serializer ใหม่
    เปลี่ยนแค่ path · ห้ามเพิ่มฟิลด์ลงใน model นี้โดยไม่ดูฝั่งช้างก่อน
    ถ้าฝั่งโน้นไม่มี ที่นี่ก็ไม่ควรมี ไม่งั้นคำว่า "เหมือนกัน" จะค่อยๆ ไม่จริง
    """

    camera_id: str = Field(..., min_length=1, max_length=64)
    image_base64: str
    ts: Optional[str] = None  # ISO 8601; ไม่ส่งมา = ใช้เวลาที่เซิร์ฟเวอร์รับ
    note: Optional[str] = None
    client_request_id: Optional[str] = Field(None, max_length=128)
    """เลขอ้างอิงของ**ฝั่งปลายทางเอง** เราไม่ตีความ ไม่ตรวจ ไม่บังคับให้ไม่ซ้ำ
    ส่งมาอย่างไรตอบกลับอย่างนั้นเป๊ะ และใช้ยิง GET กลับมาดูทีหลังได้

    มีไว้เพราะ `request_id` ที่เราออกให้ ปลายทางจะรู้ก็ต่อเมื่อได้ response แล้ว
    **ถ้า timeout หรือเน็ตหลุดกลางทาง เขาจับคู่อะไรไม่ได้เลย** ตัวนี้เขารู้ตั้งแต่
    ก่อนยิง เลยใช้ตามของที่ส่งไปแล้วไม่รู้ผลได้"""


class PersonTruthIn(BaseModel):
    """ค่าที่คนตรวจแล้ว · ไม่เก็บตั้งแต่วันแรก = วัดความแม่นย้อนหลังไม่ได้ตลอดกาล

    บทเรียนจากฝั่งช้าง: ภาพแรกจากไซต์จริงตอบว่าช้าง 6 ตัว แล้วไม่มีใครยิง /truth
    กลับมา ทุกวันนี้ยังไม่รู้ว่า 6 ถูกไหม และย้อนไปเก็บไม่ได้แล้ว
    """

    gender: Optional[Gender] = None
    age_range: Optional[AgeRange] = None
    comment: Optional[str] = None
    reviewer: Optional[str] = None


# ---------------------------------------------------------------- appearance
class Garment(BaseModel):
    type: str = "unknown"
    color: Color = "unknown"
    secondary_color: Color = "unknown"
    """สีที่สองของชิ้นเดียวกัน เช่น เสื้อลายทางขาวดำ · ไม่มีก็ unknown"""
    pattern: Pattern = "unknown"


class Appearance(BaseModel):
    """รูปพรรณสัณฐานแบบแตกฟิลด์ · ทุกช่องเป็น unknown ได้และค่าเริ่มต้นคือ unknown

    แตกเป็นฟิลด์แทนที่จะเป็นข้อความก้อนเดียว เพราะปลายทางต้องกรองได้
    ("ผู้ชาย เสื้อดำ กางเกงยีนส์") ซึ่งข้อความก้อนเดียวทำไม่ได้
    ส่วนข้อความบรรยายยาวยังมีให้ที่ `description` ระดับบน ไม่ได้ตัดทิ้ง
    """

    skin_tone: SkinTone = "unknown"
    build: Build = "unknown"
    height: Height = "unknown"
    hair_length: HairLength = "unknown"
    hair_color: HairColor = "unknown"
    glasses: YesNo = "unknown"
    face_mask: YesNo = "unknown"
    facial_hair: YesNo = "unknown"
    headwear: YesNo = "unknown"

    top: Garment = Field(default_factory=Garment)
    top_sleeve: Sleeve = "unknown"

    outer: Garment = Field(default_factory=Garment)
    """เสื้อคลุมตัวนอก แจ็กเก็ต/สูท/โค้ท · `type: "none"` = ไม่ได้ใส่

    🔴 เพิ่ม 2026-09-11 หลังยิงภาพจริงใบแรก · ในภาพนั้นทั้งสามคนใส่เสื้อคลุมทับ
    แล้ว `top` เก็บได้แค่ตัวเดียว โมเดลเลยตอบว่า "เสื้อเชิ้ตขาว" ซึ่งถูกแต่ไม่พอ
    คนที่เห็นจากไกลคือ "ผู้ชายสูทดำ" ไม่ใช่ "ผู้ชายเสื้อขาว"
    ข้อมูลนี้เคยอยู่แต่ในข้อความบรรยาย ซึ่งกรองด้วยโปรแกรมไม่ได้"""

    bottom: Garment = Field(default_factory=Garment)
    footwear: Garment = Field(default_factory=Garment)
    carrying: List[Carrying] = Field(default_factory=list)

    distinctive: str = Field("", max_length=200)
    """อะไรที่เด่นจนใช้ตามตัวได้จริง เช่น 'สายคล้องคอสีแดง' 'เป้สีส้ม'
    ว่างได้ · ห้ามยัดคำบรรยายทั่วไปลงช่องนี้ มันมีไว้ให้คนอ่านเร็วๆ แล้วชี้ตัวถูก"""


# ---------------------------------------------------------------- response
class ImageInfo(BaseModel):
    w: int
    h: int
    source: Literal["object_image", "frame_bbox", "full_frame"]
    """ภาพที่โมเดลเห็นมาจากไหน · ใช้ไล่ย้อนตอนผลเพี้ยน ว่าเราครอปพลาดเองหรือเปล่า

    object_image = ปลายทางครอปมาให้แล้ว · frame_bbox = เราครอปจากเฟรมตาม bbox
    full_frame   = เฟรมเต็มทั้งใบ (เส้น /v1/frames) ไม่มีการครอป

    🔴 `full_frame` เพิ่ม 2026-09-11 · ตอนแรกเส้น /v1/frames คืน `object_image`
    เพราะตอนนั้นมีให้เลือกแค่สองค่า **ซึ่งเป็นการโกหกปลายทางเงียบๆ** ว่าภาพนี้
    ถูกครอปมาให้หนึ่งคนแล้ว ทั้งที่เป็นเฟรมเต็ม · ฐานข้อมูลเขียนว่า "frame" อีกค่าหนึ่ง
    สองที่ไม่ตรงกันด้วย · ค่าที่ไม่มีให้เลือก ห้ามหยิบค่าที่ใกล้ที่สุดมาใส่แทน
    ให้เพิ่มค่าใหม่ ไม่งั้นฟิลด์นั้นจะเชื่อไม่ได้ทั้งฟิลด์"""

    bbox_used: Optional[List[int]] = None
    """กรอบที่เราครอปจริง `[x1,y1,x2,y2]` หลังขยาย margin และหนีบขอบภาพแล้ว

    **ไม่ใช่ค่าเดียวกับ bbox ที่ส่งมา** และนั่นคือเหตุผลที่ต้องคืนกลับไป
    ปลายทางจะได้ยืนยันด้วยตาว่า "คนที่เราอ่าน" คือคนเดียวกับ "คนที่คุณชี้"
    ไม่ใช่คนที่ยืนติดกัน · เป็น null เมื่อปลายทางส่ง crop มาเอง"""


class ModelInfo(BaseModel):
    provider: str
    name: str
    prompt_version: str
    finish_reason: Optional[str] = None
    completion_tokens: Optional[int] = None


class PersonOut(BaseModel):
    """คำตอบต่อคนหนึ่งคน · ผูกกลับด้วย `id` ที่ปลายทางส่งมา ห้ามเปลี่ยนค่า

    สามอย่างที่ Toy สั่งไว้: เพศ · ช่วงอายุ · การแต่งกาย
    ทั้งสามอย่างมาพร้อม confidence ของตัวเอง เพราะในภาพจริงมันไม่ได้ชัดเท่ากัน
    (เห็นทรงผมชัดแต่เดาอายุไม่ออก เป็นเรื่องปกติของมุมกล้องกดลงมา)
    """

    id: str
    status: PersonStatus

    where: str = ""
    """คนคนนี้อยู่ตรงไหนในเฟรม เป็นข้อความให้คนอ่านแล้วชี้ตัวถูก

    มีค่าเฉพาะเส้น `/v1/frames` ที่โมเดลหาคนเอง · เส้น `/v1/persons` เป็น "" เสมอ
    เพราะปลายทางเป็นคนชี้มาเองอยู่แล้ว ไม่ต้องมีใครมาบอกว่าคนไหน

    🔴 **ไม่ใช่ bbox และจะไม่มีวันเป็น bbox** วัดมาแล้วฝั่งช้าง โมเดลคืนพิกัด
    y=998 บนภาพสูง 628 px · ตำแหน่งที่โมเดลบอกเป็นคำพูดใช้ได้ ที่บอกเป็นตัวเลขใช้ไม่ได้
    """

    direction: Direction = "unknown"
    """เข้า/ออก จากทิศที่หันหน้าในเฟรมนี้ · อ่านคำเตือนที่ Direction ก่อนเอาไปนับยอด"""
    direction_confidence: float = Field(0.0, ge=0.0, le=1.0)

    gender: Gender = "unknown"
    gender_confidence: float = Field(0.0, ge=0.0, le=1.0)

    age_range: AgeRange = "unknown"
    age_range_confidence: float = Field(0.0, ge=0.0, le=1.0)

    appearance: Appearance = Field(default_factory=Appearance)
    appearance_confidence: float = Field(0.0, ge=0.0, le=1.0)

    description: str = ""
    """คำบรรยายภาษาไทยหนึ่งย่อหน้า สำหรับคนอ่าน ไม่ใช่สำหรับโปรแกรมกรอง"""

    overall_confidence: float = Field(0.0, ge=0.0, le=1.0)
    reason: str = ""
    """ทำไมถึงตอบเท่านี้ · โดยเฉพาะตอน degraded ปลายทางต้องแยกออกว่า
    'โมเดลล่ม' กับ 'ภาพเบลอจนดูไม่ออก' ซึ่งแก้คนละวิธี"""

    image: Optional[ImageInfo] = None

    model: Optional[ModelInfo] = None
    """🔴 เป็น null เสมอ เว้นแต่ตั้ง PEOPLE_EXPOSE_MODEL=true (Toy สั่ง 2026-09-11)

    ปลายทางไม่ต้องรู้ว่าเราใช้โมเดลอะไร ของเจ้าไหน prompt เวอร์ชันไหน
    **คีย์ยังอยู่ ไม่ได้หายไป** รูป response จะได้ไม่เปลี่ยนตามค่า env
    ซึ่งเป็นสิ่งที่ทำให้ปลายทาง parse พังแบบหาสาเหตุไม่เจอ
    ของจริงยังเก็บครบในฐานข้อมูลฝั่งเรา ไล่ดูย้อนหลังได้เหมือนเดิม"""

    timing_ms: float


class PersonsOut(BaseModel):
    request_id: str
    """เลขที่**เราออกให้** ไม่ซ้ำแน่นอน · ใช้ยิง GET /v1/persons/{request_id}"""

    ref: Optional[str] = None
    """เลขอ้างอิงของ**ฝั่งปลายทางเอง** คืนกลับเป๊ะตัวเดิม ไม่แก้ไขอะไรทั้งสิ้น

    🔴 **ตัวนี้คือของที่ปลายทางใช้ยืนยันว่าคำตอบนี้เป็นของตัวเอง** (Toy เคาะ 2026-09-11)

    ค่ามาจาก `client_request_id` ถ้าส่งมา ไม่งั้นเอา `note`
    เพราะทีมคุณสุชาติจะส่งเลขอ้างอิงมาในช่อง `note` ซึ่งมีอยู่แล้วใน spec ฝั่งช้าง
    เขาเลยไม่ต้องแก้ body ที่ส่งอยู่เลย

    **ช่องเดียวเท่านั้น** ไม่คืน `note` ซ้ำอีกช่อง · ค่าเดียวกันสองชื่อคือที่ที่
    คนอ่านเอกสารจะเดาผิดว่ามันต่างกันตรงไหน

    เราไม่การันตีว่าไม่ซ้ำ เพราะเลขนี้ไม่ใช่ของเรา ส่งซ้ำมาก็มีหลายผลตามนั้น"""

    camera_id: str
    received_at: str
    status: RequestStatus
    """ok = ทุกคนอ่านได้ · partial = บางคนไม่ได้ · degraded = ไม่ได้เลยสักคน

    🔴 ห้ามยุบเป็น ok/error สองค่า · ส่งมา 5 คน อ่านได้ 4 ไม่ใช่ทั้งสำเร็จ
    และไม่ใช่ทั้งล้มเหลว ปลายทางต้องรู้ว่าให้ retry เฉพาะคนไหน
    """

    reason: str = ""
    """ทำไมทั้ง request ถึงเป็น degraded · ว่างเมื่อ status เป็น ok

    ต้องมีช่องนี้เพราะเส้น `/v1/frames` ที่โมเดลล่ม จะไม่มีคนสักคนให้ใส่เหตุผลไว้ข้างใน
    แล้ว degraded ที่ไม่บอกเหตุผลคือสิ่งที่ปลายทางทำอะไรต่อไม่ได้เลย"""

    persons: List[PersonOut]
    summary: Dict[str, int]
    model: Optional[ModelInfo] = None
    """null เว้นแต่ PEOPLE_EXPOSE_MODEL=true · ดูเหตุผลที่ PersonOut.model"""

    timing_ms: Dict[str, float]
