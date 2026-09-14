"""แปลค่าใน JSON เป็นไทยตอนขาออก · **คีย์ยังเป็นอังกฤษ ค่าเป็นไทย** (Toy สั่ง 2026-09-12)

🔴 ทำไมแปลที่นี่ ไม่สั่งให้โมเดลตอบไทยตั้งแต่ต้น

สั่งโมเดลว่า "ตอบสีเป็นภาษาไทย" แล้วมันจะตอบ "ขาวนวล" "ครีมอมเทา" "ดำด้าน"
ซึ่งถูกต้องในภาษาไทยทุกคำ แต่ **กรองด้วยโปรแกรมไม่ได้เลย** และ `_pick` จะโยนทิ้ง
เป็น unknown หมด · ชุดค่าปิดมีไว้เพื่อให้ปลายทางกรองได้ ("หาคนเสื้อแดง")
ภาษาไทยที่โมเดลแต่งเองทำลายคุณสมบัตินั้นทันที

เลยแยกเป็นสองชั้นคนละหน้าที่:
  โมเดล + `normalize` + ฐานข้อมูล + prompt  =  **อังกฤษ ชุดค่าปิด**  (ของที่ใช้ทำงาน)
  ตอนจะตอบออกไป                              =  **ไทย**              (ของที่คนอ่าน)

ผลพลอยได้ที่สำคัญกว่าที่คิด:
- ฐานข้อมูลเดิมทั้งก้อนยังใช้ได้ ไม่ต้อง migrate ไม่ต้องแปลย้อนหลัง
- `emotion.label` ในฐานยังสะกดตาม FER2013 ซึ่งเป็นเหตุผลทั้งหมดที่เลือกสะกดแบบนั้น
  (เอาไปเทียบกับ FER ของทีมคุณสุชาติได้ตรงๆ) · แปลตั้งแต่ในฐาน = ทิ้งข้อดีนั้นไปเลย
- อยากได้อังกฤษคืนวันไหน แก้ที่เดียว ไม่ต้องไล่ทั้งระบบ

🔴 สิ่งที่ **ไม่แปล** และเหตุผล

`status` `image.source` `id` `ref` `group.ref` `request_id` `camera_id`
เป็น **ค่าโปรโตคอล ไม่ใช่คำบรรยายคน** ปลายทางเขียน `if status == "ok"` ไว้แล้ว
แปลเมื่อไหร่ = โค้ดเขาพังเงียบๆ ในวันที่ทุกอย่างดูปกติดี

`uniform.text` คือข้อความที่**ลอกมาจากชุดจริง** ("SECURITY") มันคือหลักฐาน
ไม่ใช่ค่าที่เราตีความ · แปลเมื่อไหร่ก็ไม่ใช่หลักฐานอีกต่อไป

`age_range` เป็นตัวเลข (`20-29`) · `emotion.valence` เป็นตัวเลข · ไม่ต้องแปล

`description` `where` `reason` `distinctive` เป็นภาษาไทยอยู่แล้วตั้งแต่ prompt

🔴 คีย์ที่ไม่มีตารางแปล จะถูกปล่อยผ่านเฉยๆ ไม่ใช่ถูกเดา
`test_ทุกค่าใน_schema_ต้องมีคำแปลไทย` ไล่จาก Literal ใน schemas.py เป็นตัวตั้ง
เพิ่มค่าใหม่ใน schema แล้วลืมเติมคำแปล = เทสต์ตกทันที ไม่ใช่หลุดเป็นอังกฤษ
ปนไทยไปให้ปลายทางเจอเอง
"""
from __future__ import annotations

from typing import Any, Dict

_UNKNOWN = "ไม่ทราบ"

# ใส่/ไม่ใส่ กับ มี/ไม่มี ใช้คนละที่กัน ตามที่ Toy ยกตัวอย่าง ("แว่นตา mask ใส่ ไม่ใส่")
_WEARS = {"yes": "ใส่", "no": "ไม่ใส่", "unknown": _UNKNOWN}
_HAS = {"yes": "มี", "no": "ไม่มี", "unknown": _UNKNOWN}

_COLOR = {"black": "ดำ", "white": "ขาว", "grey": "เทา", "red": "แดง",
          "orange": "ส้ม", "yellow": "เหลือง", "green": "เขียว", "blue": "ฟ้า",
          "navy": "น้ำเงิน", "purple": "ม่วง", "pink": "ชมพู", "brown": "น้ำตาล",
          "beige": "เบจ", "cream": "ครีม", "gold": "ทอง", "silver": "เงิน",
          "multicolour": "หลายสี", "unknown": _UNKNOWN}

_PATTERN = {"plain": "สีพื้น", "striped": "ลายทาง", "checked": "ลายตาราง",
            "printed": "ลายพิมพ์", "logo": "มีโลโก้", "unknown": _UNKNOWN}

_TOP = {"t_shirt": "เสื้อยืด", "shirt": "เสื้อเชิ้ต", "polo": "เสื้อโปโล",
        "blouse": "เสื้อเบลาส์", "jacket": "แจ็คเก็ต", "suit_jacket": "เสื้อสูท",
        "coat": "เสื้อโค้ท", "hoodie": "เสื้อฮู้ด", "sweater": "เสื้อไหมพรม",
        "vest": "เสื้อกั๊ก", "dress": "ชุดเดรส", "uniform": "ชุดเครื่องแบบ",
        "other": "อื่นๆ", "unknown": _UNKNOWN}

_OUTER = {"jacket": "แจ็คเก็ต", "suit_jacket": "เสื้อสูท", "blazer": "เบลเซอร์",
          "coat": "เสื้อโค้ท", "cardigan": "คาร์ดิแกน", "hoodie": "เสื้อฮู้ด",
          "vest": "เสื้อกั๊ก", "uniform": "ชุดเครื่องแบบ",
          "none": "ไม่ได้ใส่ทับ", "unknown": _UNKNOWN}

_BOTTOM = {"trousers": "กางเกงขายาว", "jeans": "กางเกงยีนส์",
           "shorts": "กางเกงขาสั้น", "skirt": "กระโปรง", "dress": "ชุดเดรส",
           "uniform": "ชุดเครื่องแบบ", "other": "อื่นๆ", "unknown": _UNKNOWN}

_FOOTWEAR = {"sneakers": "รองเท้าผ้าใบ", "leather_shoes": "รองเท้าหนัง",
             "boots": "รองเท้าบูท", "sandals": "รองเท้ารัดส้น",
             "heels": "รองเท้าส้นสูง", "slippers": "รองเท้าแตะ",
             "other": "อื่นๆ", "unknown": _UNKNOWN}

# 🔴 คีย์ = ชื่อฟิลด์ · `parent.field` ชนะ `field` เสมอ
# ต้องมี parent เพราะ `type` เป็นชื่อคีย์ของทั้ง top/outer/bottom/footwear
# แต่ชุดค่าคนละชุด · และ yes/no ของ glasses ("ใส่") กับ facial_hair ("มี") คนละคำ
FIELD: Dict[str, Dict[str, str]] = {
    "direction": {"in": "เข้า", "out": "ออก", "unknown": _UNKNOWN},
    "gender": {"male": "ชาย", "female": "หญิง", "unknown": _UNKNOWN},
    "age_group": {"child": "เด็ก", "teen": "วัยรุ่น", "adult": "ผู้ใหญ่",
                  "senior": "ผู้สูงอายุ", "unknown": _UNKNOWN},
    # 🔴 "รถเข็นเด็ก" กับ "รถเข็นวีลแชร์" ต้องเป็นคนละคำในภาษาไทย
    # คำว่า "รถเข็น" เฉยๆ ใช้เรียกได้ทั้งสองอย่าง ซึ่งเป็นความกำกวมที่ปลายทาง
    # แก้เองไม่ได้เลยเมื่อเห็นแต่ค่าในช่อง
    "mobility": {"walking": "เดินเอง", "stroller": "อยู่ในรถเข็นเด็ก",
                 "wheelchair": "นั่งรถเข็นวีลแชร์", "carried": "ถูกอุ้ม",
                 "other": "อื่นๆ", "unknown": _UNKNOWN},
    "nationality": {"thai": "ไทย", "asian_other": "เอเชียอื่น",
                    "western": "ตะวันตก", "other": "อื่นๆ", "unknown": _UNKNOWN},
    "skin_tone": {"light": "ผิวขาว", "medium": "ผิวสองสี", "tan": "ผิวแทน",
                  "dark": "ผิวเข้ม", "unknown": _UNKNOWN},
    "build": {"slim": "ผอม", "average": "ปานกลาง", "heavy": "ท้วม",
              "unknown": _UNKNOWN},
    "height": {"short": "เตี้ย", "average": "ปานกลาง", "tall": "สูง",
               "unknown": _UNKNOWN},
    "hair_length": {"bald": "ศีรษะล้าน", "very_short": "สั้นเกรียน",
                    "short": "สั้น", "medium": "ประบ่า", "long": "ยาว",
                    "tied_back": "รวบผม", "covered": "คลุมผม",
                    "unknown": _UNKNOWN},
    "hair_color": {"black": "ดำ", "dark_brown": "น้ำตาลเข้ม", "brown": "น้ำตาล",
                   "blonde": "บลอนด์", "grey": "ดอกเลา", "white": "ขาว",
                   "dyed_other": "ย้อมสีอื่น", "unknown": _UNKNOWN},
    "glasses": _WEARS,
    "face_mask": _WEARS,
    "headwear": _WEARS,
    "facial_hair": _HAS,
    "top_sleeve": {"sleeveless": "แขนกุด", "short": "แขนสั้น", "long": "แขนยาว",
                   "unknown": _UNKNOWN},
    "color": _COLOR,
    "secondary_color": _COLOR,
    "pattern": _PATTERN,
    "top.type": _TOP,
    "outer.type": _OUTER,
    "bottom.type": _BOTTOM,
    "footwear.type": _FOOTWEAR,
    "carrying": {"backpack": "เป้สะพายหลัง", "shoulder_bag": "กระเป๋าสะพายข้าง",
                 "handbag": "กระเป๋าถือ", "tote": "ถุงผ้า",
                 "shopping_bag": "ถุงช้อปปิ้ง", "luggage": "กระเป๋าเดินทาง",
                 "phone": "โทรศัพท์", "laptop": "แล็ปท็อป", "tablet": "แท็บเล็ต",
                 "document": "เอกสาร", "umbrella": "ร่ม",
                 "lanyard": "สายคล้องคอ", "badge": "ป้ายชื่อ", "box": "กล่อง",
                 "other": "อื่นๆ"},
    "uniform.kind": {"security": "รปภ.", "police": "ตำรวจ", "military": "ทหาร",
                     "medical": "บุคลากรทางการแพทย์", "cleaning": "แม่บ้าน",
                     "maintenance": "ช่างซ่อมบำรุง", "construction": "ช่างก่อสร้าง",
                     "delivery": "ไรเดอร์/ขนส่ง", "retail_staff": "พนักงานร้าน",
                     "hospitality": "พนักงานโรงแรม/ร้านอาหาร",
                     "school": "นักเรียน/นักศึกษา", "corporate": "ชุดพนักงานบริษัท",
                     "other": "เครื่องแบบอื่น", "none": "ไม่ได้ใส่เครื่องแบบ",
                     "unknown": _UNKNOWN},
    "uniform.id_badge": _HAS,
    "group.type": {"alone": "มาคนเดียว", "pair": "มากัน 2 คน",
                   "group_3_plus": "มากัน 3 คนขึ้นไป", "unknown": _UNKNOWN},
    "emotion.label": {"neutral": "เป็นกลาง", "happy": "มีความสุข", "sad": "เศร้า",
                      "angry": "โกรธ", "surprise": "ประหลาดใจ", "fear": "กลัว",
                      "disgust": "รังเกียจ", "unknown": "มองไม่เห็นหน้า"},
}

# แถวที่อ่านออกมาจากฐานข้อมูลเป็นชื่อแบน (`emotion_label`) ไม่ใช่ซ้อน (`emotion.label`)
# เส้น GET ย้อนหลังจึงต้องมีชื่อพวกนี้ด้วย ไม่งั้นหน้าเดียวกันจะไทยบ้างอังกฤษบ้าง
FIELD["emotion_label"] = FIELD["emotion.label"]
FIELD["group_type"] = FIELD["group.type"]
FIELD["uniform_kind"] = FIELD["uniform.kind"]

# 🔴 คีย์ที่ห้ามแปลเด็ดขาด แม้ชื่อจะไปพ้องกับตารางข้างบน
# มีไว้กันวันที่ใครเผลอเติม "status" หรือ "source" ลง FIELD
NEVER = frozenset({"status", "source", "id", "ref", "request_id", "camera_id",
                   "object_id", "client_request_id", "text", "prompt_version",
                   "model_name", "provider", "finish_reason", "image_source"})


def _value(path: str, key: str, v: Any) -> Any:
    if key in NEVER or not isinstance(v, str):
        return v
    table = FIELD.get(path) or FIELD.get(key)
    # ไม่มีตาราง = ปล่อยผ่าน ไม่ใช่เดา · ค่าที่ไม่รู้จักก็ปล่อยผ่านเช่นกัน
    return table.get(v, v) if table else v


def thai(obj: Any, parent: str = "") -> Any:
    """คืนก้อนใหม่ที่ค่าเป็นไทย · ไม่แก้ของเดิม ไม่แตะคีย์ ไม่แตะตัวเลข

    ของเดิมต้องไม่ถูกแก้ เพราะก้อนเดียวกันนั้นถูกเอาไปเขียนลงฐานข้อมูลด้วย
    ถ้าแก้ในที่ ฐานข้อมูลจะกลายเป็นไทยไปด้วยโดยไม่มีใครตั้งใจ
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                out[k] = thai(v, k)
            else:
                out[k] = _value(f"{parent}.{k}" if parent else k, k, v)
        return out
    if isinstance(obj, list):
        # list ของ string คือ `carrying` ซึ่งแปลด้วยตารางของคีย์ที่มันอยู่
        return [thai(x, parent) if isinstance(x, (dict, list))
                else _value(parent, parent, x) for x in obj]
    return obj
