"""แปลค่าใน JSON เป็นไทยตอนขาออก · **คีย์ยังเป็นอังกฤษ ค่าเป็นไทย**

กติกาเดียวกับ `people/th.py` ทุกข้อ (Toy สั่งไว้ตั้งแต่ 2026-09-12) ไม่ใช่กฎใหม่:
  prompt + normalize + ฐานข้อมูล = **อังกฤษ ชุดค่าปิด** (ของที่ใช้ทำงาน)
  ตอนจะตอบออกไป                  = **ไทย**              (ของที่คนอ่าน)

🔴 สิ่งที่ **ไม่แปล** ในงานนี้ และเหตุผล

`plate.text` `plate.text_raw` `plate.letters` `plate.digits` `plate.prefix`
คือ **สิ่งที่พิมพ์อยู่บนป้ายจริง** เป็นหลักฐาน ไม่ใช่ค่าที่เราตีความ
แตะเมื่อไหร่ก็ไม่ใช่หลักฐานอีกต่อไป · และมันเป็นภาษาไทยอยู่แล้วโดยธรรมชาติ

`province` เป็นภาษาไทยอยู่แล้วตั้งแต่ prompt (มันคือข้อความบนป้าย)

`status` `ref` `request_id` `camera_id` `image.source` เป็นค่าโปรโตคอล
ปลายทางเขียน `if status == "ok"` ไว้แล้ว แปลเมื่อไหร่โค้ดเขาพังเงียบๆ

เหลือของที่ต้องแปลจริงแค่สองชุด: ชนิดรถ กับ สีป้าย
"""
from __future__ import annotations

from typing import Any, Dict

_UNKNOWN = "ไม่ทราบ"

FIELD: Dict[str, Dict[str, str]] = {
    "vehicle_type": {"car": "รถเก๋ง", "pickup": "รถกระบะ",
                     "motorcycle": "รถจักรยานยนต์", "truck": "รถบรรทุก",
                     "van": "รถตู้", "bus": "รถบัส/รถโดยสาร",
                     "other": "รถประเภทอื่น",
                     # 🔴 ไม่ใช่ "ไม่ทราบ" เฉยๆ · ค่านี้แปลว่ามองไม่เห็นตัวรถ
                     # ซึ่งเป็นเคสปกติเมื่อปลายทางส่งมาแค่ครอปของป้าย
                     # คำว่า "ไม่ทราบ" ทำให้คนอ่านคิดว่าระบบพยายามแล้วไม่สำเร็จ
                     "unknown": "มองไม่เห็นตัวรถ"},
    # สีพื้นป้าย · หลักฐานที่มองเห็น ไม่ใช่ประเภทป้าย (ดู schemas.PlateColor)
    "plate_color": {"white": "ขาว", "yellow": "เหลือง", "green": "เขียว",
                    "red": "แดง", "blue": "น้ำเงิน", "black": "ดำ",
                    "orange": "ส้ม", "other": "สีอื่น", "unknown": _UNKNOWN},
    # รูปทะเบียนที่อ่านได้ · **บอกรูปของตัวหนังสือ ไม่ได้บอกจังหวัดที่จดทะเบียน**
    # เขียนตัวอย่างไว้ในคำแปลเลย เพราะคนอ่านหน้าเว็บไม่ได้เปิด schema อ่านตาม
    "plate_pattern": {"two_letter": "สองอักษร (กข 1234)",
                      "prefixed": "มีเลขนำหน้า (1กข 1234)",
                      "three_letter": "สามอักษร (กขค 123)",
                      "commercial": "รถบรรทุก/รถโดยสาร (10-0001)",
                      "unknown": "ไม่เข้ารูปทะเบียนไทย"},
}

# ชื่อแบนที่อ่านออกมาจากฐานข้อมูล (เส้น GET ย้อนหลังใช้ชื่อพวกนี้)
FIELD["plate.color"] = FIELD["plate_color"]
FIELD["plate.pattern"] = FIELD["plate_pattern"]

# 🔴 คีย์ที่ห้ามแปลเด็ดขาด แม้ชื่อจะไปพ้องกับตารางข้างบน
# `text` `text_raw` `letters` `digits` `prefix` คือหลักฐานจากป้ายจริง
NEVER = frozenset({"status", "source", "ref", "request_id", "camera_id",
                   "client_request_id", "text", "text_raw", "letters", "digits",
                   "prefix", "province", "plate_text", "plate_text_raw",
                   "prompt_version", "model_name", "provider", "finish_reason",
                   "image_source"})


def _value(path: str, key: str, v: Any) -> Any:
    if key in NEVER or not isinstance(v, str):
        return v
    table = FIELD.get(path) or FIELD.get(key)
    # ไม่มีตาราง = ปล่อยผ่าน ไม่ใช่เดา (เหมือน people/th.py)
    return table.get(v, v) if table else v


def thai(obj: Any, parent: str = "") -> Any:
    """คืนก้อนใหม่ที่ค่าเป็นไทย · ไม่แก้ของเดิม ไม่แตะคีย์ ไม่แตะตัวเลข

    ของเดิมต้องไม่ถูกแก้ เพราะก้อนเดียวกันนั้นถูกเอาไปเขียนลงฐานข้อมูลด้วย
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
        return [thai(x, parent) if isinstance(x, (dict, list))
                else _value(parent, parent, x) for x in obj]
    return obj
