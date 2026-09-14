"""ตัวเรียก VLM ของงานอ่านป้ายทะเบียน · LM Studio ตอน dev · OpenRouter ตอน prod

โครงเดียวกับ `people/llm/client.py` แต่ **ไม่ import ข้ามกัน** เหมือนที่งานคน
ไม่ import จากงานช้าง · สามงานนี้จะเปลี่ยน prompt เปลี่ยนโมเดล เปลี่ยนเพดาน
token กันคนละทางแน่นอน ตั้งเพดานร่วมกันเมื่อไหร่ก็จะมีฝั่งหนึ่งโดนตัดกลางประโยค

🔴 งานนี้ต่างจากอีกสองงานตรงที่ **มีคำตอบที่ถูกเพียงคำตอบเดียว**
"เสื้อสีขาว" ที่ผิด ยังพอใช้ตามตัวได้ · "1กข 1234" ที่ผิดหนึ่งตัว ชี้ไปที่รถคนละคัน
ทุกฟังก์ชันในไฟล์นี้จึงเลือกทาง "ไม่ตอบ" มากกว่าทาง "ตอบใกล้เคียง" เสมอ
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ..schemas import PROVINCES, Plate, PlateColor, VehicleType
from . import prompt_v1


def _literal(ann) -> set:
    import typing
    return set(typing.get_args(ann))


_VEHICLE_TYPE = _literal(VehicleType)
_PLATE_COLOR = _literal(PlateColor)
_PROVINCE_SET = set(PROVINCES)

# ชื่อย่อที่โมเดลชอบตอบ -> ชื่อเต็มในชุด
#
# 🔴 **ตารางนี้ไม่ใช่การเดา มันคือการรับสะกดคนละแบบของสิ่งเดียวกัน**
# ทิ้งคำตอบที่ถูกเพราะสะกดย่อ คือบั๊ก `laptop` เป๊ะๆ (ฝั่งงานคน 2026-09-11)
# ห้ามใส่คู่ที่ "ใกล้เคียง" ลงตารางนี้เด็ดขาด เช่น ห้ามแมป "สงขลา " กับ "สงคลา"
# เข้าหากัน · ที่นี่รับได้เฉพาะคำย่อที่คนไทยใช้แทนชื่อเต็มตัวเดียวกันจริงๆ
_PROVINCE_ALIAS = {
    "กรุงเทพ": "กรุงเทพมหานคร",
    "กรุงเทพฯ": "กรุงเทพมหานคร",
    "กทม": "กรุงเทพมหานคร",
    "กทม.": "กรุงเทพมหานคร",
    "bangkok": "กรุงเทพมหานคร",
    "พระนครศรีอยุธยา": "พระนครศรีอยุธยา",
    "อยุธยา": "พระนครศรีอยุธยา",
    "ศรีสะเกษ": "ศรีสะเกษ",
    "ศรีษะเกษ": "ศรีสะเกษ",
}

# เลขไทย -> เลขอารบิก · ป้ายบางแบบพิมพ์เลขไทย และโมเดลลอกมาตามที่เห็น
# ถ้าไม่แปลง regex จะแยกไม่ออกแล้วป้ายที่อ่านถูกจะกลายเป็น "อ่านไม่ออก"
_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")

# 🔴 รูปของป้ายทะเบียนไทย · เลขนำหน้า (ไม่บังคับ) + พยัญชนะไทย 1-3 ตัว + เลข 1-4 หลัก
#
# ช่วง `ก-ฮ` ครอบพยัญชนะไทยทั้ง 44 ตัว **โดยไม่รวมสระและวรรณยุกต์**
# ซึ่งสำคัญ: ถ้าเขียนเป็น `[ก-๙]` แบบหลวมๆ ป้ายที่โมเดลอ่านเพี้ยนเป็นสระ
# จะผ่านด่านนี้ไปเป็น "ทะเบียนที่อ่านได้" ทั้งที่มันไม่ใช่ทะเบียน
_PLATE_RE = re.compile(
    r"^(?P<prefix>[0-9]{0,2})\s*(?P<letters>[ก-ฮ]{1,3})\s*(?P<digits>[0-9]{1,4})$")


@dataclass
class LPRResult:
    data: Optional[Dict[str, Any]]
    raw: str
    finish_reason: Optional[str]
    completion_tokens: Optional[int]
    latency_ms: float
    error: Optional[str] = None

    @property
    def usable(self) -> bool:
        """`finish_reason='length'` = ตอบไม่จบ ห้ามรับมาใช้

        บทเรียนฝั่งช้าง 2026-08-16: คำตอบที่ตัดกลางคันแกะ JSON ได้บ้างไม่ได้บ้าง
        **อันที่แกะได้คืออันที่อันตราย** เพราะมันดูเหมือนคำตอบที่สมบูรณ์
        ที่นี่อาการจะหนักกว่า: ทะเบียนที่ถูกตัดท้ายคือทะเบียนของรถคันอื่น
        """
        return self.data is not None and self.finish_reason != "length"


class PlateLLM:
    def __init__(self) -> None:
        self.provider = os.environ.get("LLM_PROVIDER", "lmstudio")
        self.base_url = os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1")
        # ตัวเดียวกับอีกสองงานเป็นค่าตั้งต้น เพราะเป็นตัวที่พิสูจน์แล้วว่าอ่านภาพจริงได้
        # ⚠️ **ยังไม่ได้วัดว่ามันอ่านอักษรไทยบนป้ายได้ดีแค่ไหน** งาน OCR ตัวอักษร
        # เป็นคนละความสามารถกับงานบรรยายภาพ · แยก env ไว้แล้วเพราะน่าจะต้องเปลี่ยน
        self.model = os.environ.get("LPR_LLM_MODEL",
                                    os.environ.get("LLM_MODEL", "qwen3.6-35b-a3b"))
        self.api_key = os.environ.get("LLM_API_KEY", "not-needed")
        # คำตอบที่นี่สั้นกว่าฝั่งงานคนมาก (ไม่มีก้อน appearance 20 ฟิลด์)
        # ราว 90 tok/คัน x 6 คัน + เผื่อ = 900
        self.max_tokens = int(os.environ.get("LPR_LLM_MAX_TOKENS", "900"))
        self.timeout = float(os.environ.get("LLM_TIMEOUT_S", "25"))
        self.max_retries = int(os.environ.get("LPR_LLM_MAX_RETRIES", "1"))
        self.no_reasoning = os.environ.get("LLM_DISABLE_REASONING",
                                           "true").lower() == "true"
        self.sample_rate = float(os.environ.get("LANGSMITH_SAMPLE_RATE", "0.05"))
        self.tracing = os.environ.get("LANGSMITH_TRACING", "false").lower() == "true"

    def _should_trace(self, anomalous: bool) -> bool:
        if not self.tracing:
            return False
        return True if anomalous else random.random() < self.sample_rate

    def read(self, image_b64: str, w: int, h: int, camera_id: str = "unknown",
             image_hash: str = "", cap: int = 6) -> LPRResult:
        system, user = prompt_v1.build(w, h, camera_id, cap)
        t0 = time.perf_counter()
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from langchain_openai import ChatOpenAI

            kw = {}
            if self.no_reasoning:
                kw["extra_body"] = {"reasoning": {"enabled": False}}
            llm = ChatOpenAI(
                base_url=self.base_url, api_key=self.api_key, model=self.model,
                max_tokens=self.max_tokens, temperature=0, timeout=self.timeout,
                max_retries=self.max_retries, **kw,
            )
            msgs = [
                SystemMessage(content=system),
                HumanMessage(content=[
                    {"type": "text", "text": user},
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime_of(image_b64)};"
                                          f"base64,{image_b64}"}},
                ]),
            ]
            # 🔴 ไม่ส่ง base64 เข้า trace · ภาพงานนี้คือรถของคนจริงพร้อมทะเบียนจริง
            # ซึ่งระบุตัวเจ้าของได้ผ่านฐานทะเบียน ไม่มีเหตุผลให้ไปนอนอยู่ที่เจ้าอื่น
            resp = llm.invoke(msgs, config={
                "metadata": {"image_hash": image_hash,
                             "prompt_version": prompt_v1.PROMPT_VERSION,
                             "frame": f"{w}x{h}"},
                "run_name": "read_plate",
            })
            dt = (time.perf_counter() - t0) * 1000
            meta = resp.response_metadata or {}
            usage = getattr(resp, "usage_metadata", None) or {}
            text = resp.content if isinstance(resp.content, str) else str(resp.content)
            data, err = parse(text)
            return LPRResult(data=data, raw=text,
                             finish_reason=meta.get("finish_reason"),
                             completion_tokens=usage.get("output_tokens"),
                             latency_ms=round(dt, 1), error=err)
        except Exception as e:  # noqa: BLE001
            return LPRResult(None, "", None, None,
                             round((time.perf_counter() - t0) * 1000, 1),
                             f"{type(e).__name__}: {e}")


def mime_of(image_b64: str) -> str:
    """เดา mime จากไบต์แรกของ base64 · ก๊อปมาจากอีกสองงานโดยตั้งใจ

    ที่มา: เคยฝัง image/png ไว้ตายตัว แล้วภาพจากไซต์เป็น JPEG · บาง provider ยอม
    บางเจ้าไม่ยอม กลายเป็นบั๊กที่ขึ้นกับว่า OpenRouter สุ่มส่งไปเจ้าไหน
    """
    head = (image_b64 or "")[:12]
    if head.startswith("/9j/"):
        return "image/jpeg"
    if head.startswith("R0lGOD"):
        return "image/gif"
    if head.startswith("UklGR"):
        return "image/webp"
    return "image/png"


def parse(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """แกะ JSON จากคำตอบ ทนกับ markdown fence และข้อความนำ"""
    if not text:
        return None, "empty response"
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None, "no JSON found"
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return None, f"bad JSON: {e}"
    if not isinstance(data, dict):
        return None, "JSON is not an object"
    return data, None


def vehicles_of(data: Dict[str, Any], cap: int = 6) -> Tuple[list, Optional[str]]:
    """ดึงรายการรถออกจากคำตอบ

    🔴 แยก "ไม่มีป้ายในภาพ" ออกจาก "อ่านไม่ได้" ให้ขาด
    `{"vehicles":[]}` คือคำตอบจริงของลานจอดที่ว่าง ไม่ใช่ความล้มเหลว
    ส่วนคำตอบที่ไม่มีคีย์ `vehicles` เลย แปลว่าโมเดลไม่ได้ทำตามที่สั่ง
    ยุบสองอย่างนี้เป็นอันเดียวเมื่อไหร่ เฟรมที่โมเดลพังจะดูเหมือนลานจอดว่างเป๊ะ
    (ฝั่งช้างเสียเวลาไปทั้งวันกับบั๊กตัวนี้ 2026-08-17)
    """
    if "vehicles" not in data:
        return [], "ไม่มีคีย์ vehicles ในคำตอบ"
    raw = data.get("vehicles")
    if not isinstance(raw, list):
        return [], f"vehicles ไม่ใช่ list (ได้ {type(raw).__name__})"
    out = []
    for i, item in enumerate(raw[:cap]):
        if not isinstance(item, dict):
            continue
        ref = str(item.get("ref") or f"V{i + 1}").strip()[:16] or f"V{i + 1}"
        out.append((ref, item))
    return out, None


# ---------------------------------------------------------------- normalize
def _pick(value: Any, allowed: set, default: str = "unknown") -> str:
    v = str(value).strip().lower().replace(" ", "_") if value is not None else ""
    if v in allowed:
        return v
    if v in {"gray", "grey"} and "grey" in allowed:
        return "grey"
    if v in {"motorbike", "scooter", "motorcycle"} and "motorcycle" in allowed:
        return "motorcycle"
    if v in {"lorry", "truck"} and "truck" in allowed:
        return "truck"
    if v in {"sedan", "car", "suv", "hatchback"} and "car" in allowed:
        return "car"
    if v in {"pick_up", "pickup_truck", "ute"} and "pickup" in allowed:
        return "pickup"
    if v in {"minibus", "minivan", "van"} and "van" in allowed:
        return "van"
    # เฉดสีที่แคบเกิน ปอกคำขยายทิ้ง (บทเรียน dark_green ฝั่งงานคน 2026-09-12)
    for prefix in ("dark_", "light_", "deep_", "bright_", "pale_", "very_"):
        if v.startswith(prefix) and v[len(prefix):] in allowed:
            return v[len(prefix):]
    return default


def _conf(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _province(value: Any) -> str:
    """ชื่อจังหวัดต้องตรงกับชุด 77 ตัว ไม่งั้นว่าง

    🔴 **ห้ามใส่ fuzzy match ที่นี่เด็ดขาด** ต่อให้มันน่าดึงดูดแค่ไหน
    "สงขลา" กับ "สงคลา" ต่างกันหนึ่งตัวอักษร และ fuzzy ที่ยอมรับระยะ 1
    จะจับคู่ "ตาก" กับ "ตราด" ให้ด้วย · จังหวัดที่ผิดคือรถคนละคันในฐานทะเบียน
    เหมือนกับทะเบียนที่ผิด · ว่างไว้แล้วให้คนไปเปิดภาพดู ถูกกว่าเสมอ
    """
    v = str(value or "").strip()
    if not v:
        return ""
    v = v.replace("จังหวัด", "").replace("จ.", "").strip()
    if v in _PROVINCE_SET:
        return v
    return _PROVINCE_ALIAS.get(v.lower(), _PROVINCE_ALIAS.get(v, ""))


def _split_plate(raw: Any) -> Tuple[str, str, str, str]:
    """แยกทะเบียนดิบเป็น (prefix, letters, digits, text) · แยกไม่ออก = ว่างทั้งชุด

    ⚠️ **ฟังก์ชันนี้ไม่ซ่อมทะเบียน** มันแยกหรือไม่แยก เท่านั้น
    เคยคิดจะเติมศูนย์นำหน้าให้ครบสี่หลัก และคิดจะแปลงอักษรละตินที่หน้าตาคล้าย
    อักษรไทย ทั้งสองอย่างถูกทิ้งไปด้วยเหตุผลเดียวกัน: มันคือการเดาว่าป้ายจริง
    ควรเป็นอะไร แล้วผลลัพธ์คือทะเบียนที่ดูสมบูรณ์แบบและชี้ไปที่รถผิดคัน
    ซึ่ง **ปลายทางตรวจไม่ได้เลยแม้ถือภาพอยู่ในมือ** (เหตุผลเดียวกับที่ปิดฟิลด์
    สัญชาติของฝั่งงานคนไว้)

    ทะเบียน 1234 หลักเดียวที่หายไป กับทะเบียนที่ว่าง · อันหลังแพงกว่ามาก
    """
    text = str(raw or "").strip()
    if not text:
        return "", "", "", ""
    # ช่องว่าง/ขีด/จุดที่โมเดลใส่มา ไม่ใช่ส่วนหนึ่งของทะเบียน
    t = text.translate(_THAI_DIGITS)
    t = re.sub(r"[\s\-–—.]+", " ", t).strip()
    m = _PLATE_RE.match(t.replace(" ", ""))
    if not m:
        return "", "", "", ""
    prefix, letters, digits = m.group("prefix"), m.group("letters"), m.group("digits")
    return prefix, letters, digits, f"{prefix}{letters} {digits}"


def plate_of(item: Dict[str, Any]) -> Tuple[Plate, str]:
    """ประกอบก้อน `plate` พร้อมเหตุผลเมื่อแยกไม่ออก"""
    raw = str(item.get("plate_text") or "").strip()[:64]
    prefix, letters, digits, text = _split_plate(raw)
    plate = Plate(text_raw=raw, text=text, prefix=prefix, letters=letters,
                  digits=digits,
                  color=_pick(item.get("plate_color"), _PLATE_COLOR),
                  # 🔴 อ่านไม่ออก = ความมั่นใจศูนย์ ไม่ว่าโมเดลจะบอกมาเท่าไร
                  # ค่าความมั่นใจที่ลอยอยู่โดยไม่มีทะเบียนให้มั่นใจ คือตัวเลข
                  # ที่ปลายทางเอาไปคัดกรองแล้วได้ผลผิดโดยไม่รู้ตัว
                  confidence=_conf(item.get("confidence")) if letters else 0.0)
    if raw and not letters:
        return plate, f"อ่านเป็นรูปทะเบียนไทยไม่ได้ (โมเดลอ่านมาว่า {raw!r})"
    if not raw:
        return plate, str(item.get("reason") or "อ่านทะเบียนไม่ออก").strip()[:200]
    return plate, str(item.get("reason") or "").strip()[:200]


def normalize(item: Dict[str, Any]) -> Dict[str, Any]:
    """คำตอบดิบของโมเดลหนึ่งคัน -> ค่าที่ schema ยอมรับ

    🔴 ทุกอย่างในนี้คือการ "ลดทอน" ไม่ใช่การ "เติม" เหมือนฝั่งงานคนเป๊ะ
    ไม่มีบรรทัดไหนในไฟล์นี้เดาค่าขึ้นมาแทนโมเดล
    """
    plate, reason = plate_of(item)
    v_type = _pick(item.get("vehicle_type"), _VEHICLE_TYPE)
    return {
        "where": str(item.get("where") or "").strip()[:200],
        "plate": plate,
        "province": _province(item.get("province")),
        "province_confidence": (_conf(item.get("province_confidence"))
                                if _province(item.get("province")) else 0.0),
        "vehicle_type": v_type,
        # มองไม่เห็นตัวรถ = ไม่มีอะไรให้มั่นใจ · ตัวเลขที่ลอยอยู่โดยไม่มีคำตอบรองรับ
        # คือของที่ปลายทางเอาไปคัดกรองแล้วผิดโดยไม่รู้ตัว (เรื่องเดียวกับ plate)
        "vehicle_type_confidence": (_conf(item.get("vehicle_type_confidence"))
                                    if v_type != "unknown" else 0.0),
        "vehicle_color": str(item.get("vehicle_color") or "").strip()[:32],
        "description": str(item.get("description") or "").strip()[:300],
        "reason": reason,
    }
