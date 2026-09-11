"""ตัวเรียก VLM ของงานคน · LM Studio ตอน dev · OpenRouter ตอน prod

โครงเดียวกับ `app/llm/client.py` แต่ **ไม่ import ข้ามกัน** ทั้งสองฝั่งจะได้
เปลี่ยน prompt เปลี่ยนโมเดล เปลี่ยน max_tokens กันคนละทางได้โดยไม่กระทบกัน
ซึ่งเป็นสิ่งที่จะเกิดแน่ๆ: งานช้างตอบ 20-39 token งานคนตอบ 250-350 token
ตั้งเพดานร่วมกันเมื่อไหร่ก็จะมีฝั่งหนึ่งโดนตัดกลางประโยค

🔴 ไม่ใช้ agent ไม่ใช้ memory ไม่ใช้ chain ซ้อน · เส้นตรงเหมือนฝั่งช้าง
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ..schemas import Appearance, Garment
from . import prompt_f1, prompt_p1

# ชุดค่าที่ยอมรับ · ดึงจาก Literal ใน schemas.py โดยตรง ไม่พิมพ์ซ้ำ
# เคยมีบั๊กแนวนี้ฝั่งช้าง: `unstable` อยู่ใน schema และในเอกสารตั้งแต่วันแรก
# แต่โค้ดไม่เคยสร้างมันเลย เพราะสองที่นั้นไม่ได้ผูกกัน
_GENDER = {"male", "female", "unknown"}
_AGE = {"0-12", "13-19", "20-29", "30-39", "40-49", "50-59", "60+", "unknown"}


def _allowed(model_cls, field: str) -> set:
    """ดึงชุดค่าที่ field หนึ่งยอมรับ ออกมาจาก pydantic โดยตรง

    เพิ่มค่าใน schemas.py แล้วที่นี่รู้เอง · ไม่มีรายการที่ต้องจำไปแก้สองที่
    """
    import typing
    ann = model_cls.model_fields[field].annotation
    args = typing.get_args(ann)
    return set(args) if args else set()


@dataclass
class PersonResult:
    data: Optional[Dict[str, Any]]
    raw: str
    finish_reason: Optional[str]
    completion_tokens: Optional[int]
    latency_ms: float
    error: Optional[str] = None

    @property
    def usable(self) -> bool:
        """finish_reason='length' = ตอบไม่จบ ห้ามรับมาใช้

        วัดจริงฝั่งช้าง 2026-08-16: คำตอบที่ตัดกลางคันแกะ JSON ได้บ้างไม่ได้บ้าง
        อันที่แกะได้คือที่อันตราย เพราะมันดูเหมือนคำตอบที่สมบูรณ์
        งานนี้คำตอบยาวกว่าฝั่งช้างสิบเท่า โอกาสโดนตัดจึงสูงกว่ามาก
        """
        return self.data is not None and self.finish_reason != "length"


class PersonLLM:
    def __init__(self) -> None:
        self.provider = os.environ.get("LLM_PROVIDER", "lmstudio")
        self.base_url = os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1")
        # ใช้ตัวเดียวกับฝั่งช้างเป็นค่าตั้งต้น (qwen3-vl-32b-instruct บน prod)
        # เพราะเป็นตัวที่พิสูจน์แล้วว่าอ่านภาพจริงได้ · แยก env ไว้เผื่อวันหนึ่ง
        # งานคนต้องใช้โมเดลคนละตัว ซึ่งน่าจะเกิด งานนี้ต้องการรายละเอียดมากกว่า
        self.model = os.environ.get("PEOPLE_LLM_MODEL",
                                    os.environ.get("LLM_MODEL", "qwen3.6-35b-a3b"))
        self.api_key = os.environ.get("LLM_API_KEY", "not-needed")
        # 🔴 300 ของฝั่งช้างไม่พอแน่นอน · คำตอบที่นี่มีราว 25 ฟิลด์
        # บวกคำบรรยายไทยอีกหนึ่งย่อหน้า (ภาษาไทยกิน token มากกว่าอังกฤษราวเท่าตัว)
        # วัดจริงแล้วปรับได้ แต่เริ่มจากเผื่อไว้ ดีกว่าโดน finish=length ทุกใบ
        self.max_tokens = int(os.environ.get("PEOPLE_LLM_MAX_TOKENS", "700"))
        # เส้น /v1/frames ตอบทุกคนในครั้งเดียว คำตอบยาวตามจำนวนคน ไม่คงที่เหมือน p1
        # วัดจริง 2026-09-11: สามคน 567 tok ≈ 190 tok/คน · เพดาน 12 คน = ~2,300
        # ตั้ง 2500 เผื่อไว้ · โดน finish=length เมื่อไหร่ = เสียทั้งเฟรม ไม่ใช่เสียคนเดียว
        self.frame_max_tokens = int(os.environ.get("PEOPLE_FRAME_MAX_TOKENS", "2500"))
        self.timeout = float(os.environ.get("LLM_TIMEOUT_S", "25"))
        # 🔴 ค่าเริ่มต้นของ langchain คือ retry 2 ครั้ง ซึ่งแปลว่าตอน OpenRouter ล่ม
        # คนหนึ่งคนกิน 3 เท่าของ timeout ก่อนจะยอมแพ้ (75 วิ ที่ timeout 25)
        # ปลายทางเป็นระบบ realtime ที่หน้าประตู รอขนาดนั้นไม่ได้ · retry 1 พอ
        # (เจอ 429 ชั่วคราวยังได้ลองอีกครั้ง แต่ไม่ค้างเป็นนาที)
        self.max_retries = int(os.environ.get("PEOPLE_LLM_MAX_RETRIES", "1"))
        # ปิด thinking · ฝั่งช้างวัดได้ว่าเร็วขึ้น 4.5 เท่าและถูกลง 3.7 เท่า
        # สิ่งที่ลองแล้วไม่ได้ผล อย่าเสียเวลาลองซ้ำ ดู app/llm/client.py
        self.no_reasoning = os.environ.get("LLM_DISABLE_REASONING", "true").lower() == "true"
        self.sample_rate = float(os.environ.get("LANGSMITH_SAMPLE_RATE", "0.05"))
        self.tracing = os.environ.get("LANGSMITH_TRACING", "false").lower() == "true"

    def _should_trace(self, anomalous: bool) -> bool:
        if not self.tracing:
            return False
        return True if anomalous else random.random() < self.sample_rate

    # ---------------------------------------------------------------- call
    def describe(self, image_b64: str, object_id: str, w: int, h: int,
                 camera_id: str = "unknown", image_hash: str = "") -> PersonResult:
        system, user = prompt_p1.build(object_id, w, h, camera_id)
        return self._invoke(system, user, image_b64, image_hash,
                            prompt_p1.PROMPT_VERSION, self.max_tokens,
                            {"object_id": object_id, "crop": f"{w}x{h}"},
                            "describe_person")

    def describe_frame(self, image_b64: str, w: int, h: int,
                       camera_id: str = "unknown", image_hash: str = "",
                       cap: int = 12) -> PersonResult:
        """เส้น f1 · ทั้งเฟรม ไม่มีใครถูกชี้ ตอบทุกคนในครั้งเดียว

        คำตอบยาวตามจำนวนคน ต่างจาก p1 ที่ยาวคงที่ · โควตา token จึงต้องคนละตัว
        วัดจริง 2026-09-11 กับเฟรมสามคน: 567 tok ที่ราว 190 tok/คน
        """
        system, user = prompt_f1.build(w, h, camera_id, cap)
        return self._invoke(system, user, image_b64, image_hash,
                            prompt_f1.PROMPT_VERSION, self.frame_max_tokens,
                            {"frame": f"{w}x{h}", "cap": cap}, "describe_frame")

    def _invoke(self, system: str, user: str, image_b64: str, image_hash: str,
                prompt_version: str, max_tokens: int, meta_extra: dict,
                run_name: str) -> PersonResult:
        t0 = time.perf_counter()
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage, SystemMessage

            kw = {}
            if self.no_reasoning:
                kw["extra_body"] = {"reasoning": {"enabled": False}}
            llm = ChatOpenAI(
                base_url=self.base_url, api_key=self.api_key, model=self.model,
                max_tokens=max_tokens, temperature=0, timeout=self.timeout,
                max_retries=self.max_retries, **kw,
            )
            msgs = [
                SystemMessage(content=system),
                HumanMessage(content=[
                    {"type": "text", "text": user},
                    {"type": "image_url",
                     "image_url": {"url": f"data:{mime_of(image_b64)};base64,{image_b64}"}},
                ]),
            ]
            # 🔴 ไม่ส่ง base64 เข้า trace · ที่นี่สำคัญกว่าฝั่งช้างอีกขั้น
            # ภาพฝั่งช้างคือสัตว์ในป่า ภาพฝั่งนี้คือหน้าคนที่เดินเข้าอาคาร
            # ไม่มีเหตุผลอะไรที่มันต้องไปนอนอยู่บนเซิร์ฟเวอร์ของเจ้าอื่น
            resp = llm.invoke(msgs, config={
                "metadata": {"image_hash": image_hash,
                             "prompt_version": prompt_version, **meta_extra},
                "run_name": run_name,
            })
            dt = (time.perf_counter() - t0) * 1000
            meta = resp.response_metadata or {}
            usage = getattr(resp, "usage_metadata", None) or {}
            text = resp.content if isinstance(resp.content, str) else str(resp.content)
            data, err = parse(text)
            return PersonResult(data=data, raw=text,
                                finish_reason=meta.get("finish_reason"),
                                completion_tokens=usage.get("output_tokens"),
                                latency_ms=round(dt, 1), error=err)
        except Exception as e:
            # เน็ตหลุด / โมเดลไม่มี / timeout → คนคนนี้เป็น degraded
            # ไม่ทำให้ทั้ง request พัง คนอื่นในนัดเดียวกันยังต้องได้คำตอบ
            return PersonResult(None, "", None, None,
                                round((time.perf_counter() - t0) * 1000, 1),
                                f"{type(e).__name__}: {e}")


def mime_of(image_b64: str) -> str:
    """เดา mime จากไบต์แรกของ base64

    ก๊อปมาจาก app/llm/client.py ตั้งใจ ไม่ import ข้ามแพ็กเกจเพื่อของสิบบรรทัด
    ที่มา: เคยฝัง image/png ไว้ตายตัว แล้วภาพจากไซต์เป็น JPEG
    บาง provider ยอม บางเจ้าไม่ยอม กลายเป็นบั๊กที่ขึ้นกับว่า OpenRouter
    สุ่มส่งไปเจ้าไหน ซึ่งเป็นบั๊กที่ทำซ้ำไม่ได้
    """
    head = (image_b64 or "")[:12]
    if head.startswith("/9j/"):
        return "image/jpeg"
    if head.startswith("R0lGOD"):
        return "image/gif"
    if head.startswith("UklGR"):
        return "image/webp"
    return "image/png"


# ---------------------------------------------------------------- parse
def parse(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """แกะ JSON จากคำตอบ ทนกับ markdown fence และข้อความนำ

    ใช้ตัวแรกถึงปีกกาปิดตัวสุดท้าย เพราะคำตอบมี object ซ้อน (top/bottom/footwear)
    regex แบบ non-greedy จะตัดกลาง object · ฝั่งช้างไม่มีปัญหานี้เพราะคำตอบแบน
    """
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


def people_of(data: Dict[str, Any], cap: int = 12) -> Tuple[list, Optional[str]]:
    """ดึงรายการคนออกจากคำตอบของเส้น f1

    🔴 แยก "ไม่มีคนในเฟรม" ออกจาก "อ่านไม่ได้" ให้ชัด
    `{"people":[]}` คือคำตอบจริงของล็อบบี้ที่ว่าง ไม่ใช่ความล้มเหลว
    ส่วนคำตอบที่ไม่มีคีย์ `people` เลย แปลว่าโมเดลไม่ได้ทำตามที่สั่ง ซึ่งคนละเรื่อง
    ถ้ายุบสองอย่างนี้เป็นอันเดียว เฟรมที่โมเดลพังจะดูเหมือนล็อบบี้ว่างเป๊ะ
    ซึ่งคือบั๊กเดียวกับที่ฝั่งช้างเสียเวลาไปทั้งวัน
    """
    if "people" not in data:
        return [], "ไม่มีคีย์ people ในคำตอบ"
    raw = data.get("people")
    if not isinstance(raw, list):
        return [], f"people ไม่ใช่ list (ได้ {type(raw).__name__})"
    out = []
    for i, item in enumerate(raw[:cap]):
        if not isinstance(item, dict):
            continue
        ref = str(item.get("ref") or f"P{i + 1}").strip()[:16] or f"P{i + 1}"
        out.append((ref, str(item.get("where") or "").strip()[:200], item))
    return out, None


def _pick(value: Any, allowed: set, default: str = "unknown") -> str:
    """ค่านอกชุด = unknown ไม่ใช่ 500 และไม่ใช่ปล่อยผ่าน

    โมเดลตอบ "light-medium" หรือ "สีดำ" หรือ "N/A" ได้ทั้งนั้น
    ปล่อยผ่าน = pydantic โยน ValidationError แล้วคนคนนั้นหายไปทั้งคน
    ทั้งที่ฟิลด์อื่นอ่านได้ครบ · เสียหนึ่งช่อง ไม่ใช่เสียทั้งคน
    """
    v = str(value).strip().lower().replace(" ", "_") if value is not None else ""
    if v in allowed:
        return v
    # เผื่อสะกดแบบอเมริกัน ซึ่งโมเดลส่วนใหญ่ชอบใช้
    if v == "multicolor" and "multicolour" in allowed:
        return "multicolour"
    if v in {"gray", "grey"} and "grey" in allowed:
        return "grey"
    if v in {"true", "yes_"} and "yes" in allowed:
        return "yes"
    if v in {"false", "none", "no_"} and "no" in allowed:
        return "no"
    return default


def _conf(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _garment(raw: Any) -> Garment:
    d = raw if isinstance(raw, dict) else {}
    colors = _allowed(Garment, "color")
    return Garment(
        type=str(d.get("type", "unknown")).strip().lower().replace(" ", "_")[:32] or "unknown",
        color=_pick(d.get("color"), colors),
        secondary_color=_pick(d.get("secondary_color"), colors),
        pattern=_pick(d.get("pattern"), _allowed(Garment, "pattern")),
    )


def normalize(data: Dict[str, Any]) -> Dict[str, Any]:
    """เปลี่ยนคำตอบดิบของโมเดลให้เป็นค่าที่ schema ยอมรับ

    🔴 ทุกอย่างในนี้คือการ "ลดทอน" ไม่ใช่การ "เติม"
    ฟิลด์ไหนที่โมเดลไม่ตอบหรือตอบนอกชุด จะกลายเป็น unknown เสมอ
    ห้ามมีบรรทัดไหนในไฟล์นี้เดาค่าขึ้นมาแทนโมเดล
    """
    ap_raw = data.get("appearance") if isinstance(data.get("appearance"), dict) else {}

    appearance = Appearance(
        skin_tone=_pick(ap_raw.get("skin_tone"), _allowed(Appearance, "skin_tone")),
        build=_pick(ap_raw.get("build"), _allowed(Appearance, "build")),
        height=_pick(ap_raw.get("height"), _allowed(Appearance, "height")),
        hair_length=_pick(ap_raw.get("hair_length"), _allowed(Appearance, "hair_length")),
        hair_color=_pick(ap_raw.get("hair_color"), _allowed(Appearance, "hair_color")),
        glasses=_pick(ap_raw.get("glasses"), _allowed(Appearance, "glasses")),
        face_mask=_pick(ap_raw.get("face_mask"), _allowed(Appearance, "face_mask")),
        facial_hair=_pick(ap_raw.get("facial_hair"), _allowed(Appearance, "facial_hair")),
        headwear=_pick(ap_raw.get("headwear"), _allowed(Appearance, "headwear")),
        top=_garment(ap_raw.get("top")),
        top_sleeve=_pick(ap_raw.get("top_sleeve"), _allowed(Appearance, "top_sleeve")),
        outer=_garment(ap_raw.get("outer")),
        bottom=_garment(ap_raw.get("bottom")),
        footwear=_garment(ap_raw.get("footwear")),
        carrying=_carrying(ap_raw.get("carrying")),
        distinctive=str(ap_raw.get("distinctive") or "")[:200],
    )
    return {
        "gender": _pick(data.get("gender"), _GENDER),
        "gender_confidence": _conf(data.get("gender_confidence")),
        "age_range": _pick(data.get("age_range"), _AGE),
        "age_range_confidence": _conf(data.get("age_range_confidence")),
        "appearance": appearance,
        "appearance_confidence": _conf(data.get("appearance_confidence")),
        "description": str(data.get("description") or "").strip()[:600],
        "reason": str(data.get("reason") or "").strip()[:200],
    }


def _carrying(raw: Any) -> list:
    """รายการของที่ถือ · ตัวที่ไม่รู้จักทิ้ง ไม่ใช่ยัดเป็น other ทุกตัว

    เพดาน 6 ชิ้น · โมเดลที่หลุดเข้าโหมดบรรยายจะไล่ list ของทุกอย่างในเฟรม
    รวมถึงของที่ไม่ได้อยู่กับคนคนนี้
    """
    import typing
    allowed = set(typing.get_args(typing.get_args(
        Appearance.model_fields["carrying"].annotation)[0]))
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw[:6]:
        v = _pick(item, allowed, default="")
        if v and v not in out:
            out.append(v)
    return out
