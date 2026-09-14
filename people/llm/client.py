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

from ..schemas import Appearance, Emotion, Garment, GroupInfo, Uniform
from . import prompt_f1, prompt_p1

# ชุดค่าที่ยอมรับ · ดึงจาก Literal ใน schemas.py โดยตรง ไม่พิมพ์ซ้ำ
# เคยมีบั๊กแนวนี้ฝั่งช้าง: `unstable` อยู่ใน schema และในเอกสารตั้งแต่วันแรก
# แต่โค้ดไม่เคยสร้างมันเลย เพราะสองที่นั้นไม่ได้ผูกกัน
_GENDER = {"male", "female", "unknown"}
_DIRECTION = {"in", "out", "unknown"}
_AGE = {"0-12", "13-19", "20-29", "30-39", "40-49", "50-59", "60+", "unknown"}
_NATIONALITY = {"thai", "asian_other", "western", "other", "unknown"}
_MOBILITY = {"walking", "stroller", "wheelchair", "carried", "other", "unknown"}
"""เดินเอง / รถเข็นเด็ก / วีลแชร์ / ถูกอุ้ม · Toy สั่ง 2026-09-14

ชุดนี้**ไม่ได้ดึงจาก schema ด้วย `_allowed`** เหมือน appearance เพราะมันอยู่ระดับบน
ไม่ใช่ในก้อน Appearance · `test_ชุดค่าในโค้ดกับใน_prompt_ต้องตรงกันทุกตัว`
เฝ้าให้ตรงกับ `PersonOut.mobility` และกับ prompt ทั้งสองเส้น"""

# 🔴 สวิตช์เดียวของทั้งระบบสำหรับเรื่องสัญชาติ · ค่าเริ่มต้นคือปิด
#
# ปิดอยู่ = prompt ไม่ถาม **และ** normalize บังคับเป็น unknown อีกชั้น
# สองชั้นโดยตั้งใจ ไม่ใช่เผื่อเหนียว: วันหนึ่งจะมีคนแก้ prompt แล้วลืมสวิตช์
# หรือโมเดลตอบฟิลด์ที่ไม่ได้ถามมาเอง (เจอมาแล้วกับ `where` ตอนทำ p1)
# **ด่านที่มีชั้นเดียวคือด่านที่พังเงียบ**
#
# อ่านเหตุผลที่ปิดไว้ที่ schemas.Nationality ก่อนคิดจะเปิด
ALLOW_NATIONALITY = os.environ.get("PEOPLE_ALLOW_NATIONALITY",
                                   "false").strip().lower() == "true"

# ช่วงชีวิตจากช่วงอายุ · คำนวณ ไม่ได้ถาม (ดู schemas.AgeGroup)
_AGE_GROUP = {"0-12": "child", "13-19": "teen",
              "20-29": "adult", "30-39": "adult",
              "40-49": "adult", "50-59": "adult", "60+": "senior"}


def age_group_of(age_range: str) -> str:
    """child / teen / adult / senior จาก age_range ตัวเดียว

    ค่านอกตารางเป็น unknown เสมอ · ฟังก์ชันนี้ไม่มีทางคืนค่าที่ขัดกับ age_range
    เพราะมันอ่านจาก age_range ตัวเดียวและไม่มีอินพุตอื่น ซึ่งคือเหตุผลทั้งหมด
    ที่ไม่ไปถามโมเดลเอา
    """
    return _AGE_GROUP.get(age_range, "unknown")


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
        # 700 -> 850 (2026-09-12) ตอนเพิ่ม uniform (4 คีย์) + group + สัญชาติ
        # ราว +45 tok/คน · เผื่อไว้ ดีกว่าโดนตัดกลางแล้วเสียทั้งคน
        self.max_tokens = int(os.environ.get("PEOPLE_LLM_MAX_TOKENS", "850"))
        # เส้น /v1/frames ตอบทุกคนในครั้งเดียว คำตอบยาวตามจำนวนคน ไม่คงที่เหมือน p1
        # วัดจริง 2026-09-11: สามคน 567 tok ≈ 190 tok/คน · เพดาน 12 คน = ~2,300
        #
        # 🔴 2500 -> 3500 (2026-09-12) · uniform + group ทำให้เป็นราว 240 tok/คน
        # 240 x 12 = 2,880 ซึ่ง **ทะลุ 2500 เดิม** แปลว่าถ้าไม่ขยับพร้อมกับ prompt
        # เฟรมที่มีคนเยอะจะโดน finish=length แล้ว **เสียทั้งเฟรม ไม่ใช่เสียคนเดียว**
        # และอาการที่เห็นคือ degraded ลอยๆ ไม่ได้ชี้มาที่ prompt ที่เพิ่งแก้เลย
        self.frame_max_tokens = int(os.environ.get("PEOPLE_FRAME_MAX_TOKENS", "3500"))
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
        system, user = prompt_p1.build(object_id, w, h, camera_id,
                                       nationality=ALLOW_NATIONALITY)
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
        system, user = prompt_f1.build(w, h, camera_id, cap,
                                       nationality=ALLOW_NATIONALITY)
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
    # 🔴 "dark_green" -> "green" · วัดจริงบน prod 2026-09-12
    #
    # ยิงกรอบ รปภ. ชุดเขียวเข้มเข้าไป ได้ `uniform.color: unknown` กลับมา
    # ทั้งที่ช่องบรรยายเขียนว่า "เสื้อสีเขียวเข้ม" ชัดเจน · โมเดลอ่านถูก
    # เราทิ้งเอง เพราะ `dark_green` ไม่อยู่ในชุด **นี่คือบั๊ก `laptop` รอบที่สอง
    # ต่างแค่ฟิลด์** ชุดค่าปิดที่แคบเกินจริง = ทิ้งของดีเงียบๆ ไม่มี error ให้เห็น
    #
    # ทางแก้คือปอกคำขยายทิ้ง **ไม่ใช่เพิ่ม dark_green ลงในชุด** เพราะชุดสีตั้งใจ
    # ให้หยาบ ปลายทางต้องกรองได้ว่า "หาคนเสื้อเขียว" แล้วเจอทั้งเขียวเข้มเขียวอ่อน
    # เพิ่มเฉดเมื่อไหร่ การกรองก็แตกเป็นเสี่ยงทันที
    for prefix in ("dark_", "light_", "deep_", "bright_", "pale_", "very_"):
        if v.startswith(prefix) and v[len(prefix):] in allowed:
            return v[len(prefix):]
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


def _uniform(raw: Any) -> Uniform:
    """เครื่องแบบ · `none` (ไม่ได้ใส่) กับ `unknown` (ดูไม่ออก) ห้ามยุบเป็นอันเดียว

    `_pick` คืน default เป็น "unknown" อยู่แล้วเมื่อค่านอกชุด ซึ่งถูกต้องที่นี่:
    โมเดลตอบ "staff uniform" มา เราไม่รู้ว่าหมายถึงอะไร = ดูไม่ออก
    ไม่ใช่ = ไม่ได้ใส่ · เดาไปทาง none คือการลบคนใส่เครื่องแบบออกจากยอดนับเงียบๆ
    """
    d = raw if isinstance(raw, dict) else {}
    return Uniform(
        kind=_pick(d.get("kind"), _allowed(Uniform, "kind")),
        color=_pick(d.get("color"), _allowed(Uniform, "color")),
        id_badge=_pick(d.get("id_badge"), _allowed(Uniform, "id_badge")),
        # ข้อความบนชุดเป็น free text โดยตั้งใจ ("SECURITY", "รปภ.", ชื่อบริษัท)
        # ชุดค่าปิดตรงนี้คือการทิ้งหลักฐานที่ดีที่สุดที่เรามี
        text=str(d.get("text") or "").strip()[:40],
    )


# สีหน้าแต่ละแบบตกอยู่ช่วงไหนของสเกล 1-5 (1 = bad, 5 = very happy)
# ใช้ตอนโมเดลตอบ label มาแต่ไม่ตอบเลข หรือตอบเลขที่ขัดกับ label ของตัวเอง
_EMOTION_VALENCE = {"happy": 4, "neutral": 3, "surprise": 3,
                    "sad": 2, "fear": 2, "disgust": 2, "angry": 1}
# ช่วงที่ label หนึ่งๆ อยู่ได้ · นอกช่วงนี้คือคำตอบที่ขัดกับตัวเอง
# surprise กว้างกว่าตัวอื่นเพราะ "ตกใจดีใจ" กับ "ตกใจกลัว" หน้าเหมือนกันมาก
_EMOTION_RANGE = {"happy": (4, 5), "neutral": (3, 3), "surprise": (2, 4),
                  "sad": (1, 2), "fear": (1, 2), "disgust": (1, 2),
                  "angry": (1, 2)}

# โมเดลชอบตอบรูป adjective ("surprised") ทั้งที่เราขอ noun ตาม FER2013
# 🔴 ทิ้งไปเป็น unknown คือการโยนคำตอบที่ถูกแล้วทิ้งเพราะสะกดคนละแบบ
# ซึ่งคือบั๊กเดียวกับ `laptop` เป๊ะ · เรื่องเดียวกับ gray/grey ที่รับไว้อยู่แล้ว
_EMOTION_ALIAS = {"surprised": "surprise", "fearful": "fear", "afraid": "fear",
                  "scared": "fear", "disgusted": "disgust", "happiness": "happy",
                  "sadness": "sad", "anger": "angry", "calm": "neutral"}


def _emotion(raw: Any) -> Emotion:
    """สีหน้า + สเกล 1-5 · **บังคับให้ label กับ valence ไม่ขัดกันเอง**

    🔴 โมเดลตอบ `{"label":"happy","valence":1}` ได้สบายๆ และนั่นคือคำตอบที่
    ปลายทางเอาไปทำอะไรต่อไม่ได้เลย ต้องเลือกว่าจะเชื่ออันไหน
    **เราเชื่อ label** เพราะมันคือสิ่งที่โมเดลมองเห็น ส่วนตัวเลขคือสิ่งที่มันต้อง
    แปลงเอง ซึ่งเป็นงานคนละชนิดและเป็นงานที่มันพลาดบ่อยกว่า (เรื่องเดียวกับ
    group.size ที่เราไม่ให้มันนับ) · valence ที่หลุดช่วงของ label ถูกดึงกลับเข้าช่วง
    ไม่ใช่ทิ้งทั้งก้อน

    ไม่มี label = ไม่มีอารมณ์ให้รายงาน = unknown/0 **ไม่ใช่ neutral/3**
    "มองไม่เห็นหน้า" กับ "หน้าเฉยๆ" คนละเรื่องกันคนละชั้น เหมือน unknown กับ degraded
    """
    d = raw if isinstance(raw, dict) else {}
    raw_label = str(d.get("label") or "").strip().lower()
    label = _pick(_EMOTION_ALIAS.get(raw_label, raw_label),
                  _allowed(Emotion, "label"))
    if label == "unknown":
        return Emotion()
    try:
        valence = int(round(float(d.get("valence"))))
    except (TypeError, ValueError):
        valence = _EMOTION_VALENCE[label]
    lo, hi = _EMOTION_RANGE[label]
    valence = max(lo, min(hi, valence))
    return Emotion(label=label, valence=valence,
                   confidence=_conf(d.get("confidence")))


def _group_ref(raw: Any) -> str:
    """เลขกลุ่มดิบจากโมเดล · ขนาดกลุ่มยังไม่รู้ตรงนี้ ต้องเห็นครบทุกคนก่อน

    ค่าว่าง = โมเดลไม่ได้ตอบ ซึ่งแปลว่า "ไม่รู้" ไม่ใช่ "เดินคนเดียว"
    """
    v = str(raw or "").strip()[:16]
    return v if v and v.lower() not in {"unknown", "none", "null", "n/a"} else ""


def apply_groups(normalized: list) -> None:
    """เติม `size` กับ `type` ให้ทุกคน หลังเห็นครบทั้งเฟรมแล้ว · แก้ในที่

    🔴 **เรานับเอง ไม่ให้โมเดลนับ** เหตุผลเดียวกับที่ฝั่งช้างให้ CV นับ ไม่ให้ VLM นับ:
    โมเดลบอก "เขามากับผู้หญิงคนนั้น" ได้ดี แต่บอก "กลุ่มนี้มี 3 คน" ได้ไม่คงเส้นคงวา
    มันแปะป้ายเหมือนกันให้สองคน แล้วเขียนว่า size 3 ได้สบายๆ ในคำตอบเดียวกัน
    ป้ายคือสิ่งที่มันทำได้ การนับคือสิ่งที่เราทำได้ ต่างคนต่างทำส่วนที่ตัวเองไม่พลาด

    คนที่ไม่มี ref (โมเดลไม่ตอบ) เป็น unknown ไม่ใช่ alone · ดู GroupType
    """
    counts: Dict[str, int] = {}
    for n in normalized:
        ref = n["group"].ref
        if ref:
            counts[ref] = counts.get(ref, 0) + 1
    for n in normalized:
        ref = n["group"].ref
        if not ref:
            continue
        size = counts[ref]
        n["group"].size = size
        n["group"].type = ("alone" if size == 1 else
                           "pair" if size == 2 else "group_3_plus")


def normalize(data: Dict[str, Any], with_group: bool = False) -> Dict[str, Any]:
    """เปลี่ยนคำตอบดิบของโมเดลให้เป็นค่าที่ schema ยอมรับ

    🔴 ทุกอย่างในนี้คือการ "ลดทอน" ไม่ใช่การ "เติม"
    ฟิลด์ไหนที่โมเดลไม่ตอบหรือตอบนอกชุด จะกลายเป็น unknown เสมอ
    ห้ามมีบรรทัดไหนในไฟล์นี้เดาค่าขึ้นมาแทนโมเดล
    (`age_group` ไม่ใช่ข้อยกเว้น มันคือ `age_range` ที่หยาบลง ไม่ใช่ค่าใหม่)

    `with_group` เปิดเฉพาะเส้น `/v1/frames` ที่โมเดลเห็นทุกคนพร้อมกัน
    เส้น `/v1/persons` ปิดไว้ **แม้โมเดลจะตอบ `group` มาเองก็ทิ้ง** เพราะมันเห็น
    คนเดียวจะรู้ได้ยังไงว่าใครมาด้วยกัน · ค่าที่ฟังดูมีความหมายแต่ไม่มีฐานอะไรรองรับ
    แย่กว่าช่องว่าง (เรื่องเดิมกับ `where` ที่เส้น persons บังคับเป็น "")
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
        uniform=_uniform(ap_raw.get("uniform")),
        distinctive=str(ap_raw.get("distinctive") or "")[:200],
    )
    age_range = _pick(data.get("age_range"), _AGE)
    return {
        # 🔴 ค่านอกชุดเป็น unknown เหมือนทุกช่อง · ห้ามแปลง "entering"/"เข้า" ให้เอง
        # ถ้าโมเดลตอบนอกชุดบ่อย ให้ไปแก้ prompt ไม่ใช่มาเดาความหมายตรงนี้
        "direction": _pick(data.get("direction"), _DIRECTION),
        "direction_confidence": _conf(data.get("direction_confidence")),
        "gender": _pick(data.get("gender"), _GENDER),
        "gender_confidence": _conf(data.get("gender_confidence")),
        "age_range": age_range,
        "age_range_confidence": _conf(data.get("age_range_confidence")),
        # คำนวณจาก age_range ที่ลดทอนแล้ว ไม่ใช่จากที่โมเดลตอบดิบ
        # ("ประมาณ 35 ปี" -> age_range unknown -> age_group unknown ตามกันไป)
        "age_group": age_group_of(age_range),
        # 🔴 อยู่ในรถเข็นไหม · ถามโมเดลตรงๆ เพราะมันคือ "เห็นล้ออยู่ใต้ตัวเขาไหม"
        # ไม่ใช่ของที่คำนวณจากช่องอื่นได้ (ต่างจาก age_group ที่มาจาก age_range)
        # ค่านอกชุดเป็น unknown เหมือนทุกช่อง ไม่เดาว่า "น่าจะเดิน"
        "mobility": _pick(data.get("mobility"), _MOBILITY),
        "mobility_confidence": _conf(data.get("mobility_confidence")),
        # 🔴 ชั้นที่สองของสวิตช์สัญชาติ · ปิดอยู่ = unknown เสมอ ไม่ว่าโมเดลตอบอะไรมา
        "nationality": (_pick(data.get("nationality"), _NATIONALITY)
                        if ALLOW_NATIONALITY else "unknown"),
        "nationality_confidence": (_conf(data.get("nationality_confidence"))
                                   if ALLOW_NATIONALITY else 0.0),
        "group": (GroupInfo(ref=_group_ref(data.get("group")))
                  if with_group else GroupInfo()),
        "emotion": _emotion(data.get("emotion")),
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
