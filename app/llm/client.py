"""ตัวเรียก VLM · LM Studio ตอน dev · OpenRouter ตอน prod · สลับด้วย env ตัวเดียว

ทั้งสองฝั่งเป็น OpenAI-compatible เลยใช้ทางเดียวกันได้ ไม่ต้องมี if provider== ทั่วโค้ด

🔴 ไม่ใช้ agent ไม่ใช้ memory ไม่ใช้ chain ซ้อน · งานนี้เป็น pipeline เส้นตรง
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..schemas import Blob, LLMVerdict, SpeciesCall
from . import prompt_v1

VALID = {"elephant", "cattle", "human", "other_animal", "unknown"}


@dataclass
class LLMResult:
    verdict: Optional[LLMVerdict]
    raw: str
    finish_reason: Optional[str]
    completion_tokens: Optional[int]
    latency_ms: float
    error: Optional[str] = None

    @property
    def usable(self) -> bool:
        """finish_reason='length' = มันคิดจนหมดโควตา ไม่ได้ตอบจบ

        วัดจริง 2026-08-16: ปล่อย 1500 token แล้วมันคิดวนจนหลุดไปเขียนโค้ด cv2
        ใส่ช่องคำตอบ · คำตอบที่ตัดกลางคันเชื่อไม่ได้ ต้องตี degraded ไม่ใช่รับมาใช้เฉยๆ
        """
        return self.verdict is not None and self.finish_reason != "length"


class VisionLLM:
    def __init__(self) -> None:
        self.provider = os.environ.get("LLM_PROVIDER", "lmstudio")
        self.base_url = os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1")
        self.model = os.environ.get("LLM_MODEL", "qwen3.6-35b-a3b")
        self.api_key = os.environ.get("LLM_API_KEY", "not-needed")
        self.max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "200"))
        self.timeout = float(os.environ.get("LLM_TIMEOUT_S", "25"))
        # 🔴 ปิด thinking · วัดจริง 2026-08-16 บน qwen3.7-flash ผ่าน OpenRouter
        #   ไม่ตั้งอะไร        13.0 วิ · 801 tok · finish=length ตอบไม่จบ
        #   reasoning enabled=false  2.9 วิ ·  86 tok · finish=stop  JSON ครบ
        # เร็วขึ้น 4.5 เท่า · output token ลด 9 เท่า ซึ่งคือค่าใช้จ่ายเกือบทั้งหมด
        # ($3.29 -> $0.89 ต่อเดือนต่อกล้อง)
        #
        # ที่ลองแล้วไม่ได้ผล อย่าเสียเวลาลองซ้ำ:
        #   reasoning.effort=low            ไม่ต่างจากไม่ตั้งเลย
        #   chat_template enable_thinking   OpenRouter ไม่ส่งต่อให้ provider
        #   reasoning.max_tokens=0          provider ตอบ 400
        #   reasoning.exclude=true          🔴 กับดัก · ซ่อนความคิดจาก response
        #                                   แต่ยังคิดจริงและยังคิดเงินเต็ม
        self.no_reasoning = os.environ.get("LLM_DISABLE_REASONING", "true").lower() == "true"
        self.sample_rate = float(os.environ.get("LANGSMITH_SAMPLE_RATE", "0.05"))
        self.tracing = os.environ.get("LANGSMITH_TRACING", "false").lower() == "true"

    # ---------------------------------------------------------------- tracing
    def _should_trace(self, anomalous: bool) -> bool:
        """เฟรมผิดปกติเก็บ 100% เฟรมปกติเก็บตาม sample rate

        860 trace/วัน/กล้อง = 26,000/เดือน ทะลุแพ็กฟรี (~5,000) ตั้งแต่เดือนแรก
        ของที่จะกลับมาดูจริงคือเฟรมที่มีปัญหา ไม่ใช่เฟรมที่ถูกอยู่แล้ว
        """
        if not self.tracing:
            return False
        return True if anomalous else random.random() < self.sample_rate

    # ---------------------------------------------------------------- call
    def classify(self, image_b64: str, blobs: List[Blob], w: int, h: int,
                 image_hash: str = "") -> LLMResult:
        system, user = prompt_v1.build(blobs, w, h)
        t0 = time.perf_counter()

        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage, SystemMessage

            kw = {}
            if self.no_reasoning:
                # LM Studio ไม่รู้จัก field นี้แต่ไม่ error มันข้ามไปเฉยๆ
                kw["extra_body"] = {"reasoning": {"enabled": False}}
            llm = ChatOpenAI(
                base_url=self.base_url, api_key=self.api_key, model=self.model,
                max_tokens=self.max_tokens, temperature=0, timeout=self.timeout, **kw,
            )
            msgs = [
                SystemMessage(content=system),
                HumanMessage(content=[
                    {"type": "text", "text": user},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ]),
            ]
            # metadata ไปที่ LangSmith · 🔴 ไม่ส่งภาพ base64 เข้า trace
            # payload จะบวมและกินโควตา storage เร็วกว่ากินโควตา trace
            # ผลพลอยได้: ภาพจากไซต์อนุรักษ์ไม่ต้องไปนอนบนเซิร์ฟเวอร์เจ้าอื่น
            resp = llm.invoke(msgs, config={
                "metadata": {"image_hash": image_hash, "blobs": len(blobs),
                             "frame": f"{w}x{h}", "prompt_version": prompt_v1.PROMPT_VERSION},
                "run_name": "classify_thermal_frame",
            })
            dt = (time.perf_counter() - t0) * 1000
            meta = resp.response_metadata or {}
            usage = getattr(resp, "usage_metadata", None) or {}
            text = resp.content if isinstance(resp.content, str) else str(resp.content)
            verdict, err = parse(text, {b.id for b in blobs})
            return LLMResult(
                verdict=verdict, raw=text,
                finish_reason=meta.get("finish_reason"),
                completion_tokens=usage.get("output_tokens"),
                latency_ms=round(dt, 1), error=err,
            )
        except Exception as e:
            # เน็ตหลุด / LLM ช้า / โมเดลไม่มี → ตอบ degraded พร้อมผล CV ไม่ค้างรอ
            return LLMResult(None, "", None, None,
                             round((time.perf_counter() - t0) * 1000, 1),
                             f"{type(e).__name__}: {e}")


def parse(text: str, known_ids: set) -> Tuple[Optional[LLMVerdict], Optional[str]]:
    """แกะ JSON จากคำตอบ ทนกับ markdown fence และข้อความนำ

    โมเดลสาย thinking ชอบห่อ JSON ด้วย ```json แม้จะสั่งว่าอย่าห่อ
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

    calls = []
    for c in data.get("calls", []):
        try:
            bid = int(c["blob_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if bid not in known_ids:
            continue  # โมเดลกุ region ที่ CV ไม่ได้เจอ ทิ้ง
        sp = str(c.get("species", "unknown")).lower().strip()
        if sp not in VALID:
            sp = "unknown"
        try:
            conf = float(c.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        calls.append(SpeciesCall(blob_id=bid, species=sp,
                                 confidence=max(0.0, min(1.0, conf)),
                                 reason=str(c.get("reason", ""))[:200]))

    # ก้อนที่โมเดลไม่ตอบถึง ต้องเป็น unknown ไม่ใช่หายไปเฉยๆ
    answered = {c.blob_id for c in calls}
    for bid in sorted(known_ids - answered):
        calls.append(SpeciesCall(blob_id=bid, species="unknown", confidence=0.0,
                                 reason="model did not answer for this region"))

    merged = [[int(i) for i in g] for g in data.get("merged_ids", [])
              if isinstance(g, list) and len(g) > 1]
    return LLMVerdict(calls=calls, merged_blob_ids=merged), None
