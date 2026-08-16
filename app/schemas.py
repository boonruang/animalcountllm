"""สัญญาข้อมูลทั้งหมดของบริการ อยู่ไฟล์เดียว

ทุกอย่างที่ข้ามขอบเขต (HTTP เข้า, HTTP ออก, LLM ตอบกลับ) ต้องผ่าน model ในนี้
ไม่มี dict ลอยๆ ข้ามชั้น เพราะ dict ลอยๆ คือที่ที่ field หายไปเงียบๆ
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# ชนิดสัตว์ที่ระบบยอมรับ · unknown ต้องมีเสมอ
# ก้อนที่ระบุไม่ได้ไปอยู่ที่ unknown ห้ามยัดเป็น elephant เพื่อให้ตัวเลขสวย
Species = Literal["elephant", "cattle", "human", "other_animal", "unknown"]

Status = Literal["ok", "provisional", "degraded", "error"]
State = Literal["stable", "rising", "falling", "unstable", "unconfirmed", "cold_start"]


# ---------------------------------------------------------------- request
class FrameIn(BaseModel):
    camera_id: str = Field(..., min_length=1, max_length=64)
    image_base64: str
    ts: Optional[str] = None  # ISO 8601; ไม่ส่งมา = ใช้เวลาที่เซิร์ฟเวอร์รับ
    note: Optional[str] = None


class TruthIn(BaseModel):
    """ground truth ที่คนตรวจแล้ว · ไม่มีข้อมูลนี้ = calibrate confidence ไม่ได้ตลอดกาล"""

    counts: Dict[str, int]
    reviewer: Optional[str] = None
    comment: Optional[str] = None


# ---------------------------------------------------------------- CV layer
class Blob(BaseModel):
    """ก้อนร้อนหนึ่งก้อนที่ชั้น CV วัดได้ ยังไม่รู้ว่าเป็นสัตว์อะไร

    ทุก field ในนี้ 'วัดได้' ไม่ใช่ 'ตีความ' รันภาพเดิมซ้ำได้ค่าเดิมเป๊ะ
    """

    id: int
    bbox: List[int]  # [x, y, w, h]
    area_px: int
    aspect: float  # กว้าง/สูง
    fill_ratio: float  # area_px / (w*h) — ก้อนตันหรือกลวง
    mean_intensity: float
    contrast: float  # mean ของก้อน ลบ mean ของพื้นหลัง (สเกล 0-1)
    touches_edge: bool
    detection_confidence: float  # วัดได้ อธิบายได้ ไม่ใช่เลขจาก LLM


class CVResult(BaseModel):
    blobs: List[Blob]
    frame_w: int
    frame_h: int
    bit_depth: int  # 8 หรือ 16 · radiometric หรือ palette
    background_level: float
    threshold_used: float
    elapsed_ms: float

    @property
    def has_candidates(self) -> bool:
        """ไม่มีก้อนขนาดสัตว์เลย = ไม่ต้องจ่ายเงินถาม LLM"""
        return len(self.blobs) > 0


# ---------------------------------------------------------------- LLM layer
class SpeciesTally(BaseModel):
    """คำตอบของ VLM ต่อหนึ่งเฟรม หนึ่งชนิด · v2 ตอบรายชนิด ไม่ใช่รายก้อน

    ยาวเท่าเดิมเสมอไม่ว่าเฟรมจะรกแค่ไหน ต่างจาก v1 ที่ยาวตามจำนวนก้อน
    """

    species: Species
    count: int = Field(..., ge=0)
    confidence: float = Field(..., ge=0.0, le=1.0)


class SpeciesCall(BaseModel):
    """คำตอบของ VLM ต่อก้อนหนึ่งก้อน · schema นี้บังคับผ่าน structured output"""

    blob_id: int
    species: Species
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str = Field("", max_length=200)


class LLMVerdict(BaseModel):
    animals: List[SpeciesTally] = Field(default_factory=list)
    calls: List[SpeciesCall] = Field(default_factory=list)   # v1 เก่า ยังไม่ลบ
    merged_blob_ids: List[List[int]] = Field(default_factory=list)
    """ก้อนที่ CV แบ่งผิด LLM บอกว่าจริงๆ เป็นตัวเดียวกัน เช่น [[2,3]]"""


# ---------------------------------------------------------------- response
class Detection(BaseModel):
    """ก้อนร้อนหนึ่งก้อน · v2 เป็นข้อมูลตำแหน่งล้วน ไม่มีชนิดรายก้อน

    ชนิดสัตว์ย้ายไปอยู่ระดับเฟรมที่ RawResult.animals ตามที่ Toy ขอ
    (สัตว์อะไร กี่ตัว มั่นใจเท่าไร) เพราะรายก้อนทั้งยาวและตอบไม่ได้จริงอยู่แล้ว
    """

    id: int
    bbox: List[int]
    area_px: int
    aspect: float
    detection_confidence: float  # จาก CV · วัดได้ อธิบายได้


class RawResult(BaseModel):
    animals: List[SpeciesTally]        # ← คำตอบหลัก: สัตว์อะไร กี่ตัว มั่นใจเท่าไร
    counts: Dict[str, int]             # รูปแบบ dict ของอันบน อ่านง่ายฝั่งโปรแกรม
    regions_detected: int              # จำนวนก้อนร้อนที่ CV วัดได้ ใช้ตรวจสอบย้อนหลัง
    detections: List[Detection]        # ตำแหน่งของแต่ละก้อน
    overall_confidence: float


class FilteredResult(BaseModel):
    counts: Dict[str, int]
    confidence: float
    method: str
    accepted: bool
    reason: str
    state: State
    corroborated_frames: int = 1


class WindowInfo(BaseModel):
    span_seconds: int
    frames_used: int
    frames_expected: int
    complete: bool
    median: Dict[str, float]
    mad: Dict[str, float]


class ModelInfo(BaseModel):
    provider: str
    name: str
    prompt_version: str
    finish_reason: Optional[str] = None
    completion_tokens: Optional[int] = None


class FrameOut(BaseModel):
    request_id: str
    camera_id: str
    received_at: str
    status: Status
    raw: RawResult
    filtered: FilteredResult
    window: WindowInfo
    model: ModelInfo
    timing_ms: Dict[str, float]
