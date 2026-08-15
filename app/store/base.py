"""ชั้นเก็บข้อมูล — interface เดียว สลับ backend ได้ด้วย env

🔴 ห้ามเรียก sqlite3 ตรงๆ จากที่อื่นนอกโฟลเดอร์นี้

POC ใช้ SQLite บน App Platform ซึ่ง **ไม่มีดิสก์ถาวร ไฟล์หายทุก deploy**
พอ POC จบแล้วอยากได้ของถาวร (Managed Postgres / Droplet+Volume) จะเปลี่ยนแค่
STORE_BACKEND โดยไม่ต้องรื้อโค้ดส่วนอื่น interface นี้คือสิ่งที่ทำให้เป็นไปได้
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..filters.temporal import HistoryItem


@dataclass
class FrameRecord:
    """หนึ่งเฟรมที่เข้ามา · บันทึกทันทีก่อนวิเคราะห์ แล้วค่อยเติมผลทีหลัง

    บันทึกก่อนวิเคราะห์เพราะถ้า LLM ล่มหรือช้า เราต้องยังมี log ว่าเฟรมนี้เคยมาถึง
    ไม่ใช่หายไปทั้งเหตุการณ์
    """

    request_id: str
    camera_id: str
    ts: float
    image_hash: str
    frame_w: int = 0
    frame_h: int = 0
    bit_depth: int = 8
    image_path: Optional[str] = None
    cv_result: Optional[str] = None        # JSON
    llm_raw_response: Optional[str] = None  # เก็บทั้งก้อน ไม่ตัด
    prompt_version: Optional[str] = None
    model_name: Optional[str] = None
    provider: Optional[str] = None
    finish_reason: Optional[str] = None
    completion_tokens: Optional[int] = None
    latency_ms: float = 0.0
    status: str = "received"
    raw_counts: Dict[str, int] = field(default_factory=dict)
    filtered_counts: Dict[str, int] = field(default_factory=dict)
    filter_reason: Optional[str] = None
    state: Optional[str] = None


@dataclass
class DetectionRecord:
    frame_request_id: str
    blob_id: int
    species: str
    confidence: float
    detection_confidence: float
    bbox: List[int]


class Store(ABC):
    @abstractmethod
    def init_schema(self) -> None: ...

    @abstractmethod
    def insert_frame(self, rec: FrameRecord) -> None:
        """เรียกทันทีที่รับเฟรม ก่อนทำ CV ก่อนยิง LLM"""

    @abstractmethod
    def update_frame(self, rec: FrameRecord) -> None:
        """เรียกหลังวิเคราะห์เสร็จ เติมผลลงแถวเดิม"""

    @abstractmethod
    def insert_detections(self, rows: List[DetectionRecord]) -> None: ...

    @abstractmethod
    def window(self, camera_id: str, now: float, span_s: int) -> List[HistoryItem]:
        """ดึงหน้าต่างตามเวลา ไม่ใช่ตามจำนวนเฟรม"""

    @abstractmethod
    def provisional_streak(self, camera_id: str) -> int:
        """นับเฟรมติดกันล่าสุดที่ยังไม่ commit · ใช้คุมเพดานรอ 2 เฟรม"""

    @abstractmethod
    def get_frame(self, request_id: str) -> Optional[dict]: ...

    @abstractmethod
    def save_truth(self, request_id: str, counts: Dict[str, int],
                   reviewer: Optional[str], comment: Optional[str]) -> bool: ...

    @abstractmethod
    def rollup_and_prune(self, retention_days: int) -> dict:
        """สรุปวันละแถวก่อนลบของดิบ

        เก็บดิบ 7 วันแปลว่าเดือนหน้าตอบไม่ได้ว่า 'เดือนที่แล้วกล้องนี้เจอช้างกี่ครั้ง'
        ซึ่งเป็นคำถามที่โครงการนี้จะโดนถามแน่ๆ · 1 กล้อง 1 ปี = 365 แถว ไม่ถึง 100 KB
        """


def make_store(backend: str, dsn: str) -> Store:
    if backend == "sqlite":
        from .sqlite_store import SQLiteStore

        return SQLiteStore(dsn)
    raise ValueError(
        f"ไม่รู้จัก STORE_BACKEND={backend!r} · รองรับ 'sqlite' "
        f"(postgres ยังไม่ได้เขียน ตั้งใจไว้ตอนเลิก POC)"
    )
