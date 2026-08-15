"""SQLite backend

🔴 บน DigitalOcean App Platform ไฟล์นี้หายทุก deploy (ตัดสินใจไว้แล้ว ทาง ค.)
ไม่ใช่บั๊ก · log การเรียก LLM ที่ต้องรอดข้าม deploy ไปอยู่ที่ LangSmith แทน
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from typing import Dict, List, Optional

from ..filters.temporal import HistoryItem
from .base import DetectionRecord, FrameRecord, Store

SCHEMA = """
CREATE TABLE IF NOT EXISTS frames (
  request_id       TEXT PRIMARY KEY,
  camera_id        TEXT NOT NULL,
  ts               REAL NOT NULL,
  image_hash       TEXT,
  image_path       TEXT,
  frame_w          INTEGER, frame_h INTEGER, bit_depth INTEGER,
  cv_result        TEXT,
  llm_raw_response TEXT,
  prompt_version   TEXT, model_name TEXT, provider TEXT,
  finish_reason    TEXT, completion_tokens INTEGER,
  latency_ms       REAL,
  status           TEXT,
  raw_counts       TEXT,
  filtered_counts  TEXT,
  filter_reason    TEXT,
  state            TEXT
);
-- หน้าต่าง 100 วิ query ตัวนี้ทุก request ต้องมี index
CREATE INDEX IF NOT EXISTS idx_frames_cam_ts ON frames(camera_id, ts);

CREATE TABLE IF NOT EXISTS detections (
  id                    INTEGER PRIMARY KEY AUTOINCREMENT,
  frame_request_id      TEXT NOT NULL,
  blob_id               INTEGER,
  species               TEXT,
  confidence            REAL,
  detection_confidence  REAL,
  bbox                  TEXT
);
CREATE INDEX IF NOT EXISTS idx_det_frame ON detections(frame_request_id);

CREATE TABLE IF NOT EXISTS truth (
  frame_request_id TEXT PRIMARY KEY,
  counts           TEXT, reviewer TEXT, comment TEXT, created REAL
);

-- ของดิบอยู่ 7 วัน ตัวนี้อยู่ตลอดไป · 1 กล้อง 1 ปี = 365 แถว
CREATE TABLE IF NOT EXISTS daily_rollup (
  day            TEXT, camera_id TEXT,
  frames_total   INTEGER, frames_with_animal INTEGER, frames_filtered INTEGER,
  max_count      INTEGER, sightings INTEGER,
  avg_confidence REAL, species_seen TEXT,
  first_seen     REAL, last_seen REAL,
  PRIMARY KEY (day, camera_id)
);
"""


class SQLiteStore(Store):
    def __init__(self, dsn: str):
        path = dsn.replace("sqlite:///", "")
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        self._path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # WAL: อ่านกับเขียนไม่บล็อกกัน · เขียนแค่ 0.1 ครั้ง/วิ/กล้อง ยังไงก็เหลือเฟือ
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

    def init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def insert_frame(self, rec: FrameRecord) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO frames (request_id, camera_id, ts, image_hash,"
                " image_path, frame_w, frame_h, bit_depth, status, raw_counts, filtered_counts)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (rec.request_id, rec.camera_id, rec.ts, rec.image_hash, rec.image_path,
                 rec.frame_w, rec.frame_h, rec.bit_depth, rec.status,
                 json.dumps(rec.raw_counts), json.dumps(rec.filtered_counts)),
            )
            self._conn.commit()

    def update_frame(self, rec: FrameRecord) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE frames SET image_path=?, frame_w=?, frame_h=?, bit_depth=?,"
                " cv_result=?, llm_raw_response=?, prompt_version=?, model_name=?, provider=?,"
                " finish_reason=?, completion_tokens=?, latency_ms=?, status=?,"
                " raw_counts=?, filtered_counts=?, filter_reason=?, state=?"
                " WHERE request_id=?",
                (rec.image_path, rec.frame_w, rec.frame_h, rec.bit_depth,
                 rec.cv_result, rec.llm_raw_response, rec.prompt_version, rec.model_name,
                 rec.provider, rec.finish_reason, rec.completion_tokens, rec.latency_ms,
                 rec.status, json.dumps(rec.raw_counts), json.dumps(rec.filtered_counts),
                 rec.filter_reason, rec.state, rec.request_id),
            )
            self._conn.commit()

    def insert_detections(self, rows: List[DetectionRecord]) -> None:
        if not rows:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT INTO detections (frame_request_id, blob_id, species, confidence,"
                " detection_confidence, bbox) VALUES (?,?,?,?,?,?)",
                [(r.frame_request_id, r.blob_id, r.species, r.confidence,
                  r.detection_confidence, json.dumps(r.bbox)) for r in rows],
            )
            self._conn.commit()

    def window(self, camera_id: str, now: float, span_s: int) -> List[HistoryItem]:
        """ตามเวลา ไม่ใช่ตามจำนวนเฟรม

        LIMIT 10 จะผิดตอนกล้องหลุดแล้วกลับมา เพราะจะไปหยิบของเมื่อ 4 นาทีที่แล้วมาใช้
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT request_id, ts, raw_counts FROM frames"
                " WHERE camera_id=? AND ts > ? AND status != 'received'"
                " ORDER BY ts ASC",
                (camera_id, now - span_s),
            ).fetchall()
            conf: Dict[str, Dict[str, float]] = {}
            if rows:
                q = ",".join("?" * len(rows))
                for d in self._conn.execute(
                    f"SELECT frame_request_id, species, MAX(confidence) c FROM detections"
                    f" WHERE frame_request_id IN ({q}) GROUP BY frame_request_id, species",
                    [r["request_id"] for r in rows],
                ):
                    conf.setdefault(d["frame_request_id"], {})[d["species"]] = d["c"]

        out = []
        for r in rows:
            try:
                counts = json.loads(r["raw_counts"] or "{}")
            except json.JSONDecodeError:
                counts = {}
            out.append(HistoryItem(ts=r["ts"], counts=counts,
                                   confidence=conf.get(r["request_id"], {})))
        return out

    def provisional_streak(self, camera_id: str) -> int:
        with self._lock:
            rows = self._conn.execute(
                "SELECT status FROM frames WHERE camera_id=? AND status != 'received'"
                " ORDER BY ts DESC LIMIT 5", (camera_id,)).fetchall()
        n = 0
        for r in rows:
            if r["status"] == "provisional":
                n += 1
            else:
                break
        return n

    def get_frame(self, request_id: str) -> Optional[dict]:
        with self._lock:
            r = self._conn.execute("SELECT * FROM frames WHERE request_id=?",
                                   (request_id,)).fetchone()
            if not r:
                return None
            dets = self._conn.execute(
                "SELECT blob_id, species, confidence, detection_confidence, bbox"
                " FROM detections WHERE frame_request_id=?", (request_id,)).fetchall()
            t = self._conn.execute("SELECT counts, reviewer, comment FROM truth"
                                   " WHERE frame_request_id=?", (request_id,)).fetchone()
        out = dict(r)
        out["detections"] = [dict(d) for d in dets]
        out["truth"] = dict(t) if t else None
        return out

    def save_truth(self, request_id: str, counts: Dict[str, int],
                   reviewer: Optional[str], comment: Optional[str]) -> bool:
        import time

        with self._lock:
            if not self._conn.execute("SELECT 1 FROM frames WHERE request_id=?",
                                      (request_id,)).fetchone():
                return False
            self._conn.execute(
                "INSERT OR REPLACE INTO truth VALUES (?,?,?,?,?)",
                (request_id, json.dumps(counts), reviewer, comment, time.time()))
            self._conn.commit()
        return True

    def rollup_and_prune(self, retention_days: int) -> dict:
        import time

        cutoff = time.time() - retention_days * 86400
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO daily_rollup"
                " SELECT date(ts,'unixepoch','localtime') AS day, camera_id,"
                "   COUNT(*), SUM(CASE WHEN raw_counts NOT IN ('{}','') THEN 1 ELSE 0 END),"
                "   SUM(CASE WHEN state='falling' OR status='provisional' THEN 1 ELSE 0 END),"
                "   0, 0, 0.0, '', MIN(ts), MAX(ts)"
                " FROM frames GROUP BY day, camera_id")
            n = self._conn.execute("SELECT COUNT(*) c FROM frames WHERE ts < ?",
                                   (cutoff,)).fetchone()["c"]
            self._conn.execute(
                "DELETE FROM detections WHERE frame_request_id IN"
                " (SELECT request_id FROM frames WHERE ts < ?)", (cutoff,))
            self._conn.execute("DELETE FROM frames WHERE ts < ?", (cutoff,))
            self._conn.commit()
        # ไม่ VACUUM โดยตั้งใจ · SQLite เอาหน้าว่างกลับมาใช้ใหม่ ไฟล์จะนิ่งราว 30-40 MB
        return {"pruned_frames": n, "retention_days": retention_days}
