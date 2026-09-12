"""ที่เก็บผลของงานคน · SQLite ไฟล์เดียว คนละไฟล์กับฝั่งช้าง

🔴 บน App Platform ไฟล์นี้หายทุก deploy เหมือนฝั่งช้าง (ทาง ค. ที่เลือกไว้แล้ว)
มันไม่ใช่ระบบบันทึกถาวร มันคือที่ให้ยิง GET กลับมาดูผลของ request ที่เพิ่งส่งไป
และให้ไล่ดูว่าคำตอบดิบของโมเดลหน้าตายังไงตอนผลเพี้ยน

🔴 ค่าเริ่มต้นคือ **ไม่เก็บภาพ** (PEOPLE_SAVE_IMAGES=none)
ฝั่งช้างเก็บ "เฟรมที่น่าสนใจ" ได้สบายเพราะในภาพมีสัตว์ · ที่นี่ในภาพมีหน้าคน
ที่เดินเข้าอาคาร เก็บไว้บนดิสก์ที่ใครก็ไม่ได้ดูแลคือการสร้างภาระที่ไม่มีใครขอ
เปิดได้ถ้าต้องดีบัก แต่ต้องเป็นการตัดสินใจที่ตั้งใจ ไม่ใช่ค่าที่ติดมาเอง
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS person_requests (
  request_id  TEXT PRIMARY KEY,
  camera_id   TEXT NOT NULL,
  ts          REAL NOT NULL,
  objects_in  INTEGER,
  client_request_id TEXT,
  status      TEXT,
  latency_ms  REAL,
  note        TEXT
);
CREATE INDEX IF NOT EXISTS idx_preq_cam_ts ON person_requests(camera_id, ts);
-- 🔴 index ของ client_request_id สร้างใน _migrate() ไม่ใช่ที่นี่
-- ตารางเก่ายังไม่มีคอลัมน์นั้น ถ้าสร้าง index ตรงนี้จะพังตั้งแต่ executescript
-- ก่อน ALTER TABLE จะได้ทำงาน แล้วทั้งฐานเปิดไม่ขึ้น (เจอจริง 2026-09-11)

CREATE TABLE IF NOT EXISTS persons (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  request_id        TEXT NOT NULL,
  object_id         TEXT NOT NULL,
  camera_id         TEXT NOT NULL,
  ts                REAL NOT NULL,
  status            TEXT,
  direction         TEXT, direction_confidence REAL,
  gender            TEXT, gender_confidence REAL,
  age_range         TEXT, age_range_confidence REAL,
  age_group         TEXT,          -- คำนวณจาก age_range ไม่ได้ถามโมเดล
  nationality       TEXT, nationality_confidence REAL,
  emotion_label     TEXT, emotion_valence INTEGER, emotion_confidence REAL,
  group_ref         TEXT, group_size INTEGER, group_type TEXT,
  -- uniform_kind / uniform_text ซ้ำกับที่อยู่ใน appearance JSON โดยตั้งใจ
  -- สองช่องนี้คือของที่ปลายทางจะ GROUP BY จริง ("เดือนนี้ รปภ. เข้ากี่คน")
  -- ขุดออกจาก JSON ทุกครั้งที่ query = ช้าและเขียน SQL ยาก
  -- ที่เหลือของ appearance ไม่มีใครถาม เลยไม่แตกทั้งก้อน
  uniform_kind      TEXT, uniform_text TEXT,
  appearance        TEXT,          -- JSON ทั้งก้อน
  appearance_confidence REAL,
  description       TEXT,
  overall_confidence REAL,
  reason            TEXT,
  image_w           INTEGER, image_h INTEGER, image_source TEXT,
  image_path        TEXT,          -- ปกติ NULL ดูหัวไฟล์
  llm_raw_response  TEXT,
  prompt_version    TEXT, model_name TEXT, provider TEXT,
  finish_reason     TEXT, completion_tokens INTEGER,
  latency_ms        REAL
);
CREATE INDEX IF NOT EXISTS idx_persons_req ON persons(request_id);
CREATE INDEX IF NOT EXISTS idx_persons_cam_ts ON persons(camera_id, ts);

CREATE TABLE IF NOT EXISTS person_truth (
  request_id TEXT NOT NULL,
  object_id  TEXT NOT NULL,
  gender     TEXT, age_range TEXT, comment TEXT, reviewer TEXT, created REAL,
  PRIMARY KEY (request_id, object_id)
);
"""


class PersonStore:
    def __init__(self, dsn: str):
        path = dsn.replace("sqlite:///", "")
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        self._path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

    def init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._migrate()
            self._conn.commit()

    def _migrate(self) -> None:
        """เติมคอลัมน์ที่เพิ่มทีหลังให้ตารางเก่า

        🔴 `CREATE TABLE IF NOT EXISTS` ไม่แตะตารางที่มีอยู่แล้ว ตารางเก่าจึงยังขาด
        คอลัมน์ใหม่ แล้ว INSERT พังทุกแถวด้วย "no such column" ซึ่งบนเครื่อง dev
        จะเจอทันที แต่บน App Platform จะไม่เจอเลยเพราะไฟล์หายทุก deploy อยู่แล้ว
        **บั๊กที่ prod ไม่มีวันเจอแต่ dev เจอตลอด คือบั๊กที่จะถูกมองข้าม**
        เลยเติมให้เองตรงนี้ ไม่ต้องรอให้ใครไปลบไฟล์ทิ้ง
        """
        for table, cols in (
            ("persons", (("direction", "TEXT"), ("direction_confidence", "REAL"),
                         # 2026-09-12 · demographic + uniform + emotion
                         ("age_group", "TEXT"),
                         ("nationality", "TEXT"),
                         ("nationality_confidence", "REAL"),
                         ("emotion_label", "TEXT"),
                         ("emotion_valence", "INTEGER"),
                         ("emotion_confidence", "REAL"),
                         ("group_ref", "TEXT"), ("group_size", "INTEGER"),
                         ("group_type", "TEXT"),
                         ("uniform_kind", "TEXT"), ("uniform_text", "TEXT"))),
            ("person_requests", (("client_request_id", "TEXT"),)),
        ):
            have = {r["name"] for r in
                    self._conn.execute(f"PRAGMA table_info({table})").fetchall()}
            for col, decl in cols:
                if col not in have:
                    self._conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
                    print(f"[people] migrate: เพิ่มคอลัมน์ {table}.{col}", flush=True)
        # index ต้องมาหลังคอลัมน์ ไม่ใช่ก่อน
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_preq_client_ref"
                           " ON person_requests(client_request_id, ts)")

    # ------------------------------------------------------------ write
    def insert_request(self, request_id: str, camera_id: str, ts: float,
                       objects_in: int, note: Optional[str],
                       client_request_id: Optional[str] = None) -> None:
        """บันทึกก่อนยิงโมเดล · โมเดลล่มหรือ container ตายกลางทาง
        ก็ยังต้องมีร่องรอยว่า request นี้เคยมาถึง ไม่ใช่หายเงียบ"""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO person_requests"
                " (request_id, camera_id, ts, objects_in, status, note,"
                " client_request_id) VALUES (?,?,?,?,?,?,?)",
                (request_id, camera_id, ts, objects_in, "received", note,
                 client_request_id))
            self._conn.commit()

    def finish_request(self, request_id: str, status: str, latency_ms: float) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE person_requests SET status=?, latency_ms=? WHERE request_id=?",
                (status, latency_ms, request_id))
            self._conn.commit()

    def insert_persons(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        cols = ("request_id", "object_id", "camera_id", "ts", "status",
                "direction", "direction_confidence",
                "gender", "gender_confidence", "age_range", "age_range_confidence",
                "age_group", "nationality", "nationality_confidence",
                "emotion_label", "emotion_valence", "emotion_confidence",
                "group_ref", "group_size", "group_type",
                "uniform_kind", "uniform_text",
                "appearance", "appearance_confidence", "description",
                "overall_confidence", "reason", "image_w", "image_h", "image_source",
                "image_path", "llm_raw_response", "prompt_version", "model_name",
                "provider", "finish_reason", "completion_tokens", "latency_ms")
        with self._lock:
            self._conn.executemany(
                f"INSERT INTO persons ({','.join(cols)})"
                f" VALUES ({','.join('?' * len(cols))})",
                [tuple(r.get(c) for c in cols) for r in rows])
            self._conn.commit()

    def save_truth(self, request_id: str, object_id: str, gender: Optional[str],
                   age_range: Optional[str], reviewer: Optional[str],
                   comment: Optional[str]) -> bool:
        with self._lock:
            hit = self._conn.execute(
                "SELECT 1 FROM persons WHERE request_id=? AND object_id=?",
                (request_id, object_id)).fetchone()
            if not hit:
                return False
            self._conn.execute(
                "INSERT OR REPLACE INTO person_truth"
                " (request_id, object_id, gender, age_range, comment, reviewer, created)"
                " VALUES (?,?,?,?,?,?,?)",
                (request_id, object_id, gender, age_range, comment, reviewer, time.time()))
            self._conn.commit()
            return True

    # ------------------------------------------------------------ read
    def get_request(self, request_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            head = self._conn.execute(
                "SELECT * FROM person_requests WHERE request_id=?",
                (request_id,)).fetchone()
            if not head:
                return None
            rows = self._conn.execute(
                "SELECT * FROM persons WHERE request_id=? ORDER BY id",
                (request_id,)).fetchall()
        out = dict(head)
        persons = []
        for r in rows:
            d = dict(r)
            if d.get("appearance"):
                try:
                    d["appearance"] = json.loads(d["appearance"])
                except json.JSONDecodeError:
                    pass  # เก็บดิบไว้ดีกว่าทิ้ง คนอ่านยังแกะเองได้
            persons.append(d)
        out["persons"] = persons
        return out

    def by_client_ref(self, client_request_id: str) -> List[Dict[str, Any]]:
        """หา request จากเลขอ้างอิงของปลายทาง · ใหม่สุดก่อน

        ดูทั้งคอลัมน์ `client_request_id` และ `note` เพราะปลายทางส่งเลขอ้างอิง
        มาในช่อง note ได้ (และทีมคุณสุชาติจะส่งมาแบบนั้น)
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT request_id, ts, status FROM person_requests"
                " WHERE client_request_id=? OR note=? ORDER BY ts DESC LIMIT 20",
                (client_request_id, client_request_id)).fetchall()
        return [dict(r) for r in rows]

    def recent(self, camera_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT object_id, ts, status, direction, gender, age_range,"
                " description, overall_confidence FROM persons WHERE camera_id=?"
                " ORDER BY ts DESC LIMIT ?", (camera_id, int(limit))).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------ prune
    def prune(self, days: int) -> Dict[str, int]:
        """ลบของที่เกินอายุ · ต้องเรียกจาก cron ข้างนอก ไม่ใช่ background task

        container ตายเมื่อไหร่ก็ได้ background task ที่ตายไปกับมันคือการเก็บของ
        ไว้ตลอดกาลโดยที่ทุกคนเชื่อว่ามีตัวลบอยู่
        """
        cutoff = time.time() - days * 86400
        with self._lock:
            p = self._conn.execute("DELETE FROM persons WHERE ts < ?", (cutoff,)).rowcount
            q = self._conn.execute("DELETE FROM person_requests WHERE ts < ?",
                                   (cutoff,)).rowcount
            self._conn.commit()
        return {"persons_deleted": p, "requests_deleted": q, "older_than_days": days}
