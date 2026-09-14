"""SQLite ของงานอ่านป้ายทะเบียน · ไฟล์แยกจากงานช้างและงานคน

🔴 ทาง ค. เหมือนอีกสองงาน: ไฟล์อยู่บน App Platform ซึ่ง **หายทุก deploy**
ยอมรับข้อนี้ตั้งแต่วันแรกสำหรับ POC · ของที่ห้ามหายคือของที่ปลายทางถืออยู่แล้ว
(เขาได้ JSON กลับไปทุกครั้ง) ส่วนที่นี่คือของสำหรับไล่ย้อนหลังและวัดความแม่น

🔴 ตารางนี้เก็บทะเบียนรถของคนจริง **ซึ่งชี้ไปหาเจ้าของได้ผ่านฐานทะเบียนกรมขนส่ง**
ต่างจากงานบรรยายคนที่เก็บแค่ "เสื้อขาว กางเกงดำ" ซึ่งชี้ตัวใครไม่ได้
- ไม่เก็บภาพเป็นค่าเริ่มต้น (`LPR_SAVE_IMAGES=none`)
- มี `prune` ให้เรียกจาก cron ข้างนอก และควรตั้งให้สั้นกว่างานอื่น
- ห้ามเพิ่มตาราง "รถที่เคยเห็น" หรือ index ที่ทำให้ค้นย้อนหลังด้วยเลขทะเบียนได้ง่าย
  ในนี้ นั่นคือการสร้างระบบติดตามรถ ซึ่งไม่ใช่สิ่งที่ใครสั่งและไม่ใช่สิ่งที่
  บริการนี้เป็น (ดูกติกาข้อแรกใน CLAUDE.md)
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS lpr_requests (
  request_id  TEXT PRIMARY KEY,
  camera_id   TEXT NOT NULL,
  ts          REAL NOT NULL,
  client_request_id TEXT,
  status      TEXT,
  latency_ms  REAL,
  note        TEXT
);
CREATE INDEX IF NOT EXISTS idx_lreq_cam_ts ON lpr_requests(camera_id, ts);

CREATE TABLE IF NOT EXISTS vehicles (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  request_id        TEXT NOT NULL,
  ref               TEXT NOT NULL,
  camera_id         TEXT NOT NULL,
  ts                REAL NOT NULL,
  status            TEXT,
  plate_text        TEXT,          -- ที่ประกอบกลับแล้ว ว่าง = แยกไม่ออก
  plate_text_raw    TEXT,          -- ที่โมเดลอ่านมาดิบๆ หลักฐาน
  plate_prefix      TEXT, plate_letters TEXT, plate_digits TEXT,
  plate_color       TEXT, plate_confidence REAL,
  province          TEXT, province_confidence REAL,
  vehicle_type      TEXT, vehicle_type_confidence REAL,
  vehicle_color     TEXT,
  description       TEXT,
  overall_confidence REAL,
  reason            TEXT,
  where_text        TEXT,
  image_w           INTEGER, image_h INTEGER, image_source TEXT,
  llm_raw_response  TEXT,
  prompt_version    TEXT, model_name TEXT, provider TEXT,
  finish_reason     TEXT, completion_tokens INTEGER,
  latency_ms        REAL
);
CREATE INDEX IF NOT EXISTS idx_vehicles_req ON vehicles(request_id);
CREATE INDEX IF NOT EXISTS idx_vehicles_cam_ts ON vehicles(camera_id, ts);

CREATE TABLE IF NOT EXISTS lpr_truth (
  request_id TEXT NOT NULL,
  ref        TEXT NOT NULL,
  plate      TEXT, province TEXT, vehicle_type TEXT,
  comment    TEXT, reviewer TEXT, created REAL,
  PRIMARY KEY (request_id, ref)
);
"""


class PlateStore:
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

    @property
    def path(self) -> str:
        return self._path

    def init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._migrate()
            self._conn.commit()

    def _migrate(self) -> None:
        """เติมคอลัมน์ที่เพิ่มทีหลังให้ตารางเก่า

        🔴 **index ต้องมาหลังคอลัมน์เสมอ** `CREATE TABLE IF NOT EXISTS` ข้ามตารางเก่า
        แต่ `CREATE INDEX` ไม่ข้ามให้ · ฝั่งงานคนเคยสร้าง index บนคอลัมน์ที่
        `_migrate()` ยังไม่ทันเติม แล้ว **ฐานข้อมูลเก่าเปิดไม่ขึ้นทั้งไฟล์**
        (2026-09-12) · ตอนนี้ยังไม่มีคอลัมน์ที่ต้อง migrate เพราะเป็นเวอร์ชันแรก
        แต่โครงอยู่ตรงนี้แล้ว เติมต่อได้เลยโดยไม่ต้องคิดใหม่
        """
        for table, cols in (("vehicles", ()), ("lpr_requests", ())):
            if not cols:
                continue
            have = {r["name"] for r in
                    self._conn.execute(f"PRAGMA table_info({table})").fetchall()}
            for col, decl in cols:
                if col not in have:
                    self._conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
                    print(f"[lpr] migrate: เพิ่มคอลัมน์ {table}.{col}", flush=True)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_lreq_client_ref"
                           " ON lpr_requests(client_request_id, ts)")

    # ------------------------------------------------------------ write
    def insert_request(self, request_id: str, camera_id: str, ts: float,
                       note: Optional[str], client_request_id: Optional[str]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO lpr_requests"
                " (request_id, camera_id, ts, client_request_id, status, note)"
                " VALUES (?,?,?,?,?,?)",
                (request_id, camera_id, ts, client_request_id, "pending", note))
            self._conn.commit()

    def finish_request(self, request_id: str, status: str, latency_ms: float) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE lpr_requests SET status=?, latency_ms=? WHERE request_id=?",
                (status, latency_ms, request_id))
            self._conn.commit()

    def insert_vehicles(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        cols = ("request_id", "ref", "camera_id", "ts", "status",
                "plate_text", "plate_text_raw", "plate_prefix", "plate_letters",
                "plate_digits", "plate_color", "plate_confidence",
                "province", "province_confidence",
                "vehicle_type", "vehicle_type_confidence", "vehicle_color",
                "description", "overall_confidence", "reason", "where_text",
                "image_w", "image_h", "image_source", "llm_raw_response",
                "prompt_version", "model_name", "provider", "finish_reason",
                "completion_tokens", "latency_ms")
        with self._lock:
            self._conn.executemany(
                f"INSERT INTO vehicles ({','.join(cols)})"
                f" VALUES ({','.join('?' * len(cols))})",
                [tuple(r.get(c) for c in cols) for r in rows])
            self._conn.commit()

    def save_truth(self, request_id: str, ref: str, plate: Optional[str],
                   province: Optional[str], vehicle_type: Optional[str],
                   reviewer: Optional[str], comment: Optional[str]) -> bool:
        with self._lock:
            hit = self._conn.execute(
                "SELECT 1 FROM vehicles WHERE request_id=? AND ref=?",
                (request_id, ref)).fetchone()
            if not hit:
                return False
            self._conn.execute(
                "INSERT OR REPLACE INTO lpr_truth"
                " (request_id, ref, plate, province, vehicle_type, comment,"
                "  reviewer, created) VALUES (?,?,?,?,?,?,?,?)",
                (request_id, ref, plate, province, vehicle_type, comment,
                 reviewer, time.time()))
            self._conn.commit()
            return True

    # ------------------------------------------------------------ read
    def get_request(self, request_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            head = self._conn.execute(
                "SELECT * FROM lpr_requests WHERE request_id=?",
                (request_id,)).fetchone()
            if not head:
                return None
            rows = self._conn.execute(
                "SELECT * FROM vehicles WHERE request_id=? ORDER BY id",
                (request_id,)).fetchall()
        return {"request": dict(head), "vehicles": [dict(r) for r in rows]}

    def by_client_ref(self, client_request_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT request_id FROM lpr_requests WHERE client_request_id=?"
                " ORDER BY ts DESC", (client_request_id,)).fetchall()
        return [dict(r) for r in rows]

    def accuracy(self) -> Dict[str, int]:
        """อ่านถูกกี่แถวจากแถวที่มีคนตรวจแล้ว

        🔴 **งานนี้วัดความแม่นได้จริง ต่างจากงานอารมณ์และงานสัญชาติ**
        ป้ายทะเบียนมีคำตอบเดียวและคนเปิดภาพดูก็ตัดสินได้ · ตัวเลขจากตรงนี้
        คือสิ่งเดียวที่ทำให้ตอบทีมปลายทางได้ว่าบริการนี้ใช้ได้จริงหรือยัง
        ไม่ใช่ `plate.confidence` ซึ่งเป็นความมั่นใจของโมเดล ไม่ใช่ความถูกต้อง
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT v.plate_text, t.plate FROM lpr_truth t"
                " JOIN vehicles v ON v.request_id=t.request_id AND v.ref=t.ref"
                " WHERE t.plate IS NOT NULL AND t.plate != ''").fetchall()
        exact = sum(1 for r in rows if (r["plate_text"] or "") == r["plate"])
        blank = sum(1 for r in rows if not (r["plate_text"] or ""))
        return {"checked": len(rows), "exact": exact, "unread": blank,
                "wrong": len(rows) - exact - blank}

    def prune(self, days: int) -> Dict[str, int]:
        cutoff = time.time() - days * 86400
        with self._lock:
            v = self._conn.execute("DELETE FROM vehicles WHERE ts < ?",
                                   (cutoff,)).rowcount
            r = self._conn.execute("DELETE FROM lpr_requests WHERE ts < ?",
                                   (cutoff,)).rowcount
            self._conn.commit()
            self._conn.execute("VACUUM")
        return {"vehicles_deleted": v, "requests_deleted": r, "days": days}
