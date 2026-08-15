"""FastAPI · เส้นเดียว: รับภาพ ตอบ JSON

🔴 บริการนี้ไม่ยิงใคร ไม่เตือนใคร ไม่ตัดสินใจแทนใคร
ห้ามเพิ่ม outbound webhook / alert threshold / คิวที่ยิงกลับไปหาปลายทาง
edge box ที่เรียกเราเป็นคนตัดสินใจเอง เราแค่ตอบให้ครบพอที่เขาจะตัดสินใจได้
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

load_dotenv()

from .cv import thermal  # noqa: E402
from .filters import temporal  # noqa: E402
from .llm.client import VisionLLM  # noqa: E402
from .llm import prompt_v1  # noqa: E402
from .schemas import (Detection, FrameIn, FrameOut, FilteredResult, ModelInfo,  # noqa: E402
                      RawResult, TruthIn, WindowInfo)
from .store.base import DetectionRecord, FrameRecord, make_store  # noqa: E402

app = FastAPI(title="animalcountllm", version="0.1.0")


def _writable(preferred: str, fallback: str, is_dir: bool = False) -> str:
    """หา path ที่เขียนได้จริง ไม่ใช่เชื่อว่าเขียนได้

    🔴 บทเรียน 2026-08-16 · deploy แรกบน App Platform ตายด้วย exit code 190
    ก่อนจะ bind port ทัน แล้ว health check รายงานว่า connection refused
    ซึ่งชี้ไปผิดที่ทั้งหมด ของจริงคือ **โฟลเดอร์แอปบน buildpack เขียนไม่ได้**
    `./data/animals.db` เลยเปิดไม่ได้ตั้งแต่ตอน import
    /tmp เขียนได้เสมอ และหายทุก deploy อยู่แล้วตามทาง ค. เลยไม่ได้เสียอะไรเพิ่ม
    """
    target = preferred if is_dir else os.path.dirname(os.path.abspath(preferred))
    try:
        os.makedirs(target, exist_ok=True)
        probe = os.path.join(target, ".write_probe")
        with open(probe, "w") as f:
            f.write("1")
        os.remove(probe)
        return preferred
    except OSError as e:
        print(f"[startup] {preferred!r} เขียนไม่ได้ ({e.__class__.__name__}: {e}) "
              f"-> ใช้ {fallback!r} แทน", flush=True)
        return fallback


STORE_DSN = _writable(os.environ.get("STORE_DSN", "./data/animals.db"),
                      "/tmp/animalcountllm/animals.db")
IMAGE_DIR = _writable(os.environ.get("IMAGE_DIR", "./data/frames"),
                      "/tmp/animalcountllm/frames", is_dir=True)

STORE_ERROR: str | None = None
try:
    store = make_store(os.environ.get("STORE_BACKEND", "sqlite"), STORE_DSN)
    store.init_schema()
except Exception as e:  # noqa: BLE001
    # ห้ามตายตอน import · container ที่ตายเงียบบอกอะไรไม่ได้เลย
    # ขึ้นมาแล้วรายงานผ่าน /healthz ว่าพังเพราะอะไร ดีกว่า connection refused
    STORE_ERROR = f"{type(e).__name__}: {e}"
    print(f"[startup] 🔴 เปิดฐานข้อมูลไม่ได้: {STORE_ERROR}", flush=True)
    store = make_store("sqlite", "/tmp/animalcountllm-fallback.db")
    store.init_schema()

llm = VisionLLM()

SAVE_IMAGES = os.environ.get("SAVE_IMAGES", "interesting")
API_KEY = os.environ.get("API_KEY", "").strip()
print(f"[startup] store={STORE_DSN} images={IMAGE_DIR} provider={llm.provider} "
      f"model={llm.model} tracing={llm.tracing}", flush=True)


def _auth(key: str | None) -> None:
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=401, detail="bad or missing X-API-Key")


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


def _save_image(raw: bytes, request_id: str) -> str | None:
    """เก็บเฉพาะเฟรมที่น่าสนใจ · ทุกใบ = 3.5 GB/สัปดาห์/กล้อง"""
    try:
        os.makedirs(IMAGE_DIR, exist_ok=True)
        p = os.path.join(IMAGE_DIR, f"{request_id}.png")
        with open(p, "wb") as f:
            f.write(raw)
        return p
    except OSError:
        return None  # เก็บภาพไม่ได้ ไม่ควรทำให้ทั้ง request พัง


@app.get("/healthz")
def healthz():
    """ต้องตอบได้แม้ของบางอย่างพัง ไม่งั้น DO จะเห็นแค่ connection refused

    ซึ่งบอกไม่ได้ว่าพังเพราะอะไร เสียเวลาไล่ผิดที่
    """
    return {"status": "ok" if not STORE_ERROR else "degraded",
            "provider": llm.provider, "model": llm.model,
            "prompt_version": prompt_v1.PROMPT_VERSION,
            "store": os.environ.get("STORE_BACKEND", "sqlite"),
            "store_path": STORE_DSN, "store_error": STORE_ERROR,
            "image_dir": IMAGE_DIR, "tracing": llm.tracing}


@app.post("/v1/frames", response_model=FrameOut)
def post_frame(body: FrameIn, x_api_key: str | None = Header(default=None)):
    _auth(x_api_key)
    t_start = time.perf_counter()
    request_id = str(uuid.uuid4())
    now = time.time()
    if body.ts:
        try:
            now = datetime.fromisoformat(body.ts.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass  # ปลายทางส่งเวลาผิดรูป ใช้เวลาที่เรารับแทน ดีกว่าปฏิเสธทั้งเฟรม

    try:
        raw_bytes = base64.b64decode(body.image_base64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="image_base64 decode failed")

    image_hash = hashlib.sha256(raw_bytes).hexdigest()[:32]

    # [1] บันทึกทันที ก่อนทำอะไรทั้งสิ้น · LLM ล่มหรือช้า ยังต้องมี log ว่าเฟรมนี้เคยมาถึง
    rec = FrameRecord(request_id=request_id, camera_id=body.camera_id, ts=now,
                      image_hash=image_hash, status="received")
    store.insert_frame(rec)

    # [2] CV — deterministic ไม่ต้องมีเน็ต
    try:
        cv = thermal.analyze(raw_bytes)
    except Exception as e:
        rec.status = "error"
        rec.filter_reason = f"cv failed: {type(e).__name__}"
        store.update_frame(rec)
        raise HTTPException(status_code=400, detail=f"cannot read image: {e}")

    rec.frame_w, rec.frame_h, rec.bit_depth = cv.frame_w, cv.frame_h, cv.bit_depth
    rec.cv_result = cv.model_dump_json()

    # [3] LLM — ยิงเฉพาะตอนมีก้อนขนาดสัตว์ นี่คือที่มาของการประหยัด >90%
    detections: List[Detection] = []
    status = "ok"
    model_info = ModelInfo(provider=llm.provider, name=llm.model,
                           prompt_version=prompt_v1.PROMPT_VERSION)
    vlm_ms = 0.0

    if cv.has_candidates:
        res = llm.classify(body.image_base64, cv.blobs, cv.frame_w, cv.frame_h, image_hash)
        vlm_ms = res.latency_ms
        rec.llm_raw_response = res.raw or (res.error or "")
        rec.prompt_version = prompt_v1.PROMPT_VERSION
        rec.model_name, rec.provider = llm.model, llm.provider
        rec.finish_reason, rec.completion_tokens = res.finish_reason, res.completion_tokens
        model_info.finish_reason = res.finish_reason
        model_info.completion_tokens = res.completion_tokens

        by_id = {c.blob_id: c for c in res.verdict.calls} if res.verdict else {}
        if not res.usable:
            # LLM ล่ม/ตอบไม่จบ → ยังตอบด้วยผล CV ได้ แต่ทุกก้อนเป็น unknown
            status = "degraded"

        for b in cv.blobs:
            call = by_id.get(b.id)
            sp = call.species if (call and res.usable) else "unknown"
            sc = call.confidence if (call and res.usable) else 0.0
            detections.append(Detection(
                id=b.id, bbox=b.bbox, area_px=b.area_px, aspect=b.aspect,
                species=sp, species_confidence=sc,
                detection_confidence=b.detection_confidence,
                overall_confidence=round(sc * b.detection_confidence, 3)))

    # ---- นับ · จำนวนมาจาก CV ไม่ใช่จาก LLM
    counts: Dict[str, int] = {}
    conf_by_species: Dict[str, float] = {}
    for d in detections:
        counts[d.species] = counts.get(d.species, 0) + 1
        conf_by_species[d.species] = max(conf_by_species.get(d.species, 0.0),
                                         d.species_confidence)

    area_total = sum(d.area_px for d in detections) or 1
    overall = round(sum(d.overall_confidence * d.area_px for d in detections) / area_total, 3)

    # [4][5] corroboration + filter
    hist = store.window(body.camera_id, now, temporal.WINDOW_SECONDS)
    streak = store.provisional_streak(body.camera_id)
    filtered, window = temporal.decide(counts, conf_by_species, hist, now, streak)

    # 🔴 degraded ต้องชนะ provisional · เจอบั๊กนี้ตอนทดสอบจริง 2026-08-16
    # LLM ตอบไม่จบ (finish_reason=length) → ทุกก้อนเป็น unknown conf 0.0
    # แล้ว filter เห็น conf 0.0 < 0.60 ก็ตี provisional ทับ degraded
    # ปลายทางจะอ่านว่า "รออีกเฟรมเดี๋ยวก็รู้" ทั้งที่ความจริงคือ **โมเดลใช้ไม่ได้อยู่ตอนนี้**
    # สองอย่างนี้ต้องแก้คนละวิธี ห้ามกลบกัน
    if status == "ok" and not filtered.accepted and filtered.state == "unconfirmed":
        status = "provisional"

    # [6] บันทึกผล
    rec.status = status
    rec.raw_counts = counts
    rec.filtered_counts = filtered.counts
    rec.filter_reason = filtered.reason
    rec.state = filtered.state
    rec.latency_ms = round((time.perf_counter() - t_start) * 1000, 1)

    interesting = bool(counts) or not filtered.accepted or status != "ok"
    if SAVE_IMAGES == "all" or (SAVE_IMAGES == "interesting" and interesting):
        rec.image_path = _save_image(raw_bytes, request_id)

    store.update_frame(rec)
    store.insert_detections([
        DetectionRecord(request_id, d.id, d.species, d.species_confidence,
                        d.detection_confidence, d.bbox) for d in detections])

    return FrameOut(
        request_id=request_id, camera_id=body.camera_id, received_at=_iso(now),
        status=status,
        raw=RawResult(detections=detections, counts=counts, overall_confidence=overall),
        filtered=filtered, window=window, model=model_info,
        timing_ms={"cv": cv.elapsed_ms, "vlm": vlm_ms,
                   "total": round((time.perf_counter() - t_start) * 1000, 1)},
    )


@app.get("/v1/frames/{request_id}")
def get_frame(request_id: str, x_api_key: str | None = Header(default=None)):
    _auth(x_api_key)
    row = store.get_frame(request_id)
    if not row:
        raise HTTPException(status_code=404, detail="unknown request_id")
    for k in ("raw_counts", "filtered_counts", "cv_result"):
        if row.get(k):
            try:
                row[k] = json.loads(row[k])
            except json.JSONDecodeError:
                pass
    return JSONResponse(row)


@app.get("/v1/cameras/{camera_id}/window")
def get_window(camera_id: str, x_api_key: str | None = Header(default=None)):
    """ดูหน้าต่าง 100 วิ ปัจจุบัน · มีไว้ debug ว่า filter ตัดสินจากอะไร"""
    _auth(x_api_key)
    now = time.time()
    items = store.window(camera_id, now, temporal.WINDOW_SECONDS)
    return {"camera_id": camera_id, "span_seconds": temporal.WINDOW_SECONDS,
            "frames_used": len(items), "frames_expected": temporal.EXPECTED_FRAMES,
            "complete": len(items) >= temporal.EXPECTED_FRAMES,
            "provisional_streak": store.provisional_streak(camera_id),
            "items": [{"age_s": round(now - i.ts, 1), "counts": i.counts,
                       "confidence": i.confidence} for i in items]}


@app.post("/v1/frames/{request_id}/truth")
def post_truth(request_id: str, body: TruthIn,
               x_api_key: str | None = Header(default=None)):
    """ground truth ที่คนตรวจแล้ว · ไม่มีข้อมูลนี้ = calibrate confidence ไม่ได้ตลอดกาล"""
    _auth(x_api_key)
    if not store.save_truth(request_id, body.counts, body.reviewer, body.comment):
        raise HTTPException(status_code=404, detail="unknown request_id")
    return {"ok": True, "request_id": request_id}


@app.post("/v1/maintenance/rollup")
def rollup(x_api_key: str | None = Header(default=None)):
    """สรุปวันละแถวแล้วลบของดิบที่เกิน RETENTION_DAYS

    เรียกจาก cron ข้างนอก ไม่ใช่ background task ในนี้ เพราะ container ตายเมื่อไหร่ก็ได้
    """
    _auth(x_api_key)
    return store.rollup_and_prune(int(os.environ.get("RETENTION_DAYS", "7")))
