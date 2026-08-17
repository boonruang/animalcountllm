"""FastAPI · เส้นเดียว: รับภาพ ตอบ JSON

🔴 บริการนี้ไม่ยิงใคร ไม่เตือนใคร ไม่ตัดสินใจแทนใคร
ห้ามเพิ่ม outbound webhook / alert threshold / คิวที่ยิงกลับไปหาปลายทาง
edge box ที่เรียกเราเป็นคนตัดสินใจเอง เราแค่ตอบให้ครบพอที่เขาจะตัดสินใจได้
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

from .cv import thermal  # noqa: E402
from .filters import temporal  # noqa: E402
from .llm.client import VisionLLM  # noqa: E402
from .llm import prompt_v2 as prompt  # noqa: E402
from .llm import prompt_v3  # noqa: E402
from .schemas import (Detection, FrameIn, FrameOut, FilteredResult, ModelInfo,  # noqa: E402
                      RawResult, SpeciesTally, TruthIn, WindowInfo)
from .store.base import DetectionRecord, FrameRecord, make_store  # noqa: E402

# 🔴 เลขนี้ต้องขยับทุกครั้งที่แก้พฤติกรรมของ API
# วันนี้ (16 ส.ค.) เดากันสามรอบว่า DO รันโค้ดคอมมิตไหนอยู่ เพราะไม่มีอะไรบอก
# /healthz จะรายงานค่านี้ ดูปุ๊บรู้เลยว่า deploy ทันหรือยัง
# 🔴 โหมดของท่อ · ตัดสินโดย Toy 2026-08-17
#
# "api เส้นของเรา วางตำแหน่งไว้ว่าคือ llm ครับ ไม่ต้องใช้ numpy"
#
# llm      = ถามโมเดลตรงๆ ทุกเฟรม ไม่มีชั้น CV · รับได้ทั้ง thermal และภาพสี
#            ซึ่งเป็นสิ่งที่ไซต์จะส่งมาจริง จากกล้องหลายตัว
# cv+llm   = ท่อเดิม · CV คัดก่อนแล้วค่อยถามโมเดล · รับเฉพาะ thermal
#
# ⚠️ สิ่งที่แลกไปกับโหมด llm ต้องรู้ ไม่ใช่ค้นพบตอนบิลมา:
#   1. **ทุกเฟรมยิง LLM** ด่าน CV ที่เคยคัดออก >90% ไม่มีแล้ว
#      8,640 เฟรม/วัน/กล้อง ที่ qwen3-vl-32b ตกราว $25/เดือน/กล้อง (เดิม ~$4.7)
#   2. **จำนวนมาจาก LLM ไม่ใช่ CV** ถามซ้ำอาจได้คนละเลข ต่างจาก connected component
#      ที่นับได้เท่าเดิมทุกครั้ง · response ติดป้าย count_source ไว้ให้ปลายทางรู้
#   3. **ไม่มี detections รายก้อน** bbox ของ LLM เชื่อไม่ได้ (วัดแล้ว คืน y=998
#      บนภาพสูง 628) จึงไม่ขอมาตั้งแต่แรก · raw.detections จะว่างในโหมดนี้
#
# 🔴 รูปของ JSON ที่ตอบกลับ **ห้ามเปลี่ยน** confirm กับทีมคุณสุชาติไปแล้ว
# ฟิลด์เท่าเดิมทุกตัว ชื่อเดิม ชนิดเดิม · โหมดนี้ทำให้บางฟิลด์เป็นค่าว่างได้
# (detections = [] · regions_detected = 0) แต่ฟิลด์ยังอยู่ครบและชนิดยังถูก
# ปลายทางแยกออกว่าจำนวนมาจากไหนด้วย model.prompt_version (v3 = โมเดลนับเอง)
PIPELINE = os.environ.get("PIPELINE", "llm").lower()

APP_VERSION = "0.5.0"
BUILD_NOTES = "llm-first pipeline (prompt v3, no CV gate), thermal + colour"

app = FastAPI(title="animalcountllm", version=APP_VERSION)

# 🔴 หน้าเว็บทดสอบที่ /verify · ไฟล์นิ่งล้วน ไม่ใช่ endpoint
#
# เคยแยกเป็น static site คนละคอมโพเนนต์ใน .do/app.yaml เพื่อไม่ให้ API รู้จัก UI เลย
# **ใช้ไม่ได้จริง** App Platform ไม่อ่าน .do/app.yaml ของแอปที่สร้างไว้แล้ว
# spec ตัวจริงอยู่ใน Console · push ขึ้น main แล้ว /verify ตอบ 404 จาก FastAPI เอง
# แปลว่าคอมโพเนนต์นั้นไม่เคยถูกสร้าง · 2026-08-17 เลยย้ายมา mount ตรงนี้
# ให้ deploy_on_push ส่งขึ้นได้โดยไม่ต้องกดอะไรใน Console
#
# บรรทัดพวกนี้ไม่แตะ endpoint ไม่แตะ response spec ไม่แตะ _auth ไม่แตะ logic
# หน้าเว็บยังเป็น client ธรรมดาที่ยิง POST /v1/frames และโดน 401 เหมือนใครก็ตาม
# **ห้ามให้มันข้าม auth และห้ามให้ app/ ไปอ่านอะไรจาก ui/ นอกจากเสิร์ฟไฟล์**
_UI_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui")
if os.path.isdir(_UI_DIR):
    app.mount("/verify", StaticFiles(directory=_UI_DIR, html=True), name="verify")
else:
    print(f"[startup] ไม่พบ {_UI_DIR} · /verify จะตอบ 404", flush=True)


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


def image_size(raw: bytes) -> tuple[int, int]:
    """ขนาดภาพจากหัวไฟล์ · ไม่ decode พิกเซล ไม่แตะ numpy

    โหมด llm ต้องการขนาดไปใส่ prompt กับเก็บลง DB เท่านั้น ไม่ได้เอาไปคำนวณอะไร
    Pillow อ่านหัวไฟล์อย่างเดียวถ้าไม่เรียก .load() เลยถูกทั้งเวลาและหน่วยความจำ
    """
    from PIL import Image
    with Image.open(io.BytesIO(raw)) as im:
        return im.size


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
            "version": APP_VERSION, "build": BUILD_NOTES,
            "provider": llm.provider, "model": llm.model,
            "pipeline": PIPELINE,
            "prompt_version": (prompt_v3.PROMPT_VERSION if PIPELINE == "llm"
                               else prompt.PROMPT_VERSION),
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

    # [2] อ่านขนาดภาพ · โหมด llm ไม่แตะ numpy เลย ใช้หัวไฟล์อย่างเดียว
    try:
        cv = thermal.analyze(raw_bytes) if PIPELINE != "llm" else None
        frame_w, frame_h = (cv.frame_w, cv.frame_h) if cv else image_size(raw_bytes)
    except Exception as e:
        rec.status = "error"
        rec.filter_reason = f"cannot read image: {type(e).__name__}"
        store.update_frame(rec)
        raise HTTPException(status_code=400, detail=f"cannot read image: {e}")

    rec.frame_w, rec.frame_h = frame_w, frame_h
    if cv:
        rec.bit_depth = cv.bit_depth
        rec.cv_result = cv.model_dump_json()

    # [3] LLM — ยิงเฉพาะตอนมีก้อนขนาดสัตว์ นี่คือที่มาของการประหยัด >90%
    detections: List[Detection] = []
    animals: List[SpeciesTally] = []
    status = "ok"
    model_info = ModelInfo(provider=llm.provider, name=llm.model,
                           prompt_version=(prompt_v3.PROMPT_VERSION if PIPELINE == "llm"
                                           else prompt.PROMPT_VERSION))
    vlm_ms = 0.0

    if PIPELINE == "llm":
        # ---- เส้น llm · ถามโมเดลทุกเฟรม ไม่มีอะไรมาคัดก่อน
        # ไม่มี "ไม่ยิงเพราะไม่เจอก้อน" อีกแล้ว เพราะการไม่เจอก้อนคือสิ่งที่ทำให้
        # ช้างเต็มเฟรมกลายเป็น {"animals":[]} มาทั้งวัน (ดู prompt_v3.py)
        res = llm.look(body.image_base64, frame_w, frame_h, body.camera_id, image_hash)
        vlm_ms = res.latency_ms
        rec.llm_raw_response = res.raw or (res.error or "")
        rec.prompt_version = prompt_v3.PROMPT_VERSION
        rec.model_name, rec.provider = llm.model, llm.provider
        rec.finish_reason, rec.completion_tokens = res.finish_reason, res.completion_tokens
        model_info.finish_reason = res.finish_reason
        model_info.completion_tokens = res.completion_tokens

        if res.usable and res.verdict:
            animals = list(res.verdict.animals)
        else:
            # โมเดลล่ม/ตอบไม่จบ · โหมดนี้ไม่มีผล CV สำรอง ตอบว่าไม่รู้ ไม่ใช่ตอบว่าไม่มี
            status = "degraded"
            animals = []

    # 🔴 ไม่เจอก้อน + ภาพหน้าตาไม่เหมือน thermal = **บอกออกไป ห้ามเงียบ**
    # เพิ่ม 2026-08-16 หลังยิงภาพถ่ายช้าง RGB เข้าไปแล้วได้ 200 OK counts ว่าง
    # หน้าตาเหมือนเฟรมที่ไม่มีอะไรจริงๆ ทุกประการ ปลายทางแยกไม่ออกเลย
    # ยังไม่ยิง LLM เหมือนเดิม (ไม่มีก้อน = ไม่มีอะไรให้ถาม และ prompt เขียนไว้ว่า
    # "Bright regions are warm" ซึ่งไม่จริงกับภาพแบบนี้ ถามไปก็ได้คำตอบที่ตั้งบนคำโกหก)
    # ต่างกันแค่ **ตอบตรงๆ ว่าอ่านไม่เป็น** ไม่ใช่ตอบว่าไม่มีสัตว์
    if cv and not cv.looks_thermal:
        status = "degraded"

    if cv and cv.has_candidates:
        res = llm.classify(body.image_base64, cv.blobs, cv.frame_w, cv.frame_h, image_hash)
        vlm_ms = res.latency_ms
        rec.llm_raw_response = res.raw or (res.error or "")
        rec.prompt_version = prompt.PROMPT_VERSION
        rec.model_name, rec.provider = llm.model, llm.provider
        rec.finish_reason, rec.completion_tokens = res.finish_reason, res.completion_tokens
        model_info.finish_reason = res.finish_reason
        model_info.completion_tokens = res.completion_tokens

        if res.usable and res.verdict:
            animals = list(res.verdict.animals)
        else:
            # LLM ล่ม/ตอบไม่จบ → ยังบอกได้ว่ามีของร้อนกี่ก้อน แต่ระบุชนิดไม่ได้
            status = "degraded"
            animals = [SpeciesTally(species="unknown", count=len(cv.blobs), confidence=0.0)]

        detections = [Detection(id=b.id, bbox=b.bbox, area_px=b.area_px,
                                aspect=b.aspect,
                                detection_confidence=b.detection_confidence)
                      for b in cv.blobs]

    # ---- สรุป: สัตว์อะไร กี่ตัว มั่นใจเท่าไร
    counts: Dict[str, int] = {}
    conf_by_species: Dict[str, float] = {}
    for a in animals:
        counts[a.species] = counts.get(a.species, 0) + a.count
        conf_by_species[a.species] = max(conf_by_species.get(a.species, 0.0), a.confidence)

    # ความมั่นใจรวม = ถ่วงน้ำหนักด้วยจำนวนตัว · คูณกับความมั่นใจของการตรวจจับฝั่ง CV
    total = sum(counts.values())
    # โหมด llm ไม่มีค่าฝั่ง CV มาคูณ · ความมั่นใจคือของโมเดลล้วน ซึ่งต้องบอกตรงๆ
    # ไม่ใช่คูณ 1.0 เงียบๆ แล้วให้ปลายทางเข้าใจว่ามันผ่านการตรวจสองชั้นเหมือนเดิม
    det_mean = 1.0 if cv is None else (
        sum(b.detection_confidence for b in cv.blobs) / len(cv.blobs) if cv.blobs else 0.0)
    overall = round(
        (sum(a.confidence * a.count for a in animals) / total * det_mean) if total else 0.0, 3)

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

    # เหตุผลที่สงสัยว่าไม่ใช่ภาพ thermal ต้องไปถึงปลายทางและลง log ไม่ใช่รู้อยู่คนเดียว
    # ต่อท้าย ไม่ทับ ของเดิมที่ filter บอกไว้ยังมีค่าอยู่
    if cv and cv.plausibility_reason:
        filtered.reason = f"{filtered.reason} | cv: {cv.plausibility_reason}".lstrip(" |")

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
    # ตาราง detections เก็บชนิดที่ระดับเฟรม ผูกกับก้อนที่ใหญ่ที่สุดพอเป็นตัวแทน
    top = max(animals, key=lambda a: a.count, default=None)
    store.insert_detections([
        DetectionRecord(request_id, d.id, top.species if top else "unknown",
                        top.confidence if top else 0.0,
                        d.detection_confidence, d.bbox) for d in detections])

    return FrameOut(
        request_id=request_id, camera_id=body.camera_id, received_at=_iso(now),
        status=status,
        raw=RawResult(animals=animals, counts=counts,
                      regions_detected=len(cv.blobs) if cv else 0,
                      detections=detections, overall_confidence=overall),
        filtered=filtered, window=window, model=model_info,
        timing_ms={"cv": cv.elapsed_ms if cv else 0.0, "vlm": vlm_ms,
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
