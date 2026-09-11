"""FastAPI ของงาน smart people counting · รับ id + ภาพ ตอบ เพศ/ช่วงอายุ/รูปพรรณ

แอปตัวนี้ถูก mount ไว้ใต้ `/people` ของแอปเดิม (ดูท้าย `app/main.py`)
endpoint จริงจึงเป็น:

    POST /people/v1/persons          ปลายทางชี้คนมาเอง (id + crop หรือ id + bbox)
    POST /people/v1/frames           เฟรมเดียว ไม่มีใครถูกชี้ โมเดลหาคนเอง
    GET  /people/v1/persons/{request_id}
    GET  /people/v1/persons/by-ref/{client_request_id}   ตามด้วยเลขอ้างอิงของปลายทางเอง
    POST /people/v1/persons/{request_id}/{object_id}/truth
    POST /people/v1/maintenance/prune
    GET  /people/healthz
    GET  /people/verify/            หน้าเว็บทดสอบด้วยตา

🔴 ทำไมถึง mount ใต้แอปเดิม ไม่แยกเป็นคอมโพเนนต์ใหม่บน App Platform
เพราะ **App Platform ไม่อ่าน `.do/app.yaml` ของแอปที่สร้างไว้แล้ว** เจอมาแล้ว
ตอนทำหน้า `/verify` ของฝั่งช้าง 2026-08-17: เขียน static_sites ลงไฟล์ push ขึ้นไป
คอมโพเนนต์ไม่เคยถูกสร้าง แล้ว path ตอบ 404 จาก FastAPI เอง
ทางที่ deploy ได้เองจริงโดยไม่ต้องกดอะไรใน Console คือ mount เข้าแอปที่มีอยู่
Procfile เดิม · run_command เดิม · env เดิม

🔴 บริการนี้ไม่ยิงใคร ไม่เตือนใคร ไม่ตัดสินใจแทนใคร เหมือนฝั่งช้างทุกประการ
ไม่จับคู่ใบหน้า ไม่ระบุตัวบุคคล ไม่เก็บทะเบียนคน · ตอบคำถามที่ถามมาต่อหนึ่งภาพ
แล้วจบ · ห้ามเพิ่ม face matching / watchlist / outbound webhook ลงในไฟล์นี้
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import io
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .llm import prompt_f1, prompt_p1
from .llm.client import PersonLLM, normalize, people_of
from .schemas import (ImageInfo, ModelInfo, ObjectIn, PeopleFrameIn, PersonOut,
                      PersonTruthIn, PersonsIn, PersonsOut)
from .store import PersonStore

# 🔴 เลขนี้ต้องขยับทุกครั้งที่ **พฤติกรรมของ endpoint ฝั่งคน** เปลี่ยน
# แยกจาก APP_VERSION ของฝั่งช้างโดยตั้งใจ สองงานนี้จะ deploy ไปด้วยกันก็จริง
# แต่ปลายทางคนละทีม ต้องตอบได้ว่า "ของที่คุณเรียกอยู่เวอร์ชันอะไร" แยกกัน
PEOPLE_VERSION = "0.4.0"
BUILD_NOTES = ("person attributes (gender, age band, appearance) via VLM"
               " · /v1/persons prompt p1 · /v1/frames prompt f1")

app = FastAPI(title="smart-people-counting", version=PEOPLE_VERSION)


def _writable(preferred: str, fallback: str, is_dir: bool = False) -> str:
    """หา path ที่เขียนได้จริง ไม่ใช่เชื่อว่าเขียนได้

    บทเรียนเดียวกับฝั่งช้าง 2026-08-16: โฟลเดอร์แอปบน buildpack เขียนไม่ได้
    แล้วอาการที่เห็นคือ container ตายตอน import แล้ว health check บอกว่า
    connection refused ซึ่งชี้ไปผิดที่ทั้งหมด
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
        print(f"[people] {preferred!r} เขียนไม่ได้ ({type(e).__name__}: {e}) "
              f"-> ใช้ {fallback!r} แทน", flush=True)
        return fallback


STORE_DSN = _writable(os.environ.get("PEOPLE_STORE_DSN", "./data/people.db"),
                      "/tmp/animalcountllm/people.db")

STORE_ERROR: Optional[str] = None
try:
    store = PersonStore(STORE_DSN)
    store.init_schema()
except Exception as e:  # noqa: BLE001
    STORE_ERROR = f"{type(e).__name__}: {e}"
    print(f"[people] 🔴 เปิดฐานข้อมูลไม่ได้: {STORE_ERROR}", flush=True)
    store = PersonStore("/tmp/people-fallback.db")
    store.init_schema()

llm = PersonLLM()

# คีย์: ใช้ตัวเดียวกับฝั่งช้างเป็นค่าตั้งต้น แต่แยก env ไว้แล้ว
# วันที่ทีมคุณสุชาติได้คีย์ของงานคนไปแล้วอยากถอนสิทธิ์ฝั่งช้าง จะได้ไม่ต้องแก้โค้ด
API_KEY = (os.environ.get("PEOPLE_API_KEY") or os.environ.get("API_KEY", "")).strip()

# ไม่เก็บภาพเป็นค่าเริ่มต้น · ดูเหตุผลใน people/store.py
SAVE_IMAGES = os.environ.get("PEOPLE_SAVE_IMAGES", "none").lower()
IMAGE_DIR = _writable(os.environ.get("PEOPLE_IMAGE_DIR", "./data/people-crops"),
                      "/tmp/animalcountllm/people-crops", is_dir=True)

# 🔴 กล่องจาก tracker มักรัดพอดีตัว แล้วตัดรองเท้ากับผมทิ้ง
# ซึ่งเป็นสองอย่างที่เราถูกสั่งให้รายงาน · ขยายกรอบออก 8% ก่อนครอป
# ตั้ง 0 ได้ถ้าปลายทางส่งกรอบที่เผื่อมาแล้ว
BBOX_MARGIN = float(os.environ.get("PEOPLE_BBOX_MARGIN", "0.08"))

# ⚠️ ค่านี้ยัง **ไม่ได้วัด** ว่าช่วยเรื่องความแม่นไหม
# มีไว้กันอาการ provider ปฏิเสธภาพที่เล็กเกินไป (crop คนไกลๆ เหลือ 40x90 ได้)
# ที่วัดไปแล้วฝั่งช้าง: prompt token เท่ากันหมดที่ 640x512 แปลว่าการขยายภาพ
# ไม่ได้แปลว่าโมเดลได้ image token เพิ่มเสมอไป · อย่าเชื่อว่ายิ่งใหญ่ยิ่งแม่น
# จนกว่าจะวัดกับ crop จริงจากไซต์
MIN_CROP_PX = int(os.environ.get("PEOPLE_MIN_CROP_PX", "224"))

# หนึ่งคน = หนึ่งครั้งที่ยิงโมเดล · ยิงขนานกันเพื่อไม่ให้ 5 คนใช้เวลา 5 เท่า
# 4 เส้นพอ · มากกว่านี้ไปชนกับ rate limit ของ OpenRouter แทน
WORKERS = int(os.environ.get("PEOPLE_WORKERS", "4"))

# เพดานจำนวนคนที่เส้น /v1/frames จะรายงานต่อเฟรม
# ล็อบบี้ตอนพักเที่ยงมีคนสามสิบคนได้ ตอบครบสามสิบ = คำตอบยาวจนโดนตัดกลาง
# แล้วเสียทั้งเฟรม · สิบสองคนแรกที่ใกล้กล้องที่สุด มีค่ากว่าสามสิบคนที่ตอบไม่จบ
FRAME_MAX_PEOPLE = int(os.environ.get("PEOPLE_FRAME_MAX_PEOPLE", "12"))

# 🔴 ปลายทางไม่ต้องรู้ว่าเราใช้โมเดลอะไร ของเจ้าไหน prompt เวอร์ชันไหน
# (Toy สั่ง 2026-09-11) · คีย์ model ยังอยู่ในคำตอบแต่เป็น null
# **ไม่ถอดคีย์ทิ้ง** รูป response ที่เปลี่ยนตามค่า env คือของที่ทำให้ปลายทาง
# parse พังแบบหาสาเหตุไม่เจอ · ของจริงยังเก็บครบในฐานข้อมูล ไล่ย้อนหลังได้เหมือนเดิม
# เปิดดูตอน dev ด้วย PEOPLE_EXPOSE_MODEL=true
EXPOSE_MODEL = os.environ.get("PEOPLE_EXPOSE_MODEL", "false").lower() == "true"

print(f"[people] v{PEOPLE_VERSION} store={STORE_DSN} provider={llm.provider} "
      f"model={llm.model} save_images={SAVE_IMAGES}", flush=True)


@app.exception_handler(RequestValidationError)
def explain_validation_error(request: Request, exc: RequestValidationError):
    """422 ของ FastAPI บอกว่าฟิลด์ไหนหาย แต่ไม่บอกว่า "คุณมาผิดเส้น"

    เจอจริง 2026-09-11: Toy ส่ง body แบบเฟรมเดียว (camera_id + image_base64)
    มาที่ /v1/persons แล้วได้ `objects: Field required` ซึ่งถูกต้องแต่ช่วยอะไรไม่ได้
    ทีมปลายทางจะเจอเรื่องเดียวกันแน่นอน เพราะสองเส้นนี้รับคนละรูป

    🔴 ไม่แก้ด้วยการให้ /v1/persons รับ body แบบเฟรมเดียวแล้วเดาว่าเขาหมายถึงอะไร
    endpoint ที่แอบทำงานคนละอย่างกับชื่อตัวเอง คือบั๊กที่หาไม่เจอทีหลัง
    ตอบ 422 เหมือนเดิม ชนิดเดิม เพิ่มแค่ `hint` ที่บอกทางให้คน

    `detail` ยังเป็นรูปเดิมเป๊ะเพื่อให้ปลายทางที่ parse อยู่แล้วไม่พัง
    """
    errors = exc.errors()
    body = exc.body if isinstance(exc.body, dict) else {}
    hint = ""
    missing = {tuple(e.get("loc", ()))[-1] for e in errors if e.get("type") == "missing"}

    if "objects" in missing and body.get("image_base64"):
        hint = ("body นี้เป็นรูปของ POST /people/v1/frames (เฟรมเดียว ไม่มีใครถูกชี้) "
                "แต่ยิงมาที่ /v1/persons ซึ่งต้องมี objects[] · "
                "ยิงไปที่ /people/v1/frames แทน หรือใส่ objects พร้อม bbox ของแต่ละคน")
    elif "image_base64" in missing and body.get("objects"):
        hint = ("body นี้เป็นรูปของ POST /people/v1/persons แต่ยิงมาที่ /v1/frames "
                "ซึ่งรับภาพเดียวชื่อ image_base64 · ยิงไปที่ /people/v1/persons แทน")
    elif any(tuple(e.get("loc", ()))[-1] == "objects"
             and e.get("type") == "too_long" for e in errors):
        hint = (f"เกินเพดาน 16 คนต่อ request · แบ่งเป็นหลาย request "
                f"(ส่งมา {len(body.get('objects') or [])} คน)")

    return JSONResponse(status_code=422,
                        content=jsonable_encoder({"detail": errors,
                                                  **({"hint": hint} if hint else {})}))


def _direction_counts(persons: List[PersonOut]) -> dict:
    """เข้ากี่คน ออกกี่คน ไม่รู้กี่คน · สามตัวรวมกันต้องเท่าจำนวนคนเสมอ

    🔴 `direction_unknown` ต้องอยู่ในคำตอบเสมอ ห้ามซ่อน
    ปลายทางที่เห็นแต่ in กับ out จะเอาไปบวกเป็นยอดรวมแล้วหายไปเงียบๆ ว่าอีกกี่คน
    ที่เราบอกไม่ได้ · ตัวเลขที่ไม่ครบแต่ดูครบ แย่กว่าตัวเลขที่บอกว่าตัวเองไม่ครบ
    """
    return {"direction_in": sum(1 for p in persons if p.direction == "in"),
            "direction_out": sum(1 for p in persons if p.direction == "out"),
            "direction_unknown": sum(1 for p in persons if p.direction == "unknown")}


def _auth(key: Optional[str]) -> None:
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=401, detail="bad or missing X-API-Key")


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


def _decode(b64: str, what: str) -> bytes:
    try:
        return base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail=f"{what} decode failed")


def _crop(frame_bytes: bytes, bbox: List[int]) -> Tuple[str, int, int, List[int]]:
    """ครอปคนหนึ่งคนออกจากเฟรมเต็ม แล้วคืนเป็น base64 JPEG

    รับ [x, y, w, h] หน่วยพิกเซล · กรอบที่ล้นขอบภาพถูกหนีบให้อยู่ในภาพ
    ไม่ใช่โยน error เพราะ tracker คืนกรอบล้นขอบเป็นเรื่องปกติเวลาคนเดินออกเฟรม
    """
    from PIL import Image

    with Image.open(io.BytesIO(frame_bytes)) as im:
        im = im.convert("RGB")
        W, H = im.size
        x, y, w, h = (int(v) for v in bbox)
        if w <= 0 or h <= 0:
            raise ValueError(f"bbox กว้างหรือสูงไม่เป็นบวก: {bbox}")
        mx, my = int(w * BBOX_MARGIN), int(h * BBOX_MARGIN)
        x1, y1 = max(0, x - mx), max(0, y - my)
        x2, y2 = min(W, x + w + mx), min(H, y + h + my)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"bbox อยู่นอกภาพ {W}x{H}: {bbox}")
        crop = im.crop((x1, y1, x2, y2))

        if MIN_CROP_PX and min(crop.size) < MIN_CROP_PX:
            scale = MIN_CROP_PX / min(crop.size)
            crop = crop.resize((max(1, int(crop.width * scale)),
                                max(1, int(crop.height * scale))),
                               Image.LANCZOS)
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=90)
        return (base64.b64encode(buf.getvalue()).decode(), crop.width, crop.height,
                [x1, y1, x2, y2])


def _size_of(raw: bytes) -> Tuple[int, int]:
    """ขนาดภาพจากหัวไฟล์ ไม่ decode พิกเซล"""
    from PIL import Image
    with Image.open(io.BytesIO(raw)) as im:
        return im.size


def _save_crop(raw_b64: str, request_id: str, object_id: str,
               status: str) -> Optional[str]:
    """เก็บ crop ลงดิสก์เฉพาะตอนที่ตั้งใจให้เก็บ

    none     ไม่เก็บเลย (ค่าเริ่มต้น)
    degraded เก็บเฉพาะคนที่อ่านไม่ออก ซึ่งคือชุดที่มีค่าตอนดีบักจริงๆ
    all      เก็บทุกคน · ใช้ตอนไล่บั๊กเท่านั้น อย่าเปิดค้างไว้บน prod
    """
    if not raw_b64:
        return None
    if SAVE_IMAGES == "none":
        return None
    if SAVE_IMAGES == "degraded" and status == "ok":
        return None
    if SAVE_IMAGES not in {"all", "degraded"}:
        return None
    try:
        os.makedirs(IMAGE_DIR, exist_ok=True)
        safe = "".join(c for c in object_id if c.isalnum() or c in "-_")[:64] or "obj"
        p = os.path.join(IMAGE_DIR, f"{request_id}-{safe}.jpg")
        with open(p, "wb") as f:
            f.write(base64.b64decode(raw_b64))
        return p
    except (OSError, binascii.Error):
        return None  # เก็บภาพไม่ได้ ไม่ควรทำให้ทั้ง request พัง


def _model_info(res=None) -> Optional[ModelInfo]:
    if not EXPOSE_MODEL:
        return None
    return ModelInfo(
        provider=llm.provider, name=llm.model,
        prompt_version=prompt_p1.PROMPT_VERSION,
        finish_reason=getattr(res, "finish_reason", None),
        completion_tokens=getattr(res, "completion_tokens", None))


# ---------------------------------------------------------------- endpoints
@app.get("/healthz")
def healthz():
    """ต้องตอบได้แม้ของบางอย่างพัง · ไม่งั้นเห็นแค่ connection refused"""
    return {"status": "ok" if not STORE_ERROR else "degraded",
            "service": "smart-people-counting",
            "version": PEOPLE_VERSION, "build": BUILD_NOTES,
            # 🔴 ซ่อนที่ response แล้วแต่ยังโชว์ที่นี่ = ซ่อนไม่สำเร็จ
            # หน้า /people/healthz เปิดได้โดยไม่ต้องมีคีย์ด้วยซ้ำ
            "provider": llm.provider if EXPOSE_MODEL else None,
            "model": llm.model if EXPOSE_MODEL else None,
            "prompt_version": prompt_p1.PROMPT_VERSION if EXPOSE_MODEL else None,
            "frame_prompt_version": prompt_f1.PROMPT_VERSION if EXPOSE_MODEL else None,
            "max_objects": 16, "workers": WORKERS,
            "frame_max_people": FRAME_MAX_PEOPLE,
            "bbox_margin": BBOX_MARGIN, "min_crop_px": MIN_CROP_PX,
            "save_images": SAVE_IMAGES,
            "store_path": STORE_DSN, "store_error": STORE_ERROR,
            "auth_required": bool(API_KEY), "tracing": llm.tracing}


@app.post("/v1/persons", response_model=PersonsOut)
def post_persons(body: PersonsIn, x_api_key: Optional[str] = Header(default=None)):
    _auth(x_api_key)
    t_start = time.perf_counter()
    request_id = str(uuid.uuid4())
    now = time.time()
    if body.ts:
        try:
            now = datetime.fromisoformat(body.ts.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass  # เวลาผิดรูปจากปลายทาง ใช้เวลาที่เรารับแทน ดีกว่าปฏิเสธทั้งก้อน

    ids = [o.id for o in body.objects]
    if len(set(ids)) != len(ids):
        # id ซ้ำ = ปลายทางจับคู่คำตอบกลับไม่ได้ ตอบไปก็ไม่มีประโยชน์
        raise HTTPException(status_code=400,
                            detail=f"object id ซ้ำกันใน request เดียว: {ids}")

    store.insert_request(request_id, body.camera_id, now, len(body.objects),
                         body.note, body.client_request_id)

    # ---- [1] เตรียมภาพของแต่ละคน ทำก่อนยิงโมเดลทั้งหมด
    # decode เฟรมเต็มครั้งเดียว ไม่ใช่ครั้งละคน · เฟรม 1920x1080 สามคน
    # = decode สามรอบโดยไม่จำเป็น
    t_decode = time.perf_counter()
    frame_bytes = _decode(body.frame_base64, "frame_base64") if body.frame_base64 else None

    prepared: List[Tuple[ObjectIn, Optional[str], Optional[ImageInfo], str]] = []
    for o in body.objects:
        if o.image_base64:
            raw = _decode(o.image_base64, f"objects[{o.id}].image_base64")
            try:
                w, h = _size_of(raw)
            except Exception as e:  # noqa: BLE001
                prepared.append((o, None, None, f"อ่านภาพไม่ได้: {type(e).__name__}"))
                continue
            prepared.append((o, o.image_base64, ImageInfo(w=w, h=h,
                                                          source="object_image"), ""))
        elif o.bbox is not None:
            if frame_bytes is None:
                prepared.append((o, None, None,
                                 "ส่ง bbox มาแต่ไม่มี frame_base64 ใน request"))
                continue
            try:
                b64, w, h, used = _crop(frame_bytes, o.bbox)
            except Exception as e:  # noqa: BLE001
                prepared.append((o, None, None, f"ครอปไม่ได้: {e}"))
                continue
            prepared.append((o, b64, ImageInfo(w=w, h=h, source="frame_bbox",
                                               bbox_used=used), ""))
        else:
            prepared.append((o, None, None,
                             "object นี้ไม่มีทั้ง image_base64 และ bbox"))
    decode_ms = round((time.perf_counter() - t_decode) * 1000, 1)

    # ทุก object พังตั้งแต่ชั้นภาพ = ปลายทางส่งของผิดรูปมาทั้งก้อน บอกไปตรงๆ 400
    # ดีกว่าตอบ 200 ที่มีแต่ error ข้างใน ซึ่งปลายทางที่เช็คแค่ status code จะไม่เห็น
    if all(p[1] is None for p in prepared):
        raise HTTPException(status_code=400, detail="; ".join(
            f"{p[0].id}: {p[3]}" for p in prepared))

    # ---- [2] ยิงโมเดล หนึ่งคนหนึ่งครั้ง ขนานกัน
    t_vlm = time.perf_counter()

    def work(item):
        o, b64, info, err = item
        if b64 is None:
            return o, None, info, err
        h = hashlib.sha256(b64.encode()).hexdigest()[:32]
        res = llm.describe(b64, o.id, info.w, info.h, body.camera_id, h)
        return o, res, info, ""

    if len(prepared) == 1:
        results = [work(prepared[0])]
    else:
        with ThreadPoolExecutor(max_workers=min(WORKERS, len(prepared))) as pool:
            results = list(pool.map(work, prepared))
    vlm_wall_ms = round((time.perf_counter() - t_vlm) * 1000, 1)

    # ---- [3] ประกอบคำตอบ + บันทึก
    persons: List[PersonOut] = []
    rows = []
    vlm_sum = 0.0
    # ภาพที่ยิงไปจริงของแต่ละคน · ใช้ตอนเก็บ crop ลงดิสก์ (ปกติปิดอยู่)
    b64_by_id = {p[0].id: p[1] for p in prepared}
    for o, res, info, prep_err in results:
        if res is None:
            # ภาพใช้ไม่ได้ตั้งแต่ต้น ยังไม่เคยถึงโมเดล
            persons.append(PersonOut(id=o.id, status="error", reason=prep_err,
                                     image=info, model=_model_info(), timing_ms=0.0))
            rows.append({"request_id": request_id, "object_id": o.id,
                         "camera_id": body.camera_id, "ts": now, "status": "error",
                         "reason": prep_err, "latency_ms": 0.0,
                         "prompt_version": prompt_p1.PROMPT_VERSION,
                         "model_name": llm.model, "provider": llm.provider})
            continue

        vlm_sum += res.latency_ms
        if res.usable:
            n = normalize(res.data)
            conf = round((n["gender_confidence"] + n["age_range_confidence"]
                          + n["appearance_confidence"]) / 3, 3)
            p = PersonOut(id=o.id, status="ok", overall_confidence=conf,
                          image=info, model=_model_info(res),
                          timing_ms=res.latency_ms, **n)
        else:
            # 🔴 โมเดลล่ม/ตอบไม่จบ = degraded ห้ามตอบเป็น unknown เฉยๆ
            # unknown แปลว่า "ดูแล้วแต่ดูไม่ออก" ซึ่งเป็นคำตอบที่จบแล้ว
            # degraded แปลว่า "ยังไม่ได้ดู ลองใหม่ได้" · สองอย่างนี้แก้คนละวิธี
            # (บทเรียนตรงๆ จากฝั่งช้าง ที่ degraded เคยโดน provisional ทับ)
            why = res.error or (f"โมเดลตอบไม่จบ (finish={res.finish_reason})"
                                if res.finish_reason == "length" else "โมเดลตอบไม่เป็น JSON")
            p = PersonOut(id=o.id, status="degraded", reason=why, image=info,
                          model=_model_info(res), timing_ms=res.latency_ms)
        persons.append(p)
        rows.append({
            "request_id": request_id, "object_id": o.id, "camera_id": body.camera_id,
            "ts": now, "status": p.status,
            "direction": p.direction, "direction_confidence": p.direction_confidence,
            "gender": p.gender, "gender_confidence": p.gender_confidence,
            "age_range": p.age_range, "age_range_confidence": p.age_range_confidence,
            "appearance": p.appearance.model_dump_json(),
            "appearance_confidence": p.appearance_confidence,
            "description": p.description, "overall_confidence": p.overall_confidence,
            "reason": p.reason,
            "image_w": info.w if info else None, "image_h": info.h if info else None,
            "image_source": info.source if info else None,
            "image_path": _save_crop(b64_by_id.get(o.id) or "", request_id, o.id,
                                     p.status),
            "llm_raw_response": res.raw or (res.error or ""),
            "prompt_version": prompt_p1.PROMPT_VERSION,
            "model_name": llm.model, "provider": llm.provider,
            "finish_reason": res.finish_reason,
            "completion_tokens": res.completion_tokens,
            "latency_ms": res.latency_ms,
        })

    ok_n = sum(1 for p in persons if p.status == "ok")
    status = "ok" if ok_n == len(persons) else ("degraded" if ok_n == 0 else "partial")
    total_ms = round((time.perf_counter() - t_start) * 1000, 1)

    store.insert_persons(rows)
    store.finish_request(request_id, status, total_ms)

    return PersonsOut(
        request_id=request_id, client_request_id=body.client_request_id,
        note=body.note, camera_id=body.camera_id, received_at=_iso(now),
        status=status, persons=persons,
        summary={"objects_in": len(body.objects), "ok": ok_n,
                 "degraded": sum(1 for p in persons if p.status == "degraded"),
                 "error": sum(1 for p in persons if p.status == "error"),
                 # นับเฉพาะคนที่ตอบได้ · direction ของคน degraded เป็น unknown อยู่แล้ว
                 # และมันควรถูกนับเป็น "ไม่รู้" ไม่ใช่หายไปจากยอดรวมเฉยๆ
                 **_direction_counts(persons)},
        model=_model_info(),
        timing_ms={"decode": decode_ms, "vlm": vlm_wall_ms,
                   "vlm_sum": round(vlm_sum, 1), "total": total_ms},
    )


@app.post("/v1/frames", response_model=PersonsOut)
def post_frame(body: PeopleFrameIn, x_api_key: Optional[str] = Header(default=None)):
    """เฟรมเดียว ไม่มีใครถูกชี้ · โมเดลหาคนเอง ตอบทุกคนในครั้งเดียว

    🔴 body หน้าตาเหมือน `POST /v1/frames` ของฝั่งช้างเป๊ะ ตามที่ Toy สั่ง 2026-09-11
    ปลายทางที่ยิงฝั่งช้างอยู่แล้วเปลี่ยนแค่ path · หนึ่ง request หนึ่งภาพ ไม่มี array

    สิ่งที่แลกไปเทียบกับ `/v1/persons` **ต้องรู้ก่อนใช้ ไม่ใช่มาค้นพบทีหลัง**:
    ได้ `ref` (P1 P2 P3) ที่มีความหมายเฉพาะในคำตอบนี้ **ผูกกับ track id ของปลายทางไม่ได้**
    ตำแหน่งมาเป็นข้อความใน `where` ไม่ใช่ bbox เพราะ bbox ของ LLM เชื่อไม่ได้ (วัดแล้ว)
    แลกมากับ: เร็วกว่า ~3 เท่า ถูกกว่า ~1.4 เท่า และไม่ต้องมีใคร detect มาก่อน

    ใครที่มี bbox อยู่แล้วให้ใช้ `/v1/persons` ไม่มีเหตุผลที่จะทิ้ง id ทิ้งไป
    """
    _auth(x_api_key)
    t_start = time.perf_counter()
    request_id = str(uuid.uuid4())
    now = time.time()
    if body.ts:
        try:
            now = datetime.fromisoformat(body.ts.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass

    t_decode = time.perf_counter()
    raw_bytes = _decode(body.image_base64, "image_base64")
    try:
        frame_w, frame_h = _size_of(raw_bytes)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"อ่านภาพไม่ได้: {type(e).__name__}")
    decode_ms = round((time.perf_counter() - t_decode) * 1000, 1)

    store.insert_request(request_id, body.camera_id, now, 0, body.note,
                         body.client_request_id)
    image_hash = hashlib.sha256(body.image_base64.encode()).hexdigest()[:32]

    t_vlm = time.perf_counter()
    res = llm.describe_frame(body.image_base64, frame_w, frame_h, body.camera_id,
                             image_hash, FRAME_MAX_PEOPLE)
    vlm_ms = round((time.perf_counter() - t_vlm) * 1000, 1)

    persons: List[PersonOut] = []
    rows = []
    info = ImageInfo(w=frame_w, h=frame_h, source="full_frame")
    model_info = _model_info(res)

    if not res.usable:
        # 🔴 โมเดลล่ม/ตอบไม่จบ = degraded พร้อมรายชื่อว่าง
        # **ห้ามตอบ ok + คนศูนย์คน** เพราะนั่นแปลว่า "ล็อบบี้ว่าง" ซึ่งคนละเรื่องกันเลย
        # นี่คือบั๊กเดียวกับที่ฝั่งช้างเสียเวลาทั้งวัน ตอนภาพ RGB ได้ 200 OK counts ว่าง
        status = "degraded"
        why = res.error or (f"โมเดลตอบไม่จบ (finish={res.finish_reason})"
                            if res.finish_reason == "length" else "โมเดลตอบไม่เป็น JSON")
    else:
        found, err = people_of(res.data, FRAME_MAX_PEOPLE)
        if err:
            status, why = "degraded", err
        else:
            status, why = "ok", ""
            for ref, where, item in found:
                n = normalize(item)
                conf = round((n["gender_confidence"] + n["age_range_confidence"]
                              + n["appearance_confidence"]) / 3, 3)
                persons.append(PersonOut(id=ref, status="ok", where=where,
                                         overall_confidence=conf, image=info,
                                         model=model_info,
                                         timing_ms=res.latency_ms, **n))

    if status == "degraded":
        # ไม่มีรายคนให้รายงาน · เหตุผลอยู่ที่ระดับ request ไม่ใช่ซ่อนอยู่ในคนที่ไม่มี
        rows.append({"request_id": request_id, "object_id": "-",
                     "camera_id": body.camera_id, "ts": now, "status": "degraded",
                     "reason": why, "latency_ms": res.latency_ms,
                     "llm_raw_response": res.raw or (res.error or ""),
                     "prompt_version": prompt_f1.PROMPT_VERSION,
                     "model_name": llm.model, "provider": llm.provider,
                     "finish_reason": res.finish_reason,
                     "completion_tokens": res.completion_tokens})
    else:
        for p in persons:
            rows.append({
                "request_id": request_id, "object_id": p.id,
                "camera_id": body.camera_id, "ts": now, "status": p.status,
                "direction": p.direction, "direction_confidence": p.direction_confidence,
                "gender": p.gender, "gender_confidence": p.gender_confidence,
                "age_range": p.age_range, "age_range_confidence": p.age_range_confidence,
                "appearance": p.appearance.model_dump_json(),
                "appearance_confidence": p.appearance_confidence,
                "description": p.description, "overall_confidence": p.overall_confidence,
                # `where` ไปรวมกับ reason ในฐานข้อมูล ไม่ได้เพิ่มคอลัมน์ใหม่
                # ตารางนี้เป็นของสำหรับไล่ดูย้อนหลัง ไม่ใช่สัญญากับปลายทาง
                "reason": (f"where: {p.where}" if p.where else ""),
                "image_w": frame_w, "image_h": frame_h, "image_source": "full_frame",
                "image_path": None,
                "llm_raw_response": res.raw,
                "prompt_version": prompt_f1.PROMPT_VERSION,
                "model_name": llm.model, "provider": llm.provider,
                "finish_reason": res.finish_reason,
                "completion_tokens": res.completion_tokens,
                "latency_ms": res.latency_ms})

    total_ms = round((time.perf_counter() - t_start) * 1000, 1)
    store.insert_persons(rows)
    store.finish_request(request_id, status, total_ms)

    return PersonsOut(
        request_id=request_id, client_request_id=body.client_request_id,
        note=body.note, camera_id=body.camera_id, received_at=_iso(now),
        status=status, persons=persons, reason=why,
        # objects_in = 0 เพราะเส้นนี้ไม่มีใครถูกส่งมาให้ตรวจ · people_found คือของจริง
        summary={"objects_in": 0, "ok": len(persons), "degraded": 0, "error": 0,
                 "people_found": len(persons), **_direction_counts(persons)},
        model=model_info,
        timing_ms={"decode": decode_ms, "vlm": vlm_ms, "vlm_sum": res.latency_ms,
                   "total": total_ms},
    )


@app.get("/v1/persons/by-ref/{client_request_id}")
def get_by_client_ref(client_request_id: str,
                      x_api_key: Optional[str] = Header(default=None)):
    """ตามผลด้วยเลขอ้างอิงของปลายทางเอง

    🔴 เส้นนี้มีไว้สำหรับตอน **ยิงไปแล้วไม่รู้ผล** (timeout, เน็ตหลุด, แอปถูกปิด)
    ซึ่งเป็นตอนเดียวที่ `request_id` ของเราช่วยอะไรไม่ได้ เพราะเขาไม่เคยได้เห็นมัน

    ส่งซ้ำ id เดิมมาหลายครั้ง = มีหลายผล คืน**อันล่าสุด** และบอกจำนวนที่เจอ
    เราไม่การันตีว่าไม่ซ้ำ เพราะเลขนี้ไม่ใช่ของเรา
    """
    _auth(x_api_key)
    rows = store.by_client_ref(client_request_id)
    if not rows:
        raise HTTPException(status_code=404,
                            detail="ไม่เคยเห็น client_request_id นี้")
    latest = store.get_request(rows[0]["request_id"])
    return JSONResponse({"client_request_id": client_request_id,
                         "matches": len(rows),
                         "latest": latest})


@app.get("/v1/persons/{request_id}")
def get_persons(request_id: str, x_api_key: Optional[str] = Header(default=None)):
    _auth(x_api_key)
    row = store.get_request(request_id)
    if not row:
        raise HTTPException(status_code=404, detail="unknown request_id")
    return JSONResponse(row)


@app.post("/v1/persons/{request_id}/{object_id}/truth")
def post_truth(request_id: str, object_id: str, body: PersonTruthIn,
               x_api_key: Optional[str] = Header(default=None)):
    """ค่าที่คนตรวจแล้ว · เก็บตอนนี้เท่านั้น ย้อนไปเก็บทีหลังไม่ได้"""
    _auth(x_api_key)
    if not store.save_truth(request_id, object_id, body.gender, body.age_range,
                            body.reviewer, body.comment):
        raise HTTPException(status_code=404,
                            detail="ไม่รู้จัก request_id คู่กับ object_id นี้")
    return {"ok": True, "request_id": request_id, "object_id": object_id}


@app.post("/v1/maintenance/prune")
def prune(x_api_key: Optional[str] = Header(default=None)):
    """ลบข้อมูลที่เกินอายุ · เรียกจาก cron ข้างนอก ไม่ใช่ background task ในนี้"""
    _auth(x_api_key)
    return store.prune(int(os.environ.get("PEOPLE_RETENTION_DAYS", "7")))


# หน้าเว็บทดสอบ · ไฟล์นิ่งล้วน ยิง POST /people/v1/persons เหมือน client ทั่วไป
# และโดน 401 เหมือนใครก็ตามถ้าไม่ใส่คีย์ · ห้ามให้มันข้าม auth
_UI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")
if os.path.isdir(_UI_DIR):
    app.mount("/verify", StaticFiles(directory=_UI_DIR, html=True), name="people-verify")
else:
    print(f"[people] ไม่พบ {_UI_DIR} · /people/verify จะตอบ 404", flush=True)
