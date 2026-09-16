"""FastAPI ของงานอ่านป้ายทะเบียนไทย (LPR) · รับภาพ ตอบทะเบียน + จังหวัด + ชนิดรถ

แอปตัวนี้ถูก mount ไว้ใต้ `/lpr` ของแอปเดิม (ดูท้าย `app/main.py`)
endpoint จริงจึงเป็น:

    POST /lpr/v1/vehicle                 เส้นหลัก · หนึ่งภาพ ตอบทุกป้ายที่เห็น
    GET  /lpr/v1/vehicle/{request_id}    ดูผลย้อนหลัง
    GET  /lpr/v1/vehicle/by-ref/{id}     ตามผลด้วยเลขอ้างอิงของปลายทางเอง
    POST /lpr/v1/vehicle/{request_id}/{ref}/truth   ทะเบียนจริงที่คนตรวจแล้ว
    GET  /lpr/v1/accuracy                อ่านถูกกี่แถวจากแถวที่ตรวจแล้ว
    POST /lpr/v1/maintenance/prune       ลบของเกินอายุ (เรียกจาก cron ข้างนอก)
    GET  /lpr/healthz
    GET  /lpr/verify/                    หน้าเว็บทดสอบด้วยตา

🔴 ทำไม mount ใต้แอปเดิม ไม่แยกคอมโพเนนต์ใหม่บน App Platform
เหตุผลเดียวกับงานคนเป๊ะ: **App Platform ไม่อ่าน `.do/app.yaml` ของแอปที่สร้างไว้แล้ว**
เจอมาแล้ว 2026-08-17 · ทางที่ deploy ได้เองจริงโดยไม่ต้องกดอะไรใน Console
คือ mount เข้าแอปที่มีอยู่ Procfile เดิม run_command เดิม env เดิม

🔴 ทำไม path เป็น `/v1/vehicle` (เอกพจน์) แต่คำตอบเป็น `vehicles[]` (พหูพจน์)
ชื่อเส้นมาจากคำสั่ง Toy 2026-09-14 และจากการใช้งานจริงคือ "รถคันหนึ่งมาถึงประตู"
แต่ในเฟรมเดียวกันมีรถคันอื่นจอดอยู่ข้างหลังได้เสมอ · ถ้าบังคับให้ตอบคันเดียว
เราต้องเลือกแทนปลายทางว่าคันไหนคือ "คันที่ใช่" ซึ่งเราไม่รู้ · ตอบทุกคันที่เห็น
พร้อม `where` แล้วให้เขาเลือกเอง คือคำตอบที่ซื่อสัตย์กว่าและแก้ทีหลังได้

🔴 บริการนี้ไม่เทียบ watchlist ไม่เตือนใคร ไม่จำว่ารถคันไหนเคยมา
รับภาพ ตอบ JSON จบ · ห้ามเพิ่ม webhook / alert / ตาราง "รถที่เคยเห็น" ลงในไฟล์นี้
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import io
import os
import time
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .llm import prompt_v1
from .llm.client import (ALLOW_SHORT_DIGITS, PlateLLM, normalize,
                         vehicles_of)
from .schemas import (ImageInfo, ModelInfo, PlatePattern, VehicleIn, VehicleOut,
                      VehiclesOut, VehicleTruthIn)
from .store import PlateStore
from .th import thai

# 🔴 เลขนี้ต้องขยับทุกครั้งที่พฤติกรรมของ endpoint ฝั่งนี้เปลี่ยน
# แยกจาก APP_VERSION (ช้าง) และ PEOPLE_VERSION (คน) โดยตั้งใจ
# สามงาน deploy ไปด้วยกันก็จริง แต่ปลายทางคนละทีม ต้องตอบได้แยกกันว่า
# "ของที่คุณเรียกอยู่เวอร์ชันอะไร" · tools/ship.py ตรวจทั้งสามตัว
LPR_VERSION = "0.4.0"
BUILD_NOTES = ("thai licence plate OCR via VLM · plate split in code, not asked"
               " · 77-province closed set · prompt v1")

app = FastAPI(title="thai-lpr", version=LPR_VERSION)


def _writable(preferred: str, fallback: str, is_dir: bool = False) -> str:
    """หา path ที่เขียนได้จริง ไม่ใช่เชื่อว่าเขียนได้

    บทเรียนเดียวกับอีกสองงาน 2026-08-16: โฟลเดอร์แอปบน buildpack เขียนไม่ได้
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
        print(f"[lpr] {preferred!r} เขียนไม่ได้ ({type(e).__name__}: {e}) "
              f"-> ใช้ {fallback!r} แทน", flush=True)
        return fallback


STORE_DSN = _writable(os.environ.get("LPR_STORE_DSN", "./data/lpr.db"),
                      "/tmp/animalcountllm/lpr.db")

STORE_ERROR: Optional[str] = None
try:
    store = PlateStore(STORE_DSN)
    store.init_schema()
except Exception as e:  # noqa: BLE001
    STORE_ERROR = f"{type(e).__name__}: {e}"
    print(f"[lpr] 🔴 เปิดฐานข้อมูลไม่ได้: {STORE_ERROR}", flush=True)
    store = PlateStore("/tmp/lpr-fallback.db")
    store.init_schema()

llm = PlateLLM()

# คีย์: ใช้ตัวเดียวกับงานอื่นเป็นค่าตั้งต้น แต่แยก env ไว้แล้ว
# วันที่ปลายทางของงานนี้ได้คีย์ไป แล้วอยากถอนสิทธิ์งานอื่น จะได้ไม่ต้องแก้โค้ด
API_KEY = (os.environ.get("LPR_API_KEY") or os.environ.get("API_KEY", "")).strip()

# ไม่เก็บภาพเป็นค่าเริ่มต้น · ภาพงานนี้มีทะเบียนรถของคนจริงอยู่ในนั้น
SAVE_IMAGES = os.environ.get("LPR_SAVE_IMAGES", "none").lower()
IMAGE_DIR = _writable(os.environ.get("LPR_IMAGE_DIR", "./data/lpr-crops"),
                      "/tmp/animalcountllm/lpr-crops", is_dir=True)

# กรอบป้ายจาก ALPR ปลายทางมักรัดพอดีขอบป้าย แล้วตัดตัวอักษรริมสุดทิ้ง
# ซึ่งเป็นตัวที่ทำให้ทะเบียนทั้งพวงผิด · เผื่อขอบไว้ 10%
BBOX_MARGIN = float(os.environ.get("LPR_BBOX_MARGIN", "0.10"))

# ⚠️ ยัง **ไม่ได้วัด** ว่าการขยายภาพช่วยให้อ่านตัวอักษรแม่นขึ้นจริงไหม
# ที่วัดไปแล้วฝั่งช้าง: prompt token เท่ากันหมดที่ 640x512 แปลว่าภาพใหญ่ขึ้น
# ไม่ได้แปลว่าโมเดลได้ image token เพิ่มเสมอไป · ตั้งไว้กัน provider ปฏิเสธภาพจิ๋ว
# (ครอปป้ายจากกล้องไกลๆ เหลือ 60x25 px ได้สบาย)
MIN_CROP_PX = int(os.environ.get("LPR_MIN_CROP_PX", "224"))

# 🔴 หนึ่งภาพ หนึ่งคัน · Toy สั่ง 2026-09-16 ("เจตนาส่งคันเดียว")
#
# ปลายทางส่งภาพมาถามถึงรถคันเดียว การตอบรถที่จอดอยู่ข้างหลังมาด้วยไม่ได้ช่วยอะไร
# แต่ทำให้ปลายทางต้องเขียนโค้ดเลือกเอง ซึ่งเป็นการเลือกที่เราทำได้ดีกว่า
# เพราะเราเห็น `plate.pattern` และ `plate.confidence` ของทุกคันพร้อมกัน
#
# ยังเป็นค่า env ไม่ใช่เลข 1 ตายตัวในโค้ด เพราะวันที่ปลายทางอยากได้ทั้งลานจอด
# (ยิงจากกล้อง fix ตัวเดียว) จะได้เปลี่ยนค่าเดียว ไม่ต้องรื้อ
MAX_VEHICLES = int(os.environ.get("LPR_MAX_VEHICLES", "1"))

# กี่คันที่เรายอม "ดู" ก่อนคัดเหลือ MAX_VEHICLES · ไม่ใช่ตัวเดียวกัน
# โมเดลไม่ทำตามคำสั่งแล้วส่งมาห้าคัน เราต้องเลือกคันที่ชัดที่สุดจากห้าคันนั้น
# **ไม่ใช่หยิบคันแรกที่มันพิมพ์ออกมา** ลำดับที่มันพิมพ์ไม่ใช่ลำดับความชัด
LPR_SCAN_CAP = int(os.environ.get("LPR_SCAN_CAP", "6"))

# 🔴 ปลายทางไม่ต้องรู้ว่าเราใช้โมเดลอะไร prompt เวอร์ชันไหน (กติกาเดียวกับงานคน)
# คีย์ `model` ยังอยู่ในคำตอบเสมอแต่เป็น null · **ไม่ถอดคีย์ทิ้ง** รูป response
# ที่เปลี่ยนตามค่า env คือของที่ทำให้ปลายทาง parse พังแบบหาสาเหตุไม่เจอ
EXPOSE_MODEL = os.environ.get("LPR_EXPOSE_MODEL", "false").lower() == "true"

print(f"[lpr] v{LPR_VERSION} store={STORE_DSN} provider={llm.provider} "
      f"model={llm.model} save_images={SAVE_IMAGES}", flush=True)


@app.exception_handler(RequestValidationError)
def explain_validation_error(request: Request, exc: RequestValidationError):
    """422 ของ FastAPI บอกว่าฟิลด์ไหนหาย แต่ไม่บอกว่า "คุณมาผิดเส้น"

    งานคนเจอมาแล้วว่าปลายทางยิงผิดเส้นเป็นเรื่องปกติเมื่อ repo มีหลายบริการ
    ที่ path คล้ายกัน · `detail` ยังเป็นรูปเดิมเป๊ะเพื่อให้โค้ดที่ parse อยู่ไม่พัง
    เพิ่มแค่ `hint` ที่บอกทางให้คน
    """
    errors = exc.errors()
    body = exc.body if isinstance(exc.body, dict) else {}
    missing = {tuple(e.get("loc", ()))[-1] for e in errors if e.get("type") == "missing"}
    hint = ""
    if "image_base64" in missing and body.get("objects"):
        hint = ("body นี้เป็นรูปของ POST /people/v1/persons (งานนับคน) "
                "แต่ยิงมาที่ /lpr/v1/vehicle ซึ่งรับภาพเดียวชื่อ image_base64")
    elif "image_base64" in missing and body.get("frame_base64"):
        hint = "เส้นนี้ใช้ชื่อ image_base64 ไม่ใช่ frame_base64"
    return JSONResponse(status_code=422,
                        content=jsonable_encoder({"detail": errors,
                                                  **({"hint": hint} if hint else {})}))


def _auth(key: Optional[str]) -> None:
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=401, detail="bad or missing X-API-Key")


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


def _ref_of(body: VehicleIn) -> Optional[str]:
    """เลขอ้างอิงของปลายทาง · `client_request_id` ก่อน ไม่มีค่อยใช้ `note`

    ช่องเดียวเท่านั้น ไม่คืน `note` ซ้ำอีกช่อง · ค่าเดียวกันสองชื่อคือที่ที่
    คนอ่านเอกสารจะเดาผิดว่ามันต่างกันตรงไหน (กติกาเดียวกับงานคน)

    🔴 **ไม่ตีความ ไม่ trim ไม่ตัด ไม่แปลง · คืนตัวเดิมเป๊ะ**
    ของที่ปลายทางเอาไปเทียบว่า "ใช่ของฉันไหม" ห้ามถูกเราแตะระหว่างทาง
    เวอร์ชันแรกของไฟล์นี้ `.strip()` แล้วตัดที่ 100 ตัว ซึ่งดูไม่มีพิษภัย
    จนกระทั่งปลายทางส่ง note ยาว 120 ตัวมาแล้วเทียบกับของตัวเองไม่ตรง
    **แล้วไม่มีอะไรบอกเขาเลยว่าเราเป็นคนตัด** (`note` รับได้ถึง 500 ตัวตาม schema
    ส่วน `client_request_id` ถูกจำกัด 100 ตัวตั้งแต่ชั้น schema อยู่แล้ว
    ซึ่งเป็นที่ที่ควรจำกัด: ปฏิเสธตรงๆ ดีกว่ารับแล้วแอบตัด)
    งานคนเขียนกฎข้อนี้ไว้ก่อนแล้วและมีเทสต์เฝ้า ที่นี่เพิ่งตามมาให้ตรงกัน
    """
    return body.client_request_id or body.note


def _decode(b64: str, what: str) -> bytes:
    try:
        return base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail=f"{what} decode failed")


def _size_of(raw: bytes) -> Tuple[int, int]:
    from PIL import Image
    with Image.open(io.BytesIO(raw)) as im:
        return im.size


def _crop(raw: bytes, bbox: List[int]) -> Tuple[str, int, int, List[int]]:
    """ครอปตาม bbox แล้วคืน base64 JPEG · กรอบล้นขอบถูกหนีบ ไม่ใช่โยน error

    ALPR ปลายทางคืนกรอบล้นขอบเป็นเรื่องปกติเวลารถกำลังออกจากเฟรม
    """
    from PIL import Image

    with Image.open(io.BytesIO(raw)) as im:
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
                                max(1, int(crop.height * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=92)
        return (base64.b64encode(buf.getvalue()).decode(), crop.width, crop.height,
                [x1, y1, x2, y2])


def _save_image(raw_b64: str, request_id: str, status: str) -> None:
    """เก็บภาพลงดิสก์เฉพาะตอนที่ตั้งใจให้เก็บ · ค่าเริ่มต้นคือไม่เก็บ

    none     ไม่เก็บเลย (ค่าเริ่มต้น · ภาพมีทะเบียนรถของคนจริงอยู่)
    degraded เก็บเฉพาะภาพที่อ่านไม่ออก ซึ่งคือชุดที่มีค่าตอนดีบักจริงๆ
    all      ใช้ตอนไล่บั๊กเท่านั้น อย่าเปิดค้างไว้บน prod
    """
    if not raw_b64 or SAVE_IMAGES not in {"all", "degraded"}:
        return
    if SAVE_IMAGES == "degraded" and status == "ok":
        return
    try:
        os.makedirs(IMAGE_DIR, exist_ok=True)
        with open(os.path.join(IMAGE_DIR, f"{request_id}.jpg"), "wb") as f:
            f.write(base64.b64decode(raw_b64))
    except (OSError, binascii.Error):
        return  # เก็บภาพไม่ได้ ไม่ควรทำให้ทั้ง request พัง


def _model_info(res=None) -> Optional[ModelInfo]:
    if not EXPOSE_MODEL:
        return None
    return ModelInfo(provider=llm.provider, name=llm.model,
                     prompt_version=prompt_v1.PROMPT_VERSION,
                     finish_reason=getattr(res, "finish_reason", None),
                     completion_tokens=getattr(res, "completion_tokens", None))


def _clearest(vehicles: List[VehicleOut], keep: int) -> List[VehicleOut]:
    """คัดให้เหลือคันที่ชัดที่สุด · Toy สั่ง 2026-09-16 ("เจตนาส่งคันเดียว")

    ลำดับการคัด สำคัญกว่าที่คิด:
      1. **อ่านทะเบียนเข้ารูปได้** ก่อนเสมอ · คันที่อ่านได้แต่เบลอกว่า มีค่ากว่า
         คันที่ชัดกว่าแต่ป้ายโดนบัง เพราะบริการนี้ตอบเรื่องทะเบียน ไม่ใช่เรื่องรถ
      2. `plate.confidence` สูงกว่า
      3. ลำดับที่โมเดลพิมพ์มา (เท่ากันจริงๆ ค่อยใช้)

    🔴 **ไม่ใช่หยิบตัวแรกที่โมเดลพิมพ์** ลำดับที่มันพิมพ์คือลำดับที่มันนึกออก
    ไม่ใช่ลำดับความชัด · prompt ขอคันเดียวอยู่แล้ว ตรงนี้คือด่านกันวันที่มันไม่ทำตาม
    ซึ่งเกิดแน่ เหมือนที่ฝั่งงานคนเจอมาแล้วทุกเรื่องที่ฝากไว้กับ prompt อย่างเดียว
    """
    if keep <= 0 or len(vehicles) <= keep:
        return vehicles
    ranked = sorted(enumerate(vehicles),
                    key=lambda t: (0 if t[1].plate.text else 1,
                                   -t[1].plate.confidence, t[0]))
    # เรียงกลับตามลำดับเดิมของคันที่เลือกไว้ ปลายทางอ่านง่ายกว่า
    picked = sorted(i for i, _ in ranked[:keep])
    return [vehicles[i] for i in picked]


def _literal(ann) -> tuple:
    """ค่าใน Literal · ยอด `pattern_*` ประกอบจาก schema ไม่ได้พิมพ์ชื่อซ้ำไว้ที่นี่
    เพิ่มรูปใหม่ใน `PlatePattern` แล้วยอดจะมีถังของมันเองทันที ไม่ต้องมาแก้สองที่
    """
    import typing
    return typing.get_args(ann)


def _summary(vehicles: List[VehicleOut]) -> dict:
    """ยอดรวม · **ทุกชุดบวกได้ครบจำนวนคัน และทุกชุดมีถัง unknown**

    กฎเดียวกับ `direction` และ `mobility` ของงานคนเป๊ะ ไม่ใช่กฎใหม่
    ยอดที่ขาดไปเงียบๆ ทำให้ปลายทางไล่หารถที่หายไป โดยที่ไม่มีรถคันไหนหายจริง

    🔴 `plates_read` + `plates_unread` = จำนวนคัน · **ตัวหลังสำคัญกว่าตัวหน้า**
    มันคือตัวเดียวที่บอกว่าบริการนี้ใช้ได้จริงแค่ไหนในไซต์นั้น ก่อนจะมี /truth
    ซ่อนมันเมื่อไหร่ กราฟจะสวยขึ้นทันทีโดยที่ความจริงไม่ขยับ
    """
    # 🔴 วัดจาก `text` ไม่ใช่ `letters` · รูปรถบรรทุก (`10-0001`) ไม่มีตัวอักษรเลย
    # แต่เป็นทะเบียนที่อ่านได้เต็มตัว · นับด้วย letters เมื่อไหร่ ยอดรถบรรทุกทั้งไซต์
    # จะกลายเป็น "อ่านไม่ได้" ทั้งที่อ่านถูกทุกคัน (แก้ 2026-09-16)
    n_read = sum(1 for v in vehicles if v.plate.text)
    types = {}
    for t in ("car", "pickup", "motorcycle", "truck", "van", "bus", "other",
              "unknown"):
        types[f"type_{t}"] = sum(1 for v in vehicles if v.vehicle_type == t)
    return {
        "vehicles_found": len(vehicles),
        # 🔴 กี่คันที่เข้ารูปทะเบียนจริง · `plates_read` นับ "แยกได้" ตัวนี้นับ "เข้ารูป"
        # สองตัวนี้เท่ากันเสมอตั้งแต่ 2026-09-16 (ด่านเดียวกันแล้ว) และจะไม่เท่ากัน
        # ทันทีที่ใครผ่อนด่านใดด่านหนึ่ง ซึ่งเป็นวันที่ต้องมองเห็น
        **{f"pattern_{name}": sum(1 for v in vehicles if v.plate.pattern == name)
           for name in _literal(PlatePattern)},
        "ok": sum(1 for v in vehicles if v.status == "ok"),
        "degraded": sum(1 for v in vehicles if v.status == "degraded"),
        "plates_read": n_read,
        "plates_unread": len(vehicles) - n_read,
        "province_known": sum(1 for v in vehicles if v.province),
        "province_unknown": sum(1 for v in vehicles if not v.province),
        **types,
    }


def _out(payload: VehiclesOut) -> JSONResponse:
    """ตอบออกไปเป็นไทย · **คีย์อังกฤษ ค่าไทย** (กติกาเดียวกับงานคน)

    แปลตรงนี้ที่เดียว หลังประกอบคำตอบเสร็จ และ **หลังเขียนลงฐานแล้ว**
    ของที่ลงฐานยังเป็นอังกฤษชุดค่าปิดเหมือนเดิม ดูเหตุผลยาวใน `lpr/th.py`

    คืน `JSONResponse` ไม่ใช่ `VehiclesOut` เพราะค่าไทยไม่ผ่าน `Literal` ใน schema
    ซึ่งถูกต้องแล้ว: schema คือความจริงฝั่งใน ไม่ใช่รูปของสายที่ส่งออก
    """
    return JSONResponse(thai(jsonable_encoder(payload)))


# ---------------------------------------------------------------- endpoints
@app.get("/healthz")
def healthz():
    """ต้องตอบได้แม้ของบางอย่างพัง ไม่งั้นเห็นแค่ connection refused"""
    return {"status": "ok" if not STORE_ERROR else "degraded",
            "service": "thai-lpr",
            "version": LPR_VERSION, "build": BUILD_NOTES,
            "provider": llm.provider if EXPOSE_MODEL else None,
            "model": llm.model if EXPOSE_MODEL else None,
            "prompt_version": prompt_v1.PROMPT_VERSION if EXPOSE_MODEL else None,
            # 🔴 ไม่ใช่ข้อมูลโมเดล จึงไม่ถูกซ่อนตาม EXPOSE_MODEL
            # มันบอกว่า "คำตอบที่คุณได้รับแปลว่าอะไร" ซึ่งปลายทางต้องรู้เสมอ
            "province_set": "77 จังหวัด · ชื่อนอกชุดคืนค่าว่าง ไม่ใช่ค่าใกล้เคียง",
            "plate_split": "แยกในโค้ดจาก plate_text ไม่ได้ถามโมเดลแยกช่อง",
            "max_vehicles": MAX_VEHICLES,
            "scan_cap": LPR_SCAN_CAP,
            # ปลายทางต้องรู้ว่า "อ่านไม่ออก" ของบริการนี้รวมถึงอะไรบ้าง
            "plate_patterns": "LL DDDD · D LL DDDD · LLL DDD (จยย.)"
                              " · DD-DDDD / DDD-DDDD (รถบรรทุก/โดยสาร)"
                              " · รูปอื่นคืนค่าว่างพร้อมเหตุผล ไม่เดาให้",
            # ปลายทางต้องรู้ว่าเลขท้ายสั้นกว่าสี่หลักถูกรับหรือถูกตีตก
            "allow_short_digits": ALLOW_SHORT_DIGITS,
            "bbox_margin": BBOX_MARGIN, "min_crop_px": MIN_CROP_PX,
            "save_images": SAVE_IMAGES,
            "store_path": STORE_DSN, "store_error": STORE_ERROR,
            "auth_required": bool(API_KEY), "tracing": llm.tracing}


@app.post("/v1/vehicle")
def post_vehicle(body: VehicleIn, x_api_key: Optional[str] = Header(default=None)):
    """หนึ่งภาพ ตอบทุกป้ายที่อ่านได้ในภาพนั้น

    ลำดับงาน (เหมือนอีกสองงาน: **บันทึกก่อน ทำทีหลัง**)
      [1] decode + ครอปตาม bbox ถ้ามี
      [2] บันทึก request ลงฐานทันที ก่อนยิงโมเดล
      [3] ยิง VLM หนึ่งครั้งต่อหนึ่งภาพ
      [4] normalize + แยกทะเบียนในโค้ด
      [5] บันทึกผล + ตอบกลับ
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
        img_w, img_h = _size_of(raw_bytes)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"อ่านภาพไม่ได้: {type(e).__name__}")

    b64 = body.image_base64
    info = ImageInfo(w=img_w, h=img_h, source="full_image")
    if body.bbox:
        if len(body.bbox) != 4:
            raise HTTPException(status_code=400,
                                detail="bbox ต้องเป็น [x, y, w, h] สี่ค่า")
        try:
            b64, cw, ch, used = _crop(raw_bytes, body.bbox)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        info = ImageInfo(w=cw, h=ch, source="bbox_crop", bbox_used=used)
    decode_ms = round((time.perf_counter() - t_decode) * 1000, 1)

    store.insert_request(request_id, body.camera_id, now, body.note,
                         body.client_request_id)
    image_hash = hashlib.sha256(b64.encode()).hexdigest()[:32]

    t_vlm = time.perf_counter()
    res = llm.read(b64, info.w, info.h, body.camera_id, image_hash, MAX_VEHICLES)
    vlm_ms = round((time.perf_counter() - t_vlm) * 1000, 1)

    vehicles: List[VehicleOut] = []
    rows = []

    if not res.usable:
        # 🔴 โมเดลล่ม/ตอบไม่จบ = degraded พร้อมรายชื่อว่าง
        # **ห้ามตอบ ok + ศูนย์คัน** เพราะนั่นแปลว่า "ไม่มีป้ายในภาพ" ซึ่งคนละเรื่อง
        # บั๊กเดียวกับที่ฝั่งช้างเสียเวลาไปทั้งวันตอนภาพ RGB ได้ 200 OK counts ว่าง
        status = "degraded"
        why = res.error or (f"โมเดลตอบไม่จบ (finish={res.finish_reason})"
                            if res.finish_reason == "length" else "โมเดลตอบไม่เป็น JSON")
    else:
        found, err = vehicles_of(res.data, LPR_SCAN_CAP)
        if err:
            status, why = "degraded", err
        else:
            status, why = "ok", ""
            for ref, item in found:
                n = normalize(item)
                # ความมั่นใจรวม = เฉลี่ยของสามคำตอบหลัก (ทะเบียน จังหวัด ชนิดรถ)
                # ⚠️ ตัวที่สำคัญที่สุดคือ `plate.confidence` เสมอ ตัวนี้มีไว้เรียงลำดับ
                # ไม่ได้มีไว้ตัดสินใจ · ป้ายที่อ่านไม่ออกดึงค่านี้ลงถูกต้องแล้ว
                conf = round((n["plate"].confidence + n["province_confidence"]
                              + n["vehicle_type_confidence"]) / 3, 3)
                vehicles.append(VehicleOut(ref=ref, status="ok",
                                           overall_confidence=conf,
                                           timing_ms=res.latency_ms, **n))
            seen = len(vehicles)
            vehicles = _clearest(vehicles, MAX_VEHICLES)
            dropped = seen - len(vehicles)
            if dropped:
                # 🔴 คันที่ถูกคัดออกต้องมองเห็นในคำตอบ ไม่ใช่หายเงียบ
                # ปลายทางต้องรู้ว่าในภาพมีรถคันอื่นอยู่ด้วย ไม่งั้นวันที่เราเลือกผิดคัน
                # เขาจะไม่มีวันรู้ว่ามีให้เลือก
                why = f"ในภาพมีรถที่เห็นป้ายอีก {dropped} คัน ตอบเฉพาะคันที่ชัดที่สุด"

    if status == "degraded":
        # ไม่มีรายคันให้รายงาน · เหตุผลอยู่ที่ระดับ request ไม่ใช่ซ่อนในคันที่ไม่มี
        rows.append({"request_id": request_id, "ref": "-",
                     "camera_id": body.camera_id, "ts": now, "status": "degraded",
                     "reason": why, "latency_ms": res.latency_ms,
                     "llm_raw_response": res.raw or (res.error or ""),
                     "prompt_version": prompt_v1.PROMPT_VERSION,
                     "model_name": llm.model, "provider": llm.provider,
                     "finish_reason": res.finish_reason,
                     "completion_tokens": res.completion_tokens,
                     "image_w": info.w, "image_h": info.h,
                     "image_source": info.source})
    else:
        for v in vehicles:
            rows.append({
                "request_id": request_id, "ref": v.ref,
                "camera_id": body.camera_id, "ts": now, "status": v.status,
                "plate_text": v.plate.text, "plate_text_raw": v.plate.text_raw,
                "plate_prefix": v.plate.prefix, "plate_letters": v.plate.letters,
                "plate_digits": v.plate.digits, "plate_pattern": v.plate.pattern,
                "plate_color": v.plate.color,
                "plate_confidence": v.plate.confidence,
                "province": v.province,
                "province_confidence": v.province_confidence,
                "vehicle_type": v.vehicle_type,
                "vehicle_type_confidence": v.vehicle_type_confidence,
                "vehicle_color": v.vehicle_color,
                "vehicle_make": v.vehicle_make,
                "vehicle_make_confidence": v.vehicle_make_confidence,
                "vehicle_model": v.vehicle_model,
                "vehicle_model_confidence": v.vehicle_model_confidence,
                "vehicle_generation": v.vehicle_generation,
                "vehicle_generation_confidence": v.vehicle_generation_confidence,
                "description": v.description,
                "overall_confidence": v.overall_confidence,
                "reason": v.reason, "where_text": v.where,
                "image_w": info.w, "image_h": info.h, "image_source": info.source,
                "llm_raw_response": res.raw,
                "prompt_version": prompt_v1.PROMPT_VERSION,
                "model_name": llm.model, "provider": llm.provider,
                "finish_reason": res.finish_reason,
                "completion_tokens": res.completion_tokens,
                "latency_ms": res.latency_ms})

    total_ms = round((time.perf_counter() - t_start) * 1000, 1)
    store.insert_vehicles(rows)
    store.finish_request(request_id, status, total_ms)
    _save_image(b64, request_id, status)

    return _out(VehiclesOut(
        request_id=request_id, ref=_ref_of(body), camera_id=body.camera_id,
        received_at=_iso(now), status=status, reason=why, vehicles=vehicles,
        summary=_summary(vehicles), image=info, model=_model_info(res),
        timing_ms={"decode": decode_ms, "vlm": vlm_ms, "total": total_ms},
    ))


@app.get("/v1/vehicle/by-ref/{client_request_id}")
def get_by_client_ref(client_request_id: str,
                      x_api_key: Optional[str] = Header(default=None)):
    """ตามผลด้วยเลขอ้างอิงของปลายทางเอง

    🔴 มีไว้สำหรับตอน **ยิงไปแล้วไม่รู้ผล** (timeout, เน็ตหลุด) ซึ่งเป็นตอนเดียว
    ที่ `request_id` ของเราช่วยอะไรไม่ได้ เพราะปลายทางไม่เคยได้เห็นมัน
    ส่งซ้ำ id เดิมมาหลายครั้ง = มีหลายผล คืนอันล่าสุด และบอกจำนวนที่เจอ
    """
    _auth(x_api_key)
    rows = store.by_client_ref(client_request_id)
    if not rows:
        raise HTTPException(status_code=404, detail="ไม่เคยเห็นเลขอ้างอิงนี้")
    latest = store.get_request(rows[0]["request_id"])
    return JSONResponse(thai(jsonable_encoder(
        {"found": len(rows), "client_request_id": client_request_id, **latest})))


@app.get("/v1/vehicle/{request_id}")
def get_request(request_id: str, x_api_key: Optional[str] = Header(default=None)):
    _auth(x_api_key)
    data = store.get_request(request_id)
    if not data:
        raise HTTPException(status_code=404, detail="unknown request_id")
    return JSONResponse(thai(jsonable_encoder(data)))


@app.post("/v1/vehicle/{request_id}/{ref}/truth")
def post_truth(request_id: str, ref: str, body: VehicleTruthIn,
               x_api_key: Optional[str] = Header(default=None)):
    """ทะเบียนจริงที่คนตรวจแล้ว · **เก็บตั้งแต่วันแรก**

    ต่างจากงานอารมณ์และงานสัญชาติของฝั่งคน ตรงที่งานนี้มีคำตอบที่ถูกเพียงคำตอบเดียว
    และคนเปิดภาพดูก็ตัดสินได้ · ไม่เก็บตั้งแต่วันแรก = อีกสามเดือนยังตอบไม่ได้ว่า
    บริการนี้อ่านถูกกี่เปอร์เซ็นต์ แล้วจะเถียงกันด้วยความรู้สึกแทนตัวเลข
    """
    _auth(x_api_key)
    if not store.save_truth(request_id, ref, body.plate, body.province,
                            body.vehicle_type, body.reviewer, body.comment):
        raise HTTPException(status_code=404, detail="unknown request_id/ref")
    return {"ok": True, "request_id": request_id, "ref": ref}


@app.get("/v1/accuracy")
def accuracy(x_api_key: Optional[str] = Header(default=None)):
    """อ่านถูกกี่แถวจากแถวที่มีคนตรวจแล้ว

    🔴 `exact` เทียบแบบตรงตัวเป๊ะ ไม่มี "ใกล้เคียง" · ทะเบียนที่ผิดหนึ่งตัว
    คือทะเบียนของรถคันอื่น ไม่ใช่คำตอบที่ถูก 90%
    `unread` แยกออกมาต่างหากโดยตั้งใจ: อ่านไม่ออกกับอ่านผิด คนละปัญหา
    และแก้คนละวิธี (ตัวแรกแก้ที่กล้อง/มุม ตัวหลังแก้ที่โมเดล/prompt)
    """
    _auth(x_api_key)
    return store.accuracy()


@app.post("/v1/maintenance/prune")
def prune(x_api_key: Optional[str] = Header(default=None)):
    """ลบของเกินอายุ · เรียกจาก cron ข้างนอก ไม่ใช่ background task ในนี้
    เพราะ container ตายเมื่อไหร่ก็ได้ แล้ว task ที่ค้างอยู่ก็หายไปเงียบๆ
    """
    _auth(x_api_key)
    return store.prune(int(os.environ.get("LPR_RETENTION_DAYS", "7")))


# หน้าเว็บทดสอบ · เสิร์ฟจาก FastAPI ไม่ใช่ static_sites ของ App Platform
# (ไฟล์ .do/app.yaml ไม่ถูกอ่านสำหรับแอปที่สร้างไว้แล้ว · เจอมาแล้ว 2026-08-17)
_UI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")
if os.path.isdir(_UI):
    app.mount("/verify", StaticFiles(directory=_UI, html=True), name="lpr-verify")
