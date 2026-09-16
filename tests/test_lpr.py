"""เทสต์ของงานอ่านป้ายทะเบียนไทย (LPR)

🔴 ไม่ยิงโมเดลจริงสักข้อ · ทุกข้อใช้ `PlateLLM` ปลอมที่ยัดคำตอบดิบเข้าไปเอง
เทสต์ที่ต้องมี GPU หรือมีเน็ต คือเทสต์ที่ไม่มีใครรัน แล้วมันจะเน่าเงียบๆ

🔴 ห้ามเอาภาพรถจริงที่มีทะเบียนจริงมา commit ลง repo นี้
repo เป็น public และทะเบียนรถชี้ไปหาเจ้าของได้ผ่านฐานทะเบียนกรมขนส่ง
เทสต์ต้องการแค่ "ไฟล์ภาพที่ decode ได้" ซึ่งสี่เหลี่ยมสีเทาก็พอ
"""
import atexit
import base64
import glob
import io
import os
import sys
import tempfile
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTHONUTF8", "1")

# 🔴 ฐานข้อมูลของเทสต์ต้องเป็นไฟล์ทิ้ง **นอก repo** และต้องถูกเก็บกวาดเมื่อจบ
#
# วางไว้ใน tests/ รอบแรก แล้วมันค้างอยู่สามไฟล์หลังรันสองรอบ · ไฟล์ที่เทสต์
# ทิ้งไว้ในโฟลเดอร์ที่ track อยู่ คือของที่จะโดน commit ขึ้น GitHub วันหนึ่ง
# โดยไม่มีใครตั้งใจ และที่นี่มันคือฐานข้อมูลที่มีทะเบียนรถอยู่ข้างใน
_TEST_DB = os.path.join(tempfile.gettempdir(), f"lpr_test_{uuid.uuid4().hex}.db")
os.environ["LPR_STORE_DSN"] = _TEST_DB


@atexit.register
def _sweep():
    """WAL ทิ้งไฟล์คู่ไว้ด้วย (-wal, -shm) ลบให้ครบ ไม่ใช่ลบแค่ไฟล์หลัก"""
    for f in glob.glob(_TEST_DB + "*"):
        try:
            os.remove(f)
        except OSError:
            pass

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import lpr.main as lm  # noqa: E402
from lpr.llm.client import LPRResult, _split_plate, normalize  # noqa: E402
from lpr.th import FIELD as TH  # noqa: E402

TOP = {"request_id", "ref", "camera_id", "received_at", "status", "reason",
       "vehicles", "summary", "image", "model", "timing_ms"}
VEHICLE = {"ref", "status", "where", "plate", "province", "province_confidence",
           "vehicle_type", "vehicle_type_confidence", "vehicle_color",
           # ยี่ห้อ/รุ่น/รุ่นย่อย · Toy สั่ง 2026-09-16 · ความเชื่อมั่นคนละตัวต่อชั้น
           "vehicle_make", "vehicle_make_confidence",
           "vehicle_model", "vehicle_model_confidence",
           "vehicle_generation", "vehicle_generation_confidence",
           "description", "overall_confidence", "reason", "timing_ms"}
PLATE = {"text_raw", "text", "pattern", "prefix", "letters", "digits", "color",
         "confidence"}
SUMMARY = {"vehicles_found", "ok", "degraded", "plates_read", "plates_unread",
           "province_known", "province_unknown",
           "pattern_two_letter", "pattern_prefixed", "pattern_three_letter",
           "pattern_commercial", "pattern_unknown",
           "type_car", "type_pickup", "type_motorcycle", "type_truck",
           "type_van", "type_bus", "type_other", "type_unknown"}

client = TestClient(lm.app)


def _png(w=640, h=360, color=(120, 120, 130)) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _fake(monkeypatch, payload, finish="stop"):
    """ยัดคำตอบดิบของโมเดลเข้าไปตรงๆ · `payload` เป็น dict หรือ None (โมเดลล่ม)"""
    import json

    def fake_read(self, *a, **kw):
        if payload is None:
            return LPRResult(None, "", None, None, 12.0, "ConnectionError: dead")
        return LPRResult(payload, json.dumps(payload, ensure_ascii=False),
                         finish, 120, 12.0)

    monkeypatch.setattr(lm.llm.__class__, "read", fake_read)


def _post(monkeypatch, payload, **body):
    _fake(monkeypatch, payload)
    b = {"camera_id": "cam-lpr", "image_base64": _png(), **body}
    r = client.post("/v1/vehicle", json=b)
    assert r.status_code == 200, r.text
    return r.json()


def _flat(text: str) -> str:
    """ยุบช่องว่างทั้งหมดเหลือช่องเดียว แล้วเป็นตัวพิมพ์เล็ก

    prompt ถูกจัดย่อหน้าให้อ่านง่าย ประโยคหนึ่งจึงข้ามบรรทัดได้ · เทสต์ที่เทียบ
    สตริงดิบจะตกวันที่มีคนจัดย่อหน้าใหม่ ทั้งที่กฎยังอยู่ครบทุกตัว
    **ด่านที่ตกด้วยเหตุผลที่ไม่เกี่ยวกับสิ่งที่มันเฝ้า คือด่านที่คนจะปิดทิ้ง**
    """
    import re
    return re.sub(r"\s+", " ", text).strip().lower()


def _prompt(w: int, h: int):
    from lpr.llm import prompt_v1
    return prompt_v1.build(w, h)


ONE_CAR = {"vehicles": [{"ref": "V1", "where": "กลางภาพ", "plate_text": "1กข 1234",
                         "confidence": 0.85, "province": "สงขลา",
                         "province_confidence": 0.6, "plate_color": "white",
                         "vehicle_type": "pickup", "vehicle_type_confidence": 0.9,
                         "vehicle_color": "ขาว",
                         "description": "รถกระบะสีขาว", "reason": ""}]}


# ------------------------------------------------------------------ รูปคำตอบ
def test_รูปของ_response_ตรงกับที่ออกแบบไว้เป๊ะ(monkeypatch):
    d = _post(monkeypatch, ONE_CAR)
    assert set(d) == TOP
    assert set(d["vehicles"][0]) == VEHICLE
    assert set(d["vehicles"][0]["plate"]) == PLATE
    assert set(d["summary"]) == SUMMARY


def test_model_ถูกซ่อนเป็น_null_โดยค่าเริ่มต้น(monkeypatch):
    """ปลายทางไม่ต้องรู้ว่าเราใช้โมเดลอะไร · คีย์ยังอยู่ รูป response ไม่เปลี่ยน"""
    d = _post(monkeypatch, ONE_CAR)
    assert "model" in d and d["model"] is None


def test_ยอดรวมบวกได้ครบจำนวนคัน(monkeypatch):
    """🔴 กฎเดียวกับทุกชุดในโครงนี้ · มี unknown เสมอและบวกได้ครบ

    ผ่อนเพดานเป็นสองคันเฉพาะเทสต์นี้ · ของจริงตอบคันเดียว (Toy สั่ง 2026-09-16)
    แต่กฎ "ทุกชุดบวกได้ครบ" ต้องจริงที่ทุกจำนวน ไม่ใช่จริงเพราะมีคันเดียวพอดี
    """
    monkeypatch.setattr(lm, "MAX_VEHICLES", 2)
    payload = {"vehicles": [
        dict(ONE_CAR["vehicles"][0]),
        {"ref": "V2", "plate_text": "", "province": "", "vehicle_type": "unknown",
         "reason": "ป้ายเบลอ"},
    ]}
    s = _post(monkeypatch, payload)["summary"]
    n = s["vehicles_found"]
    assert n == 2
    assert s["plates_read"] + s["plates_unread"] == n
    assert s["province_known"] + s["province_unknown"] == n
    assert sum(v for k, v in s.items() if k.startswith("type_")) == n
    assert sum(v for k, v in s.items() if k.startswith("pattern_")) == n


# ------------------------------------------------------------------ แยกทะเบียน
@pytest.mark.parametrize("raw,want", [
    # มีเลขนำหน้า · กทม. ตั้งแต่ 2555 และจังหวัดที่หมวดอักษรเต็ม
    ("6กว 3869", ("6", "กว", "3869", "6กว 3869", "prefixed")),
    ("1กข 1234", ("1", "กข", "1234", "1กข 1234", "prefixed")),
    ("2ฒฬ-9999", ("2", "ฒฬ", "9999", "2ฒฬ 9999", "prefixed")),
    ("๑กก ๑๒๓๔", ("1", "กก", "1234", "1กก 1234", "prefixed")),  # เลขไทยบนป้ายจริงมี
    # สองอักษร · รถยนต์ส่วนบุคคลทุกจังหวัด รวม กทม. ก่อนปี 2555
    ("กท 5678", ("", "กท", "5678", "กท 5678", "two_letter")),
    ("  ขข  1234  ", ("", "ขข", "1234", "ขข 1234", "two_letter")),
    # สามอักษร · จักรยานยนต์รุ่นปัจจุบัน ตัวอย่างจากต้นฉบับคือ "กขค 123"
    ("กขค 123", ("", "กขค", "123", "กขค 123", "three_letter")),
    ("กขค 1234", ("", "กขค", "1234", "กขค 1234", "three_letter")),
    # รถบรรทุก/รถโดยสาร · ไม่มีตัวอักษรเลย และ **ต้องคงขีดไว้ในผลลัพธ์**
    ("10-0001", ("10", "", "0001", "10-0001", "commercial")),
    ("81-1234", ("81", "", "1234", "81-1234", "commercial")),
    ("700-1234", ("700", "", "1234", "700-1234", "commercial")),
])
def test_แยกทะเบียนตามรูปของป้ายไทย(raw, want):
    assert _split_plate(raw) == want


@pytest.mark.parametrize("raw", [
    "ABC 1234",        # อักษรละติน · ห้ามเดาว่าตรงกับอักษรไทยตัวไหน
    "1234",            # ไม่มีตัวอักษร
    "กขคง 1234",       # อักษรสี่ตัว ไม่ใช่รูปของป้ายไทย
    "1กขค 1234",       # เลขนำหน้าคู่กับอักษรสามตัว ไม่มีรูปนี้จริง
    "1234567890",      # เลขล้วนยาวเกินรูปรถบรรทุก
    "กข 12345",        # เลขเกินสี่หลัก
    "กี่ 123",          # มีสระ/วรรณยุกต์ ไม่ใช่พยัญชนะล้วน
    # 🔴 สามตัวล่างนี้คือเคสจริงจากภาพของ Toy 2026-09-16 · ด่านเดิมปล่อยผ่านหมด
    "6ก 3869",         # พยัญชนะตัวเดียว = โมเดลทำ "ว" หล่นไป ไม่ใช่ทะเบียนหายาก
    "6ก73869",         # อ่าน "ว" เป็น "7" · เลขไปโผล่กลางบล็อกพยัญชนะ
    "ขข 123",          # เลขท้ายสามหลัก = อ่านมาไม่ครบ
    "",
    None,
])
def test_ทะเบียนที่แยกไม่ออกต้องว่าง_ไม่ใช่เดา(raw):
    """🔴 ฟังก์ชันนี้ไม่ซ่อมทะเบียน · แยกได้หรือไม่ได้ เท่านั้น

    ทะเบียนที่ถูก "ซ่อม" จนดูสมบูรณ์แบบ ชี้ไปที่รถผิดคัน และปลายทาง
    **ตรวจไม่ได้เลยแม้ถือภาพอยู่ในมือ** เหตุผลเดียวกับที่ปิดฟิลด์สัญชาติฝั่งงานคน
    """
    assert _split_plate(raw) == ("", "", "", "", "unknown")


def test_อ่านไม่ออกเก็บของดิบไว้_และความมั่นใจเป็นศูนย์(monkeypatch):
    """ข้อความดิบคือหลักฐานเดียวที่บอกได้ว่าโมเดลเห็นอะไร ทิ้งแล้ว debug ไม่ได้

    และความมั่นใจต้องถูกกดเป็น 0 ไม่ว่าโมเดลจะบอกมาเท่าไร · ค่าความมั่นใจ
    ที่ลอยอยู่โดยไม่มีทะเบียนให้มั่นใจ คือตัวเลขที่ปลายทางเอาไปคัดกรองแล้วผิด
    """
    payload = {"vehicles": [{"ref": "V1", "plate_text": "ABC 1234",
                             "confidence": 0.95, "vehicle_type": "car"}]}
    v = _post(monkeypatch, payload)["vehicles"][0]
    assert v["plate"]["text"] == "" and v["plate"]["letters"] == ""
    assert v["plate"]["text_raw"] == "ABC 1234"
    assert v["plate"]["confidence"] == 0.0
    assert "ABC 1234" in v["reason"]


# ------------------------------------------------------------------ จังหวัด
def test_จังหวัดนอกชุด_77_ตัวต้องว่าง_ไม่ใช่ค่าใกล้เคียง():
    """🔴 ห้ามมี fuzzy match · "ตาก" กับ "ตราด" ต่างกันตัวเดียว

    จังหวัดที่ผิดคือรถคนละคันในฐานทะเบียน เหมือนทะเบียนที่ผิด
    """
    assert normalize({"province": "สงขลา"})["province"] == "สงขลา"
    assert normalize({"province": "จ.สงขลา"})["province"] == "สงขลา"
    assert normalize({"province": "กทม."})["province"] == "กรุงเทพมหานคร"
    assert normalize({"province": "อยุธยา"})["province"] == "พระนครศรีอยุธยา"
    for bad in ("สงคลา", "Songkhla ", "ไม่ทราบ", "", None, "เมืองสงขลา"):
        assert normalize({"province": bad})["province"] == "", bad


def test_จังหวัดอ่านไม่ได้_ความมั่นใจต้องเป็นศูนย์ด้วย():
    n = normalize({"province": "ดาวอังคาร", "province_confidence": 0.9})
    assert n["province"] == "" and n["province_confidence"] == 0.0


def test_มี_77_จังหวัดครบและไม่ซ้ำ():
    from lpr.schemas import PROVINCES
    assert len(PROVINCES) == 77 and len(set(PROVINCES)) == 77
    assert "กรุงเทพมหานคร" in PROVINCES and "บึงกาฬ" in PROVINCES


# ------------------------------------------------------------------ ชนิดรถ
def test_มองไม่เห็นตัวรถ_คือคำตอบที่ถูก_ไม่ใช่ความล้มเหลว(monkeypatch):
    """Toy เขียนเงื่อนไขไว้เอง: "ประเภทรถ ... กรณีเห็นตัวรถด้วย" (2026-09-14)

    ส่งครอปเฉพาะป้ายมา = มองไม่เห็นตัวรถ = unknown และ **ความมั่นใจเป็นศูนย์**
    ไม่ใช่ degraded และไม่ใช่เดาจากป้าย (ป้ายไม่ได้บอกว่ามันติดอยู่กับรถอะไร)
    """
    payload = {"vehicles": [{"ref": "V1", "plate_text": "กข 1234",
                             "confidence": 0.8, "vehicle_type": "unknown",
                             "vehicle_type_confidence": 0.7, "vehicle_color": ""}]}
    d = _post(monkeypatch, payload)
    v = d["vehicles"][0]
    assert d["status"] == "ok" and v["status"] == "ok"
    assert v["vehicle_type"] == TH["vehicle_type"]["unknown"]
    assert v["vehicle_type_confidence"] == 0.0
    assert v["plate"]["text"] == "กข 1234"      # ป้ายยังอ่านได้ตามปกติ


def test_ชนิดรถที่สะกดคนละแบบยังผ่าน():
    """`motorbike` `lorry` `sedan` คือคำตอบที่ถูก สะกดคนละแบบ

    ทิ้งคำตอบที่ถูกเพราะสะกดไม่ตรง คือบั๊ก `laptop` ฝั่งงานคน 2026-09-11
    """
    for said, want in (("motorbike", "motorcycle"), ("lorry", "truck"),
                       ("sedan", "car"), ("SUV", "car"), ("minivan", "van"),
                       ("pick_up", "pickup")):
        assert normalize({"vehicle_type": said})["vehicle_type"] == want, said
    assert normalize({"vehicle_type": "รถไถ"})["vehicle_type"] == "unknown"


# ------------------------------------------------------------------ ความล้มเหลว
def test_โมเดลล่ม_ต้องเป็น_degraded_ไม่ใช่_ok_ที่ศูนย์คัน(monkeypatch):
    """🔴 บั๊กเดียวกับที่ฝั่งช้างเสียเวลาไปทั้งวัน (ภาพ RGB ได้ 200 OK counts ว่าง)

    "ไม่มีป้ายในภาพ" กับ "อ่านภาพไม่ได้" หน้าตาเหมือนกันเป๊ะถ้ายุบเป็นอันเดียว
    แล้วปลายทางจะบันทึกว่า "ไม่มีรถผ่านประตู" ในนาทีที่ระบบเราล่ม
    """
    d = _post(monkeypatch, None)
    assert d["status"] == "degraded"
    assert d["vehicles"] == []
    assert d["reason"]


def test_โมเดลตอบไม่จบ_ห้ามรับมาใช้(monkeypatch):
    """ทะเบียนที่ถูกตัดท้ายคือทะเบียนของรถคันอื่น ไม่ใช่คำตอบที่หยาบลง"""
    _fake(monkeypatch, ONE_CAR, finish="length")
    d = client.post("/v1/vehicle", json={"camera_id": "c",
                                         "image_base64": _png()}).json()
    assert d["status"] == "degraded" and d["vehicles"] == []
    assert "length" in d["reason"]


def test_ไม่มีป้ายในภาพ_ไม่ใช่ความล้มเหลว(monkeypatch):
    d = _post(monkeypatch, {"vehicles": []})
    assert d["status"] == "ok" and d["summary"]["vehicles_found"] == 0
    assert d["reason"] == ""


def test_ไม่มีคีย์_vehicles_เลย_คือโมเดลไม่ทำตามสั่ง_ไม่ใช่ลานจอดว่าง(monkeypatch):
    d = _post(monkeypatch, {"cars": []})
    assert d["status"] == "degraded" and "vehicles" in d["reason"]


def test_ภาพเสียตอบ_400(monkeypatch):
    _fake(monkeypatch, ONE_CAR)
    # base64 ที่ decode ได้แต่ไม่ใช่ภาพ · ต้องยาวพอผ่าน min_length ของ schema
    # ไม่งั้นจะได้ 422 จากชั้น schema แทน แล้วเทสต์นี้ไม่ได้ตรวจสิ่งที่ตั้งใจจะตรวจ
    not_an_image = base64.b64encode(b"this is plain text, not an image at all").decode()
    r = client.post("/v1/vehicle", json={"camera_id": "c",
                                         "image_base64": not_an_image})
    assert r.status_code == 400, r.text


def test_เกินเพดานจำนวนคันถูกตัด_ไม่ใช่ตอบยาวไม่จบ(monkeypatch):
    many = {"vehicles": [{"ref": f"V{i}", "plate_text": f"กก {1000 + i}",
                          "confidence": 0.5} for i in range(20)]}
    d = _post(monkeypatch, many)
    assert d["summary"]["vehicles_found"] == lm.MAX_VEHICLES


def test_ตอบคันเดียวเสมอ_และเป็นคันที่ป้ายชัดที่สุด(monkeypatch):
    """🔴 Toy สั่ง 2026-09-16 · ภาพหนึ่งใบ = ถามถึงรถคันเดียว

    และคันที่เลือกต้องเป็นคันที่ **ป้ายชัดที่สุด ไม่ใช่คันแรกที่โมเดลพิมพ์ออกมา**
    ลำดับที่โมเดลพิมพ์คือลำดับที่มันนึกออก ไม่ใช่ลำดับความชัด
    """
    many = {"vehicles": [
        {"ref": "V1", "plate_text": "กก 1111", "confidence": 0.3},
        {"ref": "V2", "plate_text": "2ขข 2222", "confidence": 0.9},
        {"ref": "V3", "plate_text": "", "confidence": 0.0, "reason": "ป้ายเบลอ"},
    ]}
    d = _post(monkeypatch, many)
    assert len(d["vehicles"]) == 1
    v = d["vehicles"][0]
    assert v["ref"] == "V2" and v["plate"]["text"] == "2ขข 2222"
    # คันที่ถูกคัดออกต้องมองเห็น ไม่ใช่หายเงียบ
    assert "2 คัน" in d["reason"] and d["status"] == "ok"


def test_คันที่อ่านป้ายได้ชนะคันที่มั่นใจกว่าแต่อ่านไม่ออก(monkeypatch):
    """บริการนี้ตอบเรื่องทะเบียน ไม่ใช่เรื่องรถ · ความมั่นใจของป้ายที่อ่านไม่ออก
    ถูกกดเป็น 0 อยู่แล้ว ตรงนี้คือด่านที่ยืนยันว่าลำดับการคัดไม่กลับหัว
    """
    payload = {"vehicles": [
        {"ref": "V1", "plate_text": "ฟฟ 9", "confidence": 0.99},   # ผิดรูป
        {"ref": "V2", "plate_text": "กก 4321", "confidence": 0.4},
    ]}
    d = _post(monkeypatch, payload)
    assert [v["ref"] for v in d["vehicles"]] == ["V2"]


@pytest.mark.parametrize("raw,ต้องมีในเหตุผล", [
    ("6ก 3869", "พยัญชนะ"),      # อักษรหล่นไปหนึ่งตัว = ไปแก้ที่ prompt
    ("ขข 123", "เลขท้าย"),        # เลขไม่ครบ = ไปแก้ที่มุมกล้อง/ระยะ
    ("ABC 1234", "พยัญชนะไทย"),   # อักษรละติน
])
def test_ผิดรูปแล้วต้องบอกว่าผิดตรงไหน_ไม่ใช่แค่ว่าผิด(monkeypatch, raw, ต้องมีในเหตุผล):
    """🔴 "อ่านไม่ออก" เฉยๆ ตอบคำถามที่ต้องตอบไม่ได้: ถ่ายใหม่ หรือแก้ prompt

    อักษรตัวเดียวคู่กับเลขครบสี่ = โมเดลทำอักษรหล่น (เคสจริงของ Toy 2026-09-16)
    เลขสามหลัก = ป้ายโดนบัง/มุมกล้อง · สองอย่างนี้แก้คนละที่
    """
    d = _post(monkeypatch, {"vehicles": [{"ref": "V1", "plate_text": raw,
                                          "confidence": 0.9}]})
    v = d["vehicles"][0]
    assert v["plate"]["text"] == "" and v["plate"]["pattern"] == TH["plate_pattern"]["unknown"]
    assert v["plate"]["confidence"] == 0.0
    assert ต้องมีในเหตุผล in v["reason"]
    assert raw in v["plate"]["text_raw"]       # ของดิบยังอยู่ เป็นหลักฐาน


def test_prompt_เขียนรูปทะเบียนครบทุกแบบและคู่ที่สับสนข้ามชนิด():
    """🔴 v3 · ด่านนี้เฝ้าสองอาการที่เจอจริงในวันเดียวกัน (2026-09-16)

    1. ว/7 เป็นคู่สับสน **ข้ามชนิด** (พยัญชนะกับตัวเลข) ซึ่ง v1 ไม่ได้เตือนเลย
       มันเตือนแต่คู่พยัญชนะด้วยกัน → `6กว` กลายเป็น `6ก7`
    2. v2 สั่งว่า "พยัญชนะสองตัวเสมอ" ซึ่งเป็นคำสั่งที่ทำให้โมเดล**ดัด**ป้ายรถบรรทุก
       (`10-0001` ไม่มีตัวอักษรเลย) ให้เข้ารูปที่เราบอก · กฎที่เขียนกันพลาด
       กลายเป็นตัวสั่งให้พลาดเอง ซึ่งแย่กว่าอ่านไม่ออก
    """
    from lpr.llm import prompt_v1
    system, user = _prompt(640, 360)
    flat = _flat(system + user)
    assert prompt_v1.PROMPT_VERSION == "v4"
    assert "two thai consonants, then up to four digits" in flat
    assert "one digit, two consonants, then up to four digits" in flat
    assert "three thai consonants, then up to four digits" in flat
    assert "two or three digits, a hyphen, then four digits" in flat
    assert "ว and 7" in flat
    # ห้ามสั่งให้เติมตัวอักษรให้ป้ายรถบรรทุก
    assert "no thai letters at all and that is correct" in flat
    # ขอคันเดียว ไม่ใช่ทุกคันในภาพ
    assert "report one vehicle" in flat


def test_ป้ายรถบรรทุกนับเป็นอ่านได้_ไม่ใช่อ่านไม่ออก(monkeypatch):
    """🔴 รูป `10-0001` ไม่มีตัวอักษรสักตัวและถูกต้องตามป้ายจริง

    ตัวชี้ว่าอ่านได้คือ `text` ไม่ใช่ `letters` · เคยใช้ `letters` เป็นตัวชี้
    ซึ่งจะทำให้รถบรรทุกทั้งไซต์ขึ้นเป็น "อ่านไม่ได้" ทั้งที่อ่านถูกทุกคัน
    """
    d = _post(monkeypatch, {"vehicles": [{"ref": "V1", "plate_text": "10-0001",
                                          "confidence": 0.8,
                                          "vehicle_type": "truck"}]})
    v = d["vehicles"][0]
    assert v["plate"]["text"] == "10-0001" and v["plate"]["letters"] == ""
    assert v["plate"]["confidence"] == 0.8
    assert d["summary"]["plates_read"] == 1 and d["summary"]["plates_unread"] == 0
    assert d["summary"]["pattern_commercial"] == 1


def test_เลขท้ายสั้นถูกตีตกโดยค่าเริ่มต้น_และเปิดรับได้ด้วย_env(monkeypatch):
    """🔴 จุดที่เอกสารกับหน้างานไม่ตรงกัน และตั้งใจเลือกข้างไว้

    กรมขนส่งออก `กข 1` จริง (ต้นฉบับ: "สูงสุด 4 หลัก ตั้งแต่ 1 ถึง 9999")
    แต่ Toy ยืนยันว่าไซต์นี้สี่ตัวเสมอ · บังคับสี่หลักไว้เพราะเป็นด่านเดียวที่จับ
    "อ่านเลขขาด" ได้ · **เปิดสวิตช์แล้วเสียด่านนั้นไป** เทสต์นี้จึงตรึงทั้งสองฝั่ง
    ไว้ด้วยกัน ใครเปลี่ยนค่าเริ่มต้นจะเห็นทันทีว่ากำลังแลกอะไร
    """
    import importlib

    from lpr.llm import client as c
    assert c.ALLOW_SHORT_DIGITS is False
    assert c._split_plate("กข 1") == ("", "", "", "", "unknown")

    monkeypatch.setenv("LPR_ALLOW_SHORT_DIGITS", "true")
    c2 = importlib.reload(c)
    try:
        assert c2.ALLOW_SHORT_DIGITS is True
        assert c2._split_plate("กข 1") == ("", "กข", "1", "กข 1", "two_letter")
        # ราคาของการเปิด: เลขที่อ่านขาดกลายเป็นทะเบียนที่ถูกต้องทันที
        assert c2._split_plate("6กว 3")[4] == "prefixed"
    finally:
        monkeypatch.delenv("LPR_ALLOW_SHORT_DIGITS")
        importlib.reload(c)


# ------------------------------------------------------------------ bbox
def test_ส่ง_bbox_มาแล้วเราครอปเอง_และบอกกรอบที่ใช้จริง(monkeypatch):
    """กรอบที่คืนไป **ไม่ใช่ค่าเดียวกับที่ส่งมา** เพราะเผื่อขอบแล้วหนีบขอบภาพแล้ว
    ปลายทางเอาไปยืนยันด้วยตาได้ว่าเราอ่านป้ายของคันที่เขาชี้จริง
    """
    d = _post(monkeypatch, ONE_CAR, bbox=[100, 100, 200, 80])
    assert d["image"]["source"] == "bbox_crop"
    assert d["image"]["bbox_used"] != [100, 100, 200, 80]
    assert len(d["image"]["bbox_used"]) == 4


def test_bbox_ล้นขอบภาพไม่ทำให้พัง(monkeypatch):
    """ALPR ปลายทางคืนกรอบล้นขอบเป็นเรื่องปกติเวลารถกำลังออกจากเฟรม"""
    d = _post(monkeypatch, ONE_CAR, bbox=[600, 330, 400, 200])
    assert d["image"]["source"] == "bbox_crop"


def test_bbox_ผิดรูปตอบ_400(monkeypatch):
    _fake(monkeypatch, ONE_CAR)
    r = client.post("/v1/vehicle", json={"camera_id": "c", "image_base64": _png(),
                                         "bbox": [1, 2, 3]})
    assert r.status_code == 400


def test_ไม่ส่ง_bbox_ก็ได้_และ_bbox_used_เป็น_null(monkeypatch):
    d = _post(monkeypatch, ONE_CAR)
    assert d["image"]["source"] == "full_image"
    assert d["image"]["bbox_used"] is None


# ------------------------------------------------------------------ ย้อนหลัง
def test_ตามผลย้อนหลังด้วย_request_id_และด้วยเลขอ้างอิงของปลายทาง(monkeypatch):
    d = _post(monkeypatch, ONE_CAR, client_request_id="gate-0001")
    assert d["ref"] == "gate-0001"
    back = client.get(f"/v1/vehicle/{d['request_id']}").json()
    assert back["request"]["request_id"] == d["request_id"]
    assert back["vehicles"][0]["plate_text"] == "1กข 1234"

    by = client.get("/v1/vehicle/by-ref/gate-0001").json()
    assert by["found"] >= 1
    assert by["request"]["request_id"] == d["request_id"]


def test_เลขอ้างอิงมาใน_note_ก็ได้_เหมือนงานอื่นใน_repo(monkeypatch):
    d = _post(monkeypatch, ONE_CAR, note="ref-777")
    assert d["ref"] == "ref-777"


def test_ref_ต้องคืนตัวเดิมเป๊ะ_ห้ามแตะ(monkeypatch):
    """🔴 ของที่ปลายทางเอาไปเทียบว่า "ใช่ของฉันไหม" ห้ามถูกเราแตะระหว่างทาง

    เวอร์ชันแรกของ `_ref_of` ที่นี่ `.strip()` แล้วตัดที่ 100 ตัว ซึ่งดูไม่มีพิษภัย
    จนกระทั่งปลายทางส่ง note ยาวกว่านั้นมา แล้วเทียบกับของตัวเองไม่ตรง
    **โดยไม่มีอะไรบอกเขาเลยว่าเราเป็นคนตัด** · งานคนมีด่านนี้อยู่ก่อนแล้ว
    """
    for sent in ("  GATE-A/รถตู้-001  ", "ก" * 400, "ref#77 (ประตู 2)", "0012"):
        d = _post(monkeypatch, ONE_CAR, note=sent)
        assert d["ref"] == sent, f"ถูกแตะระหว่างทาง: {sent!r} -> {d['ref']!r}"


def test_ตัวอักษรไทยและอักขระพิเศษใน_note_ใช้ได้(monkeypatch):
    """ปลายทางตั้งเลขอ้างอิงเป็นอะไรก็ได้ มันเป็นของเขา ไม่ใช่ของเรา"""
    d = _post(monkeypatch, ONE_CAR, note="ประตู2-กล้อง3-ครั้งที่7")
    assert d["ref"] == "ประตู2-กล้อง3-ครั้งที่7"


def test_ground_truth_บันทึกได้_และวัดความแม่นได้จริง(monkeypatch):
    """🔴 งานนี้วัดความแม่นได้ ต่างจากงานอารมณ์/สัญชาติของฝั่งคน"""
    d = _post(monkeypatch, ONE_CAR)
    rid = d["request_id"]
    r = client.post(f"/v1/vehicle/{rid}/V1/truth",
                    json={"plate": "1กข 1234", "province": "สงขลา",
                          "reviewer": "toy"})
    assert r.status_code == 200
    acc = client.get("/v1/accuracy").json()
    assert acc["checked"] >= 1 and acc["exact"] >= 1

    assert client.post(f"/v1/vehicle/{rid}/V9/truth",
                       json={"plate": "x"}).status_code == 404


# ------------------------------------------------------------------ ค่าเป็นไทย
def test_ค่าเป็นไทย_แต่ทะเบียนกับจังหวัดห้ามถูกแตะ(monkeypatch):
    """🔴 ทะเบียนคือหลักฐาน ไม่ใช่ค่าที่เราตีความ · แปลแล้วไม่ใช่หลักฐานอีกต่อไป"""
    d = _post(monkeypatch, ONE_CAR)
    v = d["vehicles"][0]
    assert v["vehicle_type"] == TH["vehicle_type"]["pickup"]      # แปล
    assert v["plate"]["color"] == TH["plate_color"]["white"]      # แปล
    assert v["plate"]["text"] == "1กข 1234"                       # ห้ามแตะ
    assert v["plate"]["letters"] == "กข" and v["plate"]["digits"] == "1234"
    assert v["province"] == "สงขลา"
    assert v["status"] == "ok" and v["ref"] == "V1"               # ค่าโปรโตคอล


def test_ทุกค่าใน_schema_ต้องมีคำแปลไทย():
    """ไล่จาก Literal ใน schemas.py เป็นตัวตั้ง ไม่ใช่จากรายการที่พิมพ์ไว้เอง

    เพิ่มค่าใหม่ใน schema แล้วลืมเติมคำแปล = หลุดเป็นอังกฤษปนไทยไปให้ปลายทางเจอเอง
    ซึ่งไม่มี error ให้เห็นเลย
    """
    import typing

    from lpr import th
    from lpr.schemas import PlateColor, PlatePattern, VehicleType
    for path, ann in (("vehicle_type", VehicleType), ("plate_color", PlateColor),
                      ("plate_pattern", PlatePattern)):
        want = set(typing.get_args(ann))
        table = th.FIELD[path]
        missing = want - set(table)
        assert not missing, f"{path} ไม่มีคำแปลของ {sorted(missing)}"
        for k in want:
            assert table[k] != k, f"{path}.{k} ยังเป็นอังกฤษอยู่"


def test_แปลแล้วไม่แก้ของเดิม():
    """ก้อนเดียวกันถูกเอาไปเขียนลงฐานด้วย แก้ในที่ = ฐานกลายเป็นไทยโดยไม่ตั้งใจ"""
    from lpr.th import thai
    src = {"vehicle_type": "car", "plate": {"color": "white", "text": "กก 1"}}
    out = thai(src)
    assert src == {"vehicle_type": "car", "plate": {"color": "white", "text": "กก 1"}}
    assert out["vehicle_type"] == "รถเก๋ง"
    assert out["plate"]["color"] == "ขาว" and out["plate"]["text"] == "กก 1"


# ------------------------------------------------------------------ ด่านโครงสร้าง
def test_ชุดค่าในโค้ดกับใน_prompt_ต้องตรงกันทุกตัว():
    """🔴 บทเรียน `laptop` · ชุดค่าที่ต้องตรงกันแต่แก้คนละที่ จะไม่ตรงกัน

    ที่นี่ prompt ประกอบชุดค่าจาก schema โดยตรงอยู่แล้ว เทสต์นี้จึงเป็นด่านกัน
    วันที่มีคนเผลอพิมพ์รายการลง prompt เองเพื่อ "ให้อ่านง่ายขึ้น"
    """
    import typing

    from lpr.llm.client import _PLATE_COLOR, _VEHICLE_TYPE
    from lpr.llm.prompt_v1 import allowed_block
    from lpr.schemas import PlateColor, VehicleType

    allowed = allowed_block()
    for name, ann, in_code in (("vehicle_type", VehicleType, _VEHICLE_TYPE),
                               ("plate_color", PlateColor, _PLATE_COLOR)):
        want = set(typing.get_args(ann))
        line = [l for l in allowed.splitlines() if l.startswith(name + ":")]
        assert line, f"ไม่มีบรรทัด {name} ใน prompt เลย"
        got = {v.strip() for v in line[0].split(":", 1)[1].split(",")}
        assert want == got, f"{name} ใน prompt ไม่ตรงกับ schema: {want ^ got}"
        assert want == in_code, f"{name} ใน normalize ไม่ตรงกับ schema: {want ^ in_code}"


def test_prompt_ไม่ขอ_bbox():
    """🔴 วัดมาแล้วฝั่งช้าง โมเดลคืน y=998 บนภาพสูง 628 · ตำแหน่งขอเป็นข้อความเท่านั้น

    ⚠️ เดิมเทสต์นี้กันเรื่องยี่ห้อ/รุ่นด้วย · Toy สั่งให้ถามแล้ว 2026-09-16
    เหตุผลเดิม (มันเดา ไม่ได้อ่าน) ไม่ได้หายไป ย้ายไปอยู่ที่
    `test_ยี่ห้อรุ่นต้องว่างเมื่อมองไม่เห็นตัวรถ` ซึ่งบังคับในโค้ด ไม่ใช่ในคำสั่ง
    """
    # 🔴 เทียบแบบยุบช่องว่าง · ประโยคใน prompt ขึ้นบรรทัดใหม่กลางประโยคได้เสมอ
    # เทสต์ที่เทียบสตริงดิบจะตกวันที่มีคนจัดย่อหน้าใหม่ ทั้งที่กฎยังอยู่ครบ
    system, user = _prompt(640, 360)
    assert "bbox" not in _flat(system)
    assert "where it sits in the image" in _flat(system)
    assert "do not give pixel coordinates" in _flat(user)


def test_ยี่ห้อรุ่นต้องว่างเมื่อมองไม่เห็นตัวรถ_และรุ่นต้องมียี่ห้อ(monkeypatch):
    """🔴 Toy สั่งเพิ่มยี่ห้อ/รุ่น 2026-09-16 · บังคับในโค้ด ไม่ฝากไว้กับ prompt

    ส่งครอปป้ายมาแล้วได้ยี่ห้อกลับไป = มันเดาจากทะเบียน ซึ่งทะเบียนบอกไม่ได้เลย
    ส่วนรุ่นที่ลอยอยู่โดยไม่รู้ยี่ห้อ คือสัญญาณว่ามันเดาทั้งพวง
    """
    # มองไม่เห็นตัวรถ (ครอปป้าย) แต่โมเดลตอบยี่ห้อมา -> ต้องถูกล้างทั้งสามช่อง
    d = _post(monkeypatch, {"vehicles": [{
        "ref": "V1", "plate_text": "6กว 3869", "confidence": 0.9,
        "vehicle_type": "unknown", "vehicle_make": "Toyota",
        "vehicle_make_confidence": 0.9, "vehicle_model": "Celica",
        "vehicle_model_confidence": 0.8, "vehicle_generation": "ST182",
        "vehicle_generation_confidence": 0.7}]})
    v = d["vehicles"][0]
    assert v["vehicle_make"] == "" and v["vehicle_model"] == ""
    assert v["vehicle_generation"] == ""
    assert v["vehicle_make_confidence"] == 0.0
    assert v["vehicle_generation_confidence"] == 0.0
    # ทะเบียนยังอ่านได้ตามปกติ · ด่านนี้ไม่แตะงานหลัก
    assert v["plate"]["text"] == "6กว 3869"

    # เห็นตัวรถ · รุ่นที่ไม่มียี่ห้อต้องถูกทิ้ง ส่วนยี่ห้อล้วนเป็นคำตอบที่ปกติ
    d = _post(monkeypatch, {"vehicles": [{
        "ref": "V1", "plate_text": "6กว 3869", "confidence": 0.9,
        "vehicle_type": "car", "vehicle_make": "", "vehicle_model": "Celica",
        "vehicle_model_confidence": 0.8}]})
    v = d["vehicles"][0]
    assert v["vehicle_model"] == "" and v["vehicle_model_confidence"] == 0.0

    # คำว่า "ไม่รู้" ไม่ใช่ยี่ห้อ · ช่องอิสระไม่มี _pick คอยกันให้
    for said in ("unknown", "N/A", "ไม่ทราบ", "-"):
        d = _post(monkeypatch, {"vehicles": [{
            "ref": "V1", "plate_text": "6กว 3869", "confidence": 0.9,
            "vehicle_type": "car", "vehicle_make": said,
            "vehicle_make_confidence": 0.9}]})
        assert d["vehicles"][0]["vehicle_make"] == "", said

    # เห็นตัวรถ + ยี่ห้อจริง -> เก็บครบทั้งสามชั้น พร้อมความเชื่อมั่นคนละตัว
    d = _post(monkeypatch, {"vehicles": [{
        "ref": "V1", "plate_text": "6กว 3869", "confidence": 0.9,
        "vehicle_type": "car", "vehicle_make": "Toyota",
        "vehicle_make_confidence": 0.85, "vehicle_model": "Celica",
        "vehicle_model_confidence": 0.5, "vehicle_generation": "ST182",
        "vehicle_generation_confidence": 0.2}]})
    v = d["vehicles"][0]
    assert (v["vehicle_make"], v["vehicle_model"], v["vehicle_generation"])         == ("Toyota", "Celica", "ST182")
    assert v["vehicle_make_confidence"] == 0.85
    assert v["vehicle_model_confidence"] == 0.5
    assert v["vehicle_generation_confidence"] == 0.2
    # 🔴 ยี่ห้อ/รุ่นเป็นชื่อเฉพาะ **ห้ามถูกแปลเป็นไทย** ตอนขาออก
    assert "Toyota" in d["vehicles"][0]["vehicle_make"]


def test_prompt_สั่งให้ตอบว่าอ่านไม่ออก_ไม่ใช่เดาให้ครบ():
    """กฎที่สำคัญที่สุดของงานนี้ · หายเมื่อไหร่ทะเบียนผิดจะไหลออกไปโดยไม่มีใครรู้"""
    system, _ = _prompt(640, 360)
    assert "leave that plate unread" in _flat(system)
    assert "points at a different vehicle entirely" in _flat(system)
    assert "that is an unread plate, not a coin toss" in _flat(system)


def test_lpr_ไม่_import_ข้ามไปแพ็กเกจงานอื่น():
    """🔴 สามงานอยู่ repo เดียวกัน แต่แยกสัญญากันเด็ดขาด

    import ข้ามเมื่อไหร่ = วันหนึ่งแก้ฝั่งหนึ่งแล้วอีกฝั่งพังโดยไม่มีใครตั้งใจ
    และสัญญาของฝั่งช้างถูก confirm กับทีมปลายทางไปแล้ว ห้ามขยับ
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent / "lpr"
    for f in root.rglob("*.py"):
        text = f.read_text(encoding="utf-8")
        for bad in ("from people", "import people", "from app", "import app."):
            assert bad not in text, f"{f.name} import ข้ามแพ็กเกจ: {bad}"


def test_คำไทยในหน้า_verify_lpr_ต้องตรงกับ_lpr_th():
    """หน้า verify คีย์ตารางสีด้วย **คำไทย** ซึ่งมาจาก lpr/th.py

    แก้คำแปลฝั่ง Python แล้วลืมแก้หน้าเว็บ = ป้ายกลายเป็นสีเทาหมด
    **โดยไม่มี error ให้เห็นเลย** หน้ายังโหลดปกติทุกประการ
    (อาการเดียวกับที่หน้า /people/verify เคยเจอ)
    """
    import pathlib
    import re

    from lpr import th
    html = (pathlib.Path(__file__).resolve().parent.parent
            / "lpr" / "ui" / "index.html").read_text(encoding="utf-8")
    m = re.search(r"const PC = \{(.*?)\};", html, re.S)
    assert m, "หาตาราง PC ในหน้าเว็บไม่เจอ"
    keys = set(re.findall(r"'([^']*[ก-๙][^']*)'\s*:", m.group(1)))
    assert keys == set(th.FIELD["plate_color"].values())


def test_healthz_ตอบได้และบอกสิ่งที่ปลายทางต้องรู้():
    h = client.get("/healthz").json()
    assert h["service"] == "thai-lpr"
    assert h["version"] == lm.LPR_VERSION
    # 🔴 ไม่ถูกซ่อนตาม EXPOSE_MODEL เพราะมันคือ "คำตอบที่คุณได้รับแปลว่าอะไร"
    assert "province_set" in h and "plate_split" in h
    assert h["model"] is None and h["prompt_version"] is None


def test_ยิง_body_ของงานคนมาที่เส้นนี้_ต้องบอกว่าไปผิดเส้น():
    r = client.post("/v1/vehicle", json={"camera_id": "c",
                                         "objects": [{"id": "1"}]})
    assert r.status_code == 422
    assert "people" in r.json().get("hint", "")
