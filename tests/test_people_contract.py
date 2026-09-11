"""🔴 รูปของ JSON ฝั่งคน · เทสต์นี้มีไว้ให้ "เหตุผลที่ฟังขึ้น" ดังตอน CI

เขียนไว้ตั้งแต่วันแรกทั้งที่ยังไม่ได้ confirm กับทีมคุณสุชาติ เพราะฝั่งช้าง
สอนมาแล้วว่าตอนกำลังแก้ท่ออยู่ การเพิ่มฟิลด์ใหม่มัน "สมเหตุสมผลมาก" เสมอ
(count_source กับ image_type เกือบหลุดขึ้น prod มาแล้ว)
ตอนนี้ฟิลด์ยังขยับได้ ขยับแล้วมาแก้ไฟล์นี้ให้ตรง · วันที่ confirm กับปลายทางแล้ว
ไฟล์นี้จะกลายเป็นด่านจริง แล้วห้ามแก้อีกเลยนอกจากปลายทางตกลงด้วย

รันได้โดยไม่ต้องมี LLM ไม่ต้องมี GPU ไม่ต้องมีเน็ต · ชี้ base_url ไปพอร์ตที่
ต่อไม่ติดแน่นอน แล้วทุกคนจะเป็น degraded ซึ่งคือเคสที่ต้องทดสอบอยู่แล้ว:
**ตอบเต็มรูปแม้โมเดลใช้ไม่ได้ ไม่ใช่ตอบสั้นลง**
"""
import base64
import io
import os

os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ["PEOPLE_STORE_DSN"] = "./data/people-contract-test.db"
os.environ["API_KEY"] = ""
os.environ["PEOPLE_API_KEY"] = ""
os.environ["LLM_BASE_URL"] = "http://127.0.0.1:9"   # ต่อไม่ติดแน่นอน
os.environ["LLM_TIMEOUT_S"] = "1"
os.environ["PEOPLE_LLM_MAX_RETRIES"] = "0"  # เทสต์ไม่ต้องรอ retry ของที่ต่อไม่ติดอยู่แล้ว

from fastapi.testclient import TestClient  # noqa: E402

import people.main as pm  # noqa: E402

TOP = {"request_id", "camera_id", "received_at", "status", "persons", "summary",
       "model", "timing_ms"}
PERSON = {"id", "status", "gender", "gender_confidence", "age_range",
          "age_range_confidence", "appearance", "appearance_confidence",
          "description", "overall_confidence", "reason", "image", "model",
          "timing_ms"}
APPEARANCE = {"skin_tone", "build", "height", "hair_length", "hair_color",
              "glasses", "face_mask", "facial_hair", "headwear", "top",
              "top_sleeve", "outer", "bottom", "footwear", "carrying", "distinctive"}
GARMENT = {"type", "color", "secondary_color", "pattern"}
MODEL = {"provider", "name", "prompt_version", "finish_reason", "completion_tokens"}
TIMING = {"decode", "vlm", "vlm_sum", "total"}
SUMMARY = {"objects_in", "ok", "degraded", "error"}

client = TestClient(pm.app)


def _png(w: int = 120, h: int = 260, color=(90, 90, 110)) -> str:
    """ภาพสังเคราะห์ · **ห้ามเอาภาพคนจริงมา commit ลง repo**

    repo นี้เป็น public และภาพตัวอย่างที่ทีมคุณสุชาติส่งมาคือหน้าคนจริง
    ที่เดินเข้าอาคารจริง ไม่มีใครในภาพนั้นเคยตกลงให้เอาไปไว้บน GitHub
    เทสต์ต้องการแค่ "ไฟล์ภาพที่ decode ได้" ซึ่งสี่เหลี่ยมสีเทาก็พอ
    """
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _post(objects, frame=None):
    body = {"camera_id": "cam-contract", "objects": objects}
    if frame:
        body["frame_base64"] = frame
    r = client.post("/v1/persons", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_รูปของ_response_ตรงกับที่ออกแบบไว้เป๊ะ():
    d = _post([{"id": "ID-1", "image_base64": _png()}])
    assert set(d) == TOP
    assert set(d["summary"]) == SUMMARY
    assert set(d["model"]) == MODEL
    assert set(d["timing_ms"]) == TIMING
    p = d["persons"][0]
    assert set(p) == PERSON
    assert set(p["appearance"]) == APPEARANCE
    for part in ("top", "outer", "bottom", "footwear"):
        assert set(p["appearance"][part]) == GARMENT
    assert set(p["image"]) == {"w", "h", "source", "bbox_used"}


def test_ชนิดของค่าถูกต้อง():
    """ปลายทางเขียนโค้ดตามชนิดพวกนี้ เปลี่ยนชนิดพังพอๆ กับเปลี่ยนชื่อ"""
    d = _post([{"id": "ID-1", "image_base64": _png()}])
    assert isinstance(d["persons"], list)
    assert isinstance(d["summary"]["objects_in"], int)
    p = d["persons"][0]
    assert isinstance(p["id"], str)
    assert isinstance(p["gender"], str)
    assert isinstance(p["gender_confidence"], float)
    assert isinstance(p["age_range"], str)
    assert isinstance(p["description"], str)
    assert isinstance(p["appearance"]["carrying"], list)
    assert isinstance(p["timing_ms"], float)


def test_id_ที่ส่งไปต้องกลับมาเหมือนเดิมและเรียงเดิม():
    """ปลายทางผูกคำตอบกลับด้วย id นี้ตัวเดียว · เพี้ยนเมื่อไหร่คือสลับตัวคน"""
    ids = ["011", "021", "ID:022"]
    d = _post([{"id": i, "image_base64": _png()} for i in ids])
    assert [p["id"] for p in d["persons"]] == ids


def test_llm_ล่ม_ก็ยังตอบเต็มรูป_ไม่ใช่ตอบสั้นลง():
    """โมเดลต่อไม่ติดใน conftest นี้อยู่แล้ว ทุกคนจึงเป็น degraded

    ห้ามตอบเป็น gender ว่างหรือ appearance หาย · ปลายทางที่ validate เข้ม
    จะปฏิเสธทั้งก้อน แล้วเราจะไม่รู้เลยว่าเพราะโมเดลล่ม
    """
    d = _post([{"id": "ID-1", "image_base64": _png()}])
    p = d["persons"][0]
    assert p["status"] == "degraded"
    assert d["status"] == "degraded"
    assert set(p) == PERSON
    assert p["gender"] == "unknown"
    assert p["age_range"] == "unknown"
    assert p["reason"], "degraded ต้องบอกเหตุผล ไม่ใช่เงียบ"


def test_unknown_กับ_degraded_ต้องแยกกันออก():
    """บทเรียนตรงๆ จากฝั่งช้าง ที่ degraded เคยโดน provisional ทับ

    unknown = ดูแล้วแต่ดูไม่ออก (จบแล้ว) · degraded = ยังไม่ได้ดู (retry ได้)
    ถ้าสองอย่างนี้หน้าตาเหมือนกัน ปลายทางจะ retry เคสที่ retry ไปก็เท่าเดิม
    """
    d = _post([{"id": "ID-1", "image_base64": _png()}])
    p = d["persons"][0]
    # โมเดลล่ม ไม่ใช่ "ดูไม่ออก" · status ต้องบอกเรื่องนี้ ไม่ใช่ให้ไปเดาจาก gender
    assert p["status"] != "ok"
    assert p["overall_confidence"] == 0.0


def test_ส่ง_bbox_มาแล้วเราครอปเอง():
    """เคส iplus_sample2: เฟรมเต็ม + กรอบหลายกรอบ"""
    d = _post([{"id": "a", "bbox": [10, 10, 60, 120]},
               {"id": "b", "bbox": [100, 20, 50, 150]}],
              frame=_png(640, 480))
    assert [p["id"] for p in d["persons"]] == ["a", "b"]
    for p in d["persons"]:
        assert p["image"]["source"] == "frame_bbox"
        # ครอปแล้วต้องถูกขยายให้ถึงขั้นต่ำ ไม่ใช่ส่งภาพจิ๋วไปให้ provider ปฏิเสธ
        assert min(p["image"]["w"], p["image"]["h"]) >= pm.MIN_CROP_PX
        # 🔴 ต้องคืนกรอบที่ครอปจริงกลับไป ปลายทางจะได้ยืนยันว่าอ่านถูกคน
        x1, y1, x2, y2 = p["image"]["bbox_used"]
        assert x2 > x1 and y2 > y1


def test_bbox_ล้นขอบภาพไม่ทำให้พัง():
    """tracker คืนกรอบล้นขอบเป็นเรื่องปกติตอนคนเดินออกเฟรม"""
    d = _post([{"id": "edge", "bbox": [600, 440, 200, 200]}], frame=_png(640, 480))
    im = d["persons"][0]["image"]
    assert im["source"] == "frame_bbox"
    # กรอบที่คืนกลับต้องถูกหนีบให้อยู่ในภาพจริง ไม่ใช่สะท้อนค่าที่ส่งมาเฉยๆ
    assert im["bbox_used"][2] <= 640 and im["bbox_used"][3] <= 480


def test_ส่ง_crop_มาเอง_ไม่มี_bbox_used():
    """null แปลว่า 'เราไม่ได้ครอป คุณเลือกภาพมาเอง' ไม่ใช่ 'ครอปไม่สำเร็จ'"""
    d = _post([{"id": "c", "image_base64": _png()}])
    assert d["persons"][0]["image"]["bbox_used"] is None


def test_object_ที่ไม่มีทั้งภาพและ_bbox_เสียแค่ตัวมันเอง():
    """ส่งมา 2 คน ผิดรูป 1 คน ต้องยังได้คำตอบของอีกคน ไม่ใช่ 400 ทั้งก้อน"""
    d = _post([{"id": "good", "image_base64": _png()}, {"id": "bad"}])
    by = {p["id"]: p for p in d["persons"]}
    assert by["bad"]["status"] == "error"
    assert by["bad"]["reason"]
    assert by["good"]["status"] == "degraded"   # โมเดลต่อไม่ติด แต่ถึงชั้นโมเดลแล้ว
    assert d["status"] == "degraded"            # ไม่มีใคร ok เลยในเทสต์นี้


def test_ทุก_object_ผิดรูป_ตอบ_400_ไม่ใช่_200_ที่ว่างเปล่า():
    r = client.post("/v1/persons", json={"camera_id": "c", "objects": [{"id": "x"}]})
    assert r.status_code == 400
    assert "x" in r.json()["detail"]


def test_id_ซ้ำใน_request_เดียวโดนปฏิเสธ():
    """ปลายทางจับคู่คำตอบกลับไม่ได้ ตอบไปก็ไม่มีประโยชน์"""
    r = client.post("/v1/persons", json={
        "camera_id": "c",
        "objects": [{"id": "dup", "image_base64": _png()},
                    {"id": "dup", "image_base64": _png()}]})
    assert r.status_code == 400


def test_เกิน_16_คนต่อ_request_โดนปฏิเสธตั้งแต่ชั้น_schema():
    """หนึ่งคน = หนึ่งครั้งที่จ่ายค่าโมเดล · ไม่มีเพดาน = บิลบานโดยไม่มีใครเห็น"""
    r = client.post("/v1/persons", json={
        "camera_id": "c",
        "objects": [{"id": f"p{i}", "image_base64": _png(8, 8)} for i in range(17)]})
    assert r.status_code == 422


def test_ยิงผลกลับมาดูได้ด้วย_request_id():
    d = _post([{"id": "ID-1", "image_base64": _png()}])
    r = client.get(f"/v1/persons/{d['request_id']}")
    assert r.status_code == 200
    assert r.json()["persons"][0]["object_id"] == "ID-1"
    assert client.get("/v1/persons/ไม่มีอยู่จริง").status_code == 404


def test_healthz_ตอบได้แม้โมเดลใช้ไม่ได้():
    h = client.get("/healthz").json()
    assert h["service"] == "smart-people-counting"
    assert h["prompt_version"] == "p1"
    # ค่าเริ่มต้นต้องไม่เก็บภาพคน · ถ้าวันหนึ่งมีคนเปลี่ยน default เทสต์นี้จะดัง
    assert h["save_images"] == "none"
