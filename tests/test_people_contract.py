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
import uuid

os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ["PEOPLE_STORE_DSN"] = "./data/people-contract-test.db"
os.environ["API_KEY"] = ""
os.environ["PEOPLE_API_KEY"] = ""
os.environ["LLM_BASE_URL"] = "http://127.0.0.1:9"   # ต่อไม่ติดแน่นอน
os.environ["LLM_TIMEOUT_S"] = "1"
os.environ["PEOPLE_LLM_MAX_RETRIES"] = "0"  # เทสต์ไม่ต้องรอ retry ของที่ต่อไม่ติดอยู่แล้ว

from fastapi.testclient import TestClient  # noqa: E402

import people.main as pm  # noqa: E402

TOP = {"request_id", "ref", "camera_id", "received_at",
       "status", "reason", "persons", "summary", "model", "timing_ms"}
PERSON = {"id", "status", "where", "direction", "direction_confidence",
          "gender", "gender_confidence", "age_range",
          "age_range_confidence", "age_group", "nationality",
          "nationality_confidence", "group", "emotion",
          "appearance", "appearance_confidence",
          "description", "overall_confidence", "reason", "image", "model",
          "timing_ms"}
APPEARANCE = {"skin_tone", "build", "height", "hair_length", "hair_color",
              "glasses", "face_mask", "facial_hair", "headwear", "top",
              "top_sleeve", "outer", "bottom", "footwear", "carrying", "uniform",
              "distinctive"}
GARMENT = {"type", "color", "secondary_color", "pattern"}
UNIFORM = {"kind", "color", "id_badge", "text"}
GROUP = {"ref", "size", "type"}
EMOTION = {"label", "valence", "confidence"}
MODEL = {"provider", "name", "prompt_version", "finish_reason", "completion_tokens"}
TIMING = {"decode", "vlm", "vlm_sum", "total"}
# 🔴 ยอดรวมของ Demographic Analytics · Toy สั่ง 2026-09-12
# ทุกชุดต้องมี unknown และต้องบวกได้ครบจำนวนคน ดู main._demographic_counts
DEMOGRAPHIC = {"age_child", "age_teen", "age_adult", "age_senior", "age_unknown",
               "gender_male", "gender_female", "gender_unknown",
               "in_uniform", "uniform_unknown",
               "group_alone", "group_pair", "group_3_plus", "group_unknown",
               "groups_found",
               "emotion_positive", "emotion_neutral", "emotion_negative",
               "emotion_unknown"}
SUMMARY = {"objects_in", "ok", "degraded", "error",
           "direction_in", "direction_out", "direction_unknown"} | DEMOGRAPHIC

client = TestClient(pm.app)

# 🔴 ค่าใน response เป็น **ภาษาไทย** ตั้งแต่ 2026-09-12 (คีย์ยังเป็นอังกฤษ)
# เทสต์อ้าง `people.th.FIELD` ไม่ใช่พิมพ์คำไทยไว้เอง · พิมพ์เองเมื่อไหร่
# ก็จะได้ตารางแปลชุดที่สามที่ต้องคอยไล่แก้ตาม ซึ่งคือของที่เพิ่งถอดออกจากหน้าเว็บไป
from people.th import FIELD as TH  # noqa: E402

UNKNOWN = TH["direction"]["unknown"]


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
    # model ถูกซ่อนเป็น null โดยค่าเริ่มต้น · รูปของมันยังต้องถูกล็อกไว้
    # เพราะพอเปิด PEOPLE_EXPOSE_MODEL=true มันจะโผล่มาเป็นรูปนี้
    assert d["model"] is None
    from people.schemas import ModelInfo
    assert set(ModelInfo.model_fields) == MODEL
    assert set(d["timing_ms"]) == TIMING
    p = d["persons"][0]
    assert set(p) == PERSON
    assert set(p["appearance"]) == APPEARANCE
    for part in ("top", "outer", "bottom", "footwear"):
        assert set(p["appearance"][part]) == GARMENT
    assert set(p["appearance"]["uniform"]) == UNIFORM
    assert set(p["group"]) == GROUP
    assert set(p["emotion"]) == EMOTION
    assert set(p["image"]) == {"w", "h", "source", "bbox_used"}


def test_model_ถูกซ่อนเป็น_null_โดยค่าเริ่มต้น():
    """🔴 ปลายทางไม่ต้องรู้ว่าเราใช้โมเดลอะไร (Toy สั่ง 2026-09-11)

    คีย์ต้องยังอยู่ · รูป response ที่เปลี่ยนตามค่า env คือของที่ทำให้ปลายทาง
    parse พังแบบหาสาเหตุไม่เจอ
    """
    d = _post([{"id": "ID-1", "image_base64": _png()}])
    assert "model" in d and d["model"] is None
    assert "model" in d["persons"][0] and d["persons"][0]["model"] is None
    # และห้ามหลุดทางหน้า healthz ซึ่งเปิดได้โดยไม่ต้องมีคีย์
    h = client.get("/healthz").json()
    assert h["model"] is None and h["provider"] is None


def test_direction_มีค่าเสมอและยอดรวมต้องครบ():
    """in + out + unknown ต้องเท่าจำนวนคนเสมอ ไม่งั้นปลายทางบวกยอดแล้วขาด"""
    d = _post([{"id": "a", "image_base64": _png()}, {"id": "b", "image_base64": _png()}])
    s = d["summary"]
    assert s["direction_in"] + s["direction_out"] + s["direction_unknown"] == 2
    for p in d["persons"]:
        assert p["direction"] in set(TH["direction"].values())


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
    assert p["gender"] == TH["gender"]["unknown"]
    # age_range เป็นตัวเลข (`20-29`) ไม่ได้แปล · `unknown` ของมันจึงยังเป็นคำอังกฤษ
    # 🔴 ตั้งใจ ไม่ใช่ตกหล่น · ฟิลด์ที่ค่าเป็นช่วงตัวเลข แปลแล้วไม่ได้อะไรขึ้นมา
    # แต่เสียความสามารถในการ sort/compare ของปลายทางไป
    assert p["age_range"] == "unknown"
    assert p["age_group"] == TH["age_group"]["unknown"]
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
    # ซ่อนที่ response แล้วแต่ยังโชว์ที่ healthz = ซ่อนไม่สำเร็จ
    # หน้านี้เปิดได้โดยไม่ต้องมีคีย์ด้วยซ้ำ
    assert h["prompt_version"] is None
    assert h["frame_prompt_version"] is None
    # ค่าเริ่มต้นต้องไม่เก็บภาพคน · ถ้าวันหนึ่งมีคนเปลี่ยน default เทสต์นี้จะดัง
    assert h["save_images"] == "none"


# ---------------------------------------------------------------- /v1/frames
# เส้นที่สอง · body หน้าตาเหมือนฝั่งช้างเป๊ะ ไม่มีใครถูกชี้ โมเดลหาคนเอง

def test_body_ของ_frames_รับของฝั่งช้างได้ครบทุกฟิลด์():
    """🔴 ปลายทางที่ยิงฝั่งช้างอยู่แล้วต้องเปลี่ยนแค่ path ตามที่ Toy สั่ง

    เทียบกับ `app.schemas.FrameIn` **ของจริง** ไม่ใช่รายการที่พิมพ์ไว้เอง
    ฝั่งโน้นขยับเมื่อไหร่ที่นี่รู้ทันที

    เดิมเทสต์นี้บังคับให้ "เท่ากันเป๊ะ" · คลายเป็น "ต้องรับได้ครบ" ตอนเพิ่ม
    `client_request_id` ซึ่งเป็น optional · ของที่เพิ่มได้มีเงื่อนไขเดียว
    **ต้องไม่บังคับ** ไม่งั้น body เดิมของฝั่งช้างจะยิงไม่ผ่าน ซึ่งผิดคำสัญญา
    """
    from app.schemas import FrameIn
    from people.schemas import PeopleFrameIn
    ours, theirs = PeopleFrameIn.model_fields, FrameIn.model_fields
    assert set(theirs) <= set(ours), f"รับของฝั่งช้างไม่ครบ: {set(theirs) - set(ours)}"
    for name in set(ours) - set(theirs):
        assert not ours[name].is_required(), f"{name} เป็นฟิลด์บังคับ ทำให้ body เดิมพัง"


def test_frames_ตอบเต็มรูปแม้โมเดลล่ม_และบอกเหตุผลที่ระดับ_request():
    """ไม่มีคนสักคนให้ใส่เหตุผลไว้ข้างใน เหตุผลจึงต้องอยู่ระดับบน"""
    r = client.post("/v1/frames", json={"camera_id": "cam-f", "image_base64": _png(640, 480)})
    assert r.status_code == 200, r.text
    d = r.json()
    assert set(d) == TOP
    assert d["status"] == "degraded"
    assert d["persons"] == []
    assert d["reason"], "degraded ต้องบอกเหตุผล ไม่ใช่เงียบ"
    assert d["summary"]["people_found"] == 0


def test_frames_รับ_ts_กับ_note_แบบเดียวกับช้าง():
    r = client.post("/v1/frames", json={
        "camera_id": "cam-f", "image_base64": _png(64, 64),
        "ts": "2026-08-16T06:30:00+07:00", "note": "optional free text"})
    assert r.status_code == 200
    assert r.json()["received_at"].startswith("2026-08-15T23:30"), "ts ที่ส่งมาต้องถูกใช้"


def test_frames_ภาพเสียตอบ_400():
    r = client.post("/v1/frames", json={"camera_id": "c", "image_base64": "!!!"})
    assert r.status_code == 400


def test_ไม่มีคนในเฟรม_ไม่ใช่ความล้มเหลว():
    """🔴 {"people":[]} = ล็อบบี้ว่าง ซึ่งเป็นคำตอบจริง

    ต้องแยกจาก "โมเดลพัง" ให้ขาด ไม่งั้นเฟรมที่โมเดลพังจะดูเหมือนล็อบบี้ว่างเป๊ะ
    ซึ่งคือบั๊กเดียวกับที่ฝั่งช้างเสียเวลาไปทั้งวัน
    """
    from people.llm.client import people_of
    empty, err = people_of({"people": []})
    assert empty == [] and err is None
    missing, err2 = people_of({"animals": []})
    assert missing == [] and err2, "ไม่มีคีย์ people = โมเดลไม่ทำตามที่สั่ง คนละเรื่องกับว่าง"


def test_frames_เกินเพดานคนถูกตัด_ไม่ใช่ตอบยาวไม่จบ():
    from people.llm.client import people_of
    many = {"people": [{"ref": f"P{i}"} for i in range(50)]}
    got, err = people_of(many, cap=12)
    assert err is None and len(got) == 12


def test_where_มีเฉพาะเส้น_frames():
    """เส้น persons ปลายทางชี้เองอยู่แล้ว ไม่ต้องมีใครมาบอกว่าคนไหน"""
    d = _post([{"id": "ID-1", "image_base64": _png()}])
    assert d["persons"][0]["where"] == ""


# ---------------------------------------------------------------- ยิงผิดเส้น
# เจอจริง 2026-09-11: Toy ส่ง body แบบเฟรมเดียวมาที่ /v1/persons แล้วได้
# `objects: Field required` ซึ่งถูกแต่ช่วยอะไรไม่ได้ · ทีมปลายทางจะเจอเหมือนกัน

def test_ส่ง_body_แบบเฟรมมาที่_persons_ต้องบอกว่าไปผิดเส้น():
    r = client.post("/v1/persons", json={"camera_id": "cam-01",
                                         "image_base64": _png(), "note": "x"})
    assert r.status_code == 422
    j = r.json()
    assert "/people/v1/frames" in j["hint"]
    # detail ต้องเป็นรูปเดิมของ FastAPI ปลายทางที่ parse อยู่แล้วห้ามพัง
    assert isinstance(j["detail"], list) and j["detail"][0]["type"] == "missing"


def test_ส่ง_body_แบบ_objects_มาที่_frames_ต้องบอกว่าไปผิดเส้น():
    r = client.post("/v1/frames", json={"camera_id": "c",
                                        "objects": [{"id": "a", "image_base64": _png()}]})
    assert r.status_code == 422
    assert "/people/v1/persons" in r.json()["hint"]


def test_เกินเพดานคนบอกว่าเกินเท่าไร():
    r = client.post("/v1/persons", json={
        "camera_id": "c",
        "objects": [{"id": f"p{i}", "image_base64": _png(8, 8)} for i in range(20)]})
    assert r.status_code == 422
    assert "20" in r.json()["hint"]


def test_frames_บอกว่าภาพที่อ่านคือเฟรมเต็ม_ไม่ใช่ครอปของใคร():
    """🔴 เคยคืน object_image ซึ่งแปลว่า "ปลายทางครอปมาให้แล้ว" ซึ่งไม่จริง

    ค่าที่ไม่มีให้เลือก ห้ามหยิบค่าที่ใกล้ที่สุดมาใส่แทน ทั้งฟิลด์จะเชื่อไม่ได้
    """
    r = client.post("/v1/frames", json={"camera_id": "c", "image_base64": _png(640, 480)})
    assert r.status_code == 200
    # โมเดลต่อไม่ติดในเทสต์นี้ persons จึงว่าง · ตรวจที่ schema แทนว่าค่าใหม่มีจริง
    from people.schemas import ImageInfo
    import typing
    assert "full_frame" in typing.get_args(ImageInfo.model_fields["source"].annotation)


# ---------------------------------------------------------------- อ้างอิงกลับ
# "ผู้ใช้จะรู้ได้ยังไงว่า response เป็นของตัวเอง" · คำถามของ Toy 2026-09-11

def test_เลขอ้างอิงส่งมาใน_note_แล้วคืนกลับใน_ref():
    """🔴 ทางที่ทีมคุณสุชาติจะใช้จริง (Toy เคาะ 2026-09-11)

    เขาส่งเลขอ้างอิงมาในช่อง `note` ซึ่งมีอยู่แล้วใน spec ฝั่งช้าง
    ไม่ต้องแก้ body ที่ส่งอยู่เลย แล้วรับกลับที่ `ref`
    """
    ref = "iplus-2026-09-11-000123"
    f = client.post("/v1/frames", json={
        "camera_id": "cam-01", "image_base64": _png(), "note": ref}).json()
    assert f["ref"] == ref

    d = client.post("/v1/persons", json={
        "camera_id": "c", "note": ref,
        "objects": [{"id": "ID-1", "image_base64": _png()}]}).json()
    assert d["ref"] == ref


def test_client_request_id_ชนะ_note_ถ้าส่งมาทั้งคู่():
    """ช่องที่ชื่อตรงกว่าต้องชนะ ไม่ใช่เอาอันหลังทับเงียบๆ"""
    d = client.post("/v1/frames", json={
        "camera_id": "c", "image_base64": _png(),
        "client_request_id": "ตัวจริง", "note": "ข้อความธรรมดา"}).json()
    assert d["ref"] == "ตัวจริง"


def test_ไม่ส่งเลขอ้างอิงมาก็ได้_ref_เป็น_null():
    """ของเดิมที่ยิงอยู่แล้วต้องไม่พังเพราะฟิลด์ใหม่"""
    d = _post([{"id": "ID-1", "image_base64": _png()}])
    assert d["ref"] is None


def test_ref_ต้องคืนตัวเดิมเป๊ะ_ห้ามแตะ():
    """ปลายทางเอาไปเทียบว่า 'ใช่ของฉันไหม' · trim หรือแปลงอะไรสักอย่าง = เทียบไม่ตรง"""
    weird = "  ref/ที่มี ช่องว่าง+อักขระ_แปลก#42  "
    d = client.post("/v1/frames", json={
        "camera_id": "c", "image_base64": _png(), "note": weird}).json()
    assert d["ref"] == weird


def test_ตามผลด้วยเลขอ้างอิงของตัวเองได้():
    """เคสจริง: ยิงไปแล้ว timeout ไม่เคยเห็น request_id ของเรา"""
    ref = f"ref-timeout-{uuid.uuid4()}"
    # ส่งมาทาง note ซึ่งเป็นทางที่ปลายทางจะใช้จริง · lookup ต้องหาเจอเหมือนกัน
    client.post("/v1/frames", json={"camera_id": "c", "image_base64": _png(),
                                    "note": ref})
    r = client.get(f"/v1/persons/by-ref/{ref}")
    assert r.status_code == 200
    j = r.json()
    assert j["ref"] == ref and j["matches"] >= 1
    assert j["latest"]["request_id"]
    assert client.get("/v1/persons/by-ref/ไม่เคยส่ง").status_code == 404


def test_ส่ง_ref_ซ้ำ_ได้หลายผล_และบอกว่ากี่อัน():
    """เราไม่การันตีว่าไม่ซ้ำ เพราะเลขนี้ไม่ใช่ของเรา · ต้องบอกตรงๆ ไม่ใช่แกล้งเลือกให้"""
    # 🔴 ref ต้องไม่ซ้ำกับการรันครั้งก่อน · ฐานข้อมูลเทสต์อยู่ข้ามรอบ
    # เขียนครั้งแรกด้วยค่าคงที่ แล้วมันเขียวรอบแรกแล้วตกรอบสอง (assert 6 == 2)
    # ซึ่งแย่กว่าตกตั้งแต่แรก เพราะรอบแรกมันบอกว่าผ่าน
    ref = f"ref-ซ้ำ-{uuid.uuid4()}"
    for _ in range(2):
        client.post("/v1/frames", json={"camera_id": "c", "image_base64": _png(),
                                        "note": ref})
    assert client.get(f"/v1/persons/by-ref/{ref}").json()["matches"] == 2


# --------------------------------------------- Demographic Analytics 2026-09-12

def test_ยอดรวม_demographic_ต้องบวกได้ครบจำนวนคนทุกชุด():
    """🔴 กฎเดียวกับ direction · ปลายทางเอาไปทำกราฟแล้วยอดต้องตรงเสมอ

    ยอดที่ขาดไปเงียบๆ ทำให้เขาไล่หาคนที่หายไป โดยที่ไม่มีใครหายไปจริง
    เทสต์นี้เดินผ่านเส้นที่โมเดลล่มทั้งหมด ซึ่งคือเคสที่ยอดต้องยังครบอยู่ดี
    """
    n = 3
    d = _post([{"id": f"p{i}", "image_base64": _png()} for i in range(n)])
    s = d["summary"]
    assert s["age_child"] + s["age_teen"] + s["age_adult"] + s["age_senior"] \
        + s["age_unknown"] == n
    assert s["gender_male"] + s["gender_female"] + s["gender_unknown"] == n
    assert s["group_alone"] + s["group_pair"] + s["group_3_plus"] \
        + s["group_unknown"] == n
    assert s["emotion_positive"] + s["emotion_neutral"] + s["emotion_negative"] \
        + s["emotion_unknown"] == n
    assert s["in_uniform"] <= n and s["uniform_unknown"] <= n


def test_ยอดรวม_demographic_ของเส้น_frames_ต้องเท่ากับ_people_found():
    d = client.post("/v1/frames", json={"camera_id": "cam-1",
                                        "image_base64": _png(400, 300)}).json()
    s = d["summary"]
    found = s["people_found"]
    assert s["age_child"] + s["age_teen"] + s["age_adult"] + s["age_senior"] \
        + s["age_unknown"] == found
    assert s["emotion_positive"] + s["emotion_neutral"] + s["emotion_negative"] \
        + s["emotion_unknown"] == found
    assert s["groups_found"] <= found


def test_group_ของเส้น_persons_เป็น_unknown_เสมอ_ซึ่งถูกแล้ว():
    """เส้นนี้ยิงโมเดลทีละคนจาก crop ของคนนั้น ไม่มีใครเห็นภาพรวม

    🔴 ตอบ alone ที่นี่คือการโกหกว่า "คนนี้มาคนเดียว" ทั้งที่เราแค่ไม่ได้ดู
    ดู schemas.GroupType
    """
    d = _post([{"id": "a", "image_base64": _png()}])
    g = d["persons"][0]["group"]
    assert g == {"ref": "", "size": 0, "type": UNKNOWN}
    assert d["summary"]["group_unknown"] == 1
    assert d["summary"]["groups_found"] == 0


def test_อารมณ์เป็นสเกล_1_ถึง_5_และ_0_คือไม่รู้():
    """1 = bad, 5 = very happy (Toy เคาะ 2026-09-12) · 0 = มองไม่เห็นหน้า"""
    d = _post([{"id": "a", "image_base64": _png()}])
    e = d["persons"][0]["emotion"]
    assert set(e) == EMOTION
    assert isinstance(e["valence"], int) and 0 <= e["valence"] <= 5
    # โมเดลต่อไม่ติดในเทสต์ = ไม่เคยดูหน้าใคร = ต้องเป็น 0 ไม่ใช่ 3
    assert e["valence"] == 0 and e["label"] == TH["emotion.label"]["unknown"]


def test_สัญชาติปิดอยู่โดยค่าเริ่มต้น_และ_healthz_บอกตรงๆ_ว่าปิด():
    """🔴 คีย์ยังอยู่เสมอ รูป response ไม่เปลี่ยนตามค่า env (กฎเดียวกับ model)

    แต่ต่างจาก model ตรงที่ **ต้องบอกปลายทางว่าปิดอยู่** ไม่งั้นเขาจะนั่งไล่หาว่า
    ทำไมฟิลด์นี้ไม่เคยมีค่า แล้วโทษโมเดลทั้งที่เราปิดไว้เอง
    """
    d = _post([{"id": "a", "image_base64": _png()}])
    p = d["persons"][0]
    assert p["nationality"] == UNKNOWN and p["nationality_confidence"] == 0.0
    h = client.get("/healthz").json()
    assert h["nationality_enabled"] is False
    assert "1=bad" in h["emotion_scale"] and "5=very happy" in h["emotion_scale"]


def test_uniform_อยู่ใน_appearance_และมีค่าเริ่มต้นเป็น_unknown():
    d = _post([{"id": "a", "image_base64": _png()}])
    u = d["persons"][0]["appearance"]["uniform"]
    assert u == {"kind": UNKNOWN, "color": UNKNOWN,
                 "id_badge": TH["uniform.id_badge"]["unknown"], "text": ""}
