"""🔴 สัญญาของ JSON ที่ตอบกลับ · confirm กับทีมคุณสุชาติไปแล้ว ห้ามเปลี่ยนรูป

Toy สั่งไว้ 2026-08-17: "json response spec ต้องเหมือนเดิมนะครับ เพราะ confirm
กับ team คุณสุชาติไว้ ห้ามเปลี่ยนแปลง"

เทสต์นี้ไม่ได้ดูว่าค่าถูกไหม ดูว่า **รูปยังเหมือนเดิมไหม** ฟิลด์หายคือปลายทางพัง
ฟิลด์เกินคือปลายทางที่ validate เข้มจะปฏิเสธทั้งก้อน สองอย่างนี้แพงพอกัน

ระหว่างเปลี่ยนมาเป็นท่อ llm-first เคยเผลอเพิ่ม count_source กับ image_type เข้าไป
ตอนนั้นดูสมเหตุสมผลมาก เพราะความหมายของ counts เปลี่ยนไปจริงๆ แต่มันคือการ
เปลี่ยนสัญญาที่ตกลงไว้แล้ว **เทสต์นี้มีไว้ให้ความสมเหตุสมผลแบบนั้นดังขึ้นตอน CI
ไม่ใช่ตอนปลายทางพัง**
"""
import base64
import os
import pathlib

os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ["STORE_DSN"] = "./data/contract-test.db"
os.environ["API_KEY"] = ""
os.environ["LLM_BASE_URL"] = "http://127.0.0.1:9"   # ต่อไม่ติดแน่นอน = ไม่ยิงเน็ตจริง
os.environ["LLM_TIMEOUT_S"] = "1"

from fastapi.testclient import TestClient  # noqa: E402

import app.main as m  # noqa: E402

TOP = {"request_id", "camera_id", "received_at", "status", "raw", "filtered",
       "window", "model", "timing_ms"}
RAW = {"animals", "counts", "regions_detected", "detections", "overall_confidence"}
FILTERED = {"counts", "confidence", "method", "accepted", "reason", "state",
            "corroborated_frames"}
WINDOW = {"span_seconds", "frames_used", "frames_expected", "complete",
          "median", "mad"}
MODEL = {"provider", "name", "prompt_version", "finish_reason", "completion_tokens"}
TIMING = {"cv", "vlm", "total"}


def _post():
    b64 = base64.b64encode(
        pathlib.Path("docs/samples/three-animals.png").read_bytes()).decode()
    client = TestClient(m.app)
    r = client.post("/v1/frames",
                    json={"camera_id": "cam-contract", "image_base64": b64})
    assert r.status_code == 200, r.text
    return r.json()


def test_รูปของ_response_ตรงกับที่ตกลงไว้เป๊ะ():
    d = _post()
    assert set(d) == TOP
    assert set(d["raw"]) == RAW
    assert set(d["filtered"]) == FILTERED
    assert set(d["window"]) == WINDOW
    assert set(d["model"]) == MODEL
    assert set(d["timing_ms"]) == TIMING


def test_ชนิดของค่ายังเหมือนเดิม():
    """ปลายทางเขียนโค้ดตามชนิดพวกนี้ไปแล้ว เปลี่ยนชนิดพังพอๆ กับเปลี่ยนชื่อ"""
    d = _post()
    assert isinstance(d["raw"]["counts"], dict)
    assert isinstance(d["raw"]["detections"], list)
    assert isinstance(d["raw"]["regions_detected"], int)
    assert isinstance(d["raw"]["overall_confidence"], float)
    assert isinstance(d["filtered"]["accepted"], bool)
    assert isinstance(d["window"]["complete"], bool)
    assert d["status"] in {"ok", "provisional", "degraded", "error"}
    assert d["filtered"]["state"] in {"stable", "rising", "falling", "unstable",
                                      "unconfirmed", "cold_start"}


def test_llm_ล่ม_ก็ยังตอบเต็มรูป_ไม่ใช่ตอบสั้นลง():
    """เส้น llm ไม่มีผล CV สำรองแล้ว ตอนโมเดลล่มจึงเสี่ยงจะตอบไม่ครบรูป

    เทสต์นี้ยิงไปที่พอร์ตที่ไม่มีอะไรฟัง = โมเดลล่มแน่นอน แล้วยังต้องได้ 200
    พร้อมฟิลด์ครบ ไม่ใช่ 500 หรือ JSON ที่หายไปครึ่งหนึ่ง
    """
    d = _post()
    assert d["status"] == "degraded"
    assert set(d) == TOP
    assert d["raw"]["counts"] == {}
