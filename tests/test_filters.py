"""เทสต์ชั้นกรองตามเวลา · ไม่ต้องมี LLM ไม่ต้องมี DB ไม่ต้องมีเน็ต

เทสต์สำคัญที่สุดในไฟล์นี้คือ test_elephant_arriving_is_not_filtered_out
ถ้าข้อนั้นตก แปลว่าระบบเตือนภัยกรองเหตุการณ์ที่มันมีไว้เพื่อจับทิ้ง

    python tests/test_filters.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.filters.temporal import HistoryItem, decide, pooled_confidence  # noqa: E402

T0 = 1_000_000.0


def hist(counts_seq, conf=0.8, start=T0, step=10.0):
    """สร้างประวัติจากลำดับจำนวน เช่น [0,0,0,2] = สามเฟรมว่าง แล้วเจอ 2 ตัว"""
    return [
        HistoryItem(ts=start + i * step,
                    counts={"elephant": c} if c else {},
                    confidence={"elephant": conf} if c else {})
        for i, c in enumerate(counts_seq)
    ]


def test_cold_start_is_honest():
    """หน้าต่างว่าง = หลัง deploy ทุกครั้ง (SQLite หายไปกับ container)

    ต้องบอกตรงๆ ว่า cold start ไม่ใช่แกล้งกรองแล้วตอบเหมือนปกติ
    """
    f, w = decide({"elephant": 2}, {"elephant": 0.8}, [], T0)
    assert f.state == "cold_start"
    assert "cold start" in f.reason
    assert w.complete is False and w.frames_used == 0
    assert f.counts == {"elephant": 2}, "cold start ต้องยังตอบค่าดิบ ไม่ใช่กลืนทิ้ง"


def test_elephant_arriving_is_not_filtered_out():
    """🔴 ข้อสำคัญที่สุดของทั้งไฟล์

    ช้างเดินเข้าเฟรม = 0,0,0,0,0 แล้วจู่ๆ เป็น 3
    Hampel ตรงๆ จะบอกว่า '3 ห่างจาก median 0 เกินไป ตัดทิ้ง' แล้วเงียบในนาทีที่สำคัญที่สุด
    fast attack ต้องปล่อยผ่าน
    """
    h = hist([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    f, _ = decide({"elephant": 3}, {"elephant": 0.82}, h, T0 + 100)
    assert f.counts.get("elephant") == 3, f"ช้างมาแล้วต้องรายงาน ได้ {f.counts}"
    assert f.state == "rising"
    assert f.accepted is True


def test_single_crazy_reading_is_held():
    """เฟรมเดียวที่ตอบว่าเห็นช้าง 40 ตัว ทั้งที่ 10 เฟรมก่อนหน้าเห็น 3 ตัวนิ่งๆ

    ค่ากระโดดขึ้นก็จริง แต่ fast attack ตั้งใจให้ผ่าน — เราจึงยอมรับตัวเลข
    แต่ต้องมี reason อธิบายว่าเกิดอะไรขึ้น ไม่ใช่ผ่านเงียบๆ
    """
    h = hist([3] * 10)
    f, _ = decide({"elephant": 40}, {"elephant": 0.7}, h, T0 + 100)
    assert "jump" in f.reason, f"ต้องบันทึกว่ามันกระโดด: {f.reason}"
    assert f.state == "rising"


def test_disappearance_is_held_back():
    """ช้าง 3 ตัวนิ่งมา 10 เฟรม แล้วเฟรมเดียวเห็น 0

    อาจเป็นช้างเดินออกจริง หรืออาจเป็นเมฆบัง/โมเดลพลาด
    slow release: ห้ามประกาศว่าไม่มีแล้วจากเฟรมเดียว
    """
    h = hist([3] * 10)
    f, _ = decide({}, {}, h, T0 + 100)
    assert f.counts.get("elephant") == 3, f"ยังไม่ควรประกาศว่าหมด ได้ {f.counts}"
    assert f.accepted is False
    assert f.state == "falling"
    assert "held" in f.reason


def test_real_departure_is_eventually_accepted():
    """ถ้าหายไปจริงหลายเฟรมติด ต้องยอมรับ ไม่ใช่ค้างตลอดกาล"""
    h = hist([3, 3, 3, 3, 3, 3, 0, 0, 0, 0])
    f, _ = decide({}, {}, h, T0 + 100)
    assert f.counts.get("elephant", 0) == 0, f"หายจริงต้องยอมรับ ได้ {f.counts}"
    assert f.state == "falling"


def test_low_confidence_waits_for_next_frame():
    """conf < 0.60 → provisional ตอบทันที ไม่ค้าง HTTP request แต่ยังไม่ commit"""
    h = hist([0] * 5)
    f, _ = decide({"elephant": 2}, {"elephant": 0.52}, h, T0 + 50)
    assert f.accepted is False
    assert f.state == "unconfirmed"
    assert "awaiting next frame" in f.reason


def test_low_confidence_confirmed_by_next_frame():
    """เฟรมถัดไปเห็นตรงกัน → commit พร้อมบอกว่ายืนยันจากกี่เฟรม"""
    h = hist([0, 0, 0, 2], conf=0.55)
    f, _ = decide({"elephant": 2}, {"elephant": 0.55}, h, T0 + 40)
    assert f.accepted is True
    assert f.corroborated_frames >= 2
    assert f.confidence > 0.55, "รวมสองเฟรมแล้วต้องมั่นใจขึ้นเล็กน้อย"


def test_conflicting_frames_report_unstable():
    """เฟรมนี้บอกช้าง เฟรมที่แล้วบอกวัว · ห้ามเลือกให้ ต้องบอกว่ามันขัดกัน

    สถานะนี้เคยอยู่ใน schema และในเอกสารตั้งแต่แรก **แต่โค้ดไม่เคยสร้างมันเลย**
    เจอตอน Toy ถามว่า state มีค่าอะไรบ้าง (2026-08-16) ถ้าไม่เจอ
    ทีมปลายทางจะเขียนโค้ดรอรับค่าที่ไม่มีวันมา
    """
    h = [HistoryItem(ts=T0 + i * 10, counts={"cattle": 2}, confidence={"cattle": 0.7})
         for i in range(5)]
    f, _ = decide({"elephant": 2}, {"elephant": 0.8}, h, T0 + 50)
    assert f.state == "unstable", f"ต้องเป็น unstable ได้ {f.state}"
    assert f.accepted is False
    assert "elephant" in f.reason and "cattle" in f.reason, \
        "reason ต้องบอกทั้งสองคำตอบ ปลายทางจะได้ตัดสินเอง"


def test_same_species_is_not_a_conflict():
    """ชนิดเดิมแต่จำนวนต่าง ไม่ใช่ความขัดแย้ง ปล่อยให้ Hampel จัดการตามปกติ"""
    h = [HistoryItem(ts=T0 + i * 10, counts={"elephant": 2}, confidence={"elephant": 0.7})
         for i in range(5)]
    f, _ = decide({"elephant": 3}, {"elephant": 0.8}, h, T0 + 50)
    assert f.state != "unstable"


def test_unknown_is_not_a_conflict():
    """unknown ไม่ถือว่าขัดกับชนิดใด มันแปลว่า 'ดูไม่ออก' ไม่ใช่ 'เป็นอย่างอื่น'"""
    h = [HistoryItem(ts=T0 + i * 10, counts={"unknown": 2}, confidence={"unknown": 0.9})
         for i in range(5)]
    f, _ = decide({"elephant": 2}, {"elephant": 0.8}, h, T0 + 50)
    assert f.state != "unstable"


def test_unknown_never_waits_for_the_next_frame():
    """🔴 regression · "ดูไม่ออก" คือคำตอบที่จบแล้ว ไม่ใช่คำตอบชั่วคราว

    วัดเจอ 2026-08-16 ว่าโมเดลให้ confidence กับคำตอบ unknown คนละความหมาย:
      qwen3.7-flash   0.00-0.20  ("ไม่มั่นใจอะไรเลย")
      gemma-3-12b-it  0.90-1.00  ("มั่นใจมากว่าดูไม่ออก")
    เกณฑ์ 0.60 เลยอ่านสองอย่างนี้ต่างกัน ทั้งที่มันพูดเรื่องเดียวกัน
    ปล่อยไว้ = unknown ของ flash ค้าง provisional ทุกเฟรมตลอดกาล
    """
    h = hist([0] * 5)
    f, _ = decide({"unknown": 2}, {"unknown": 0.0}, h, T0 + 50)
    assert f.state != "unconfirmed", "unknown ต้องไม่ค้างรอเฟรมถัดไป"
    assert "awaiting" not in f.reason
    assert f.counts.get("unknown") == 2, "แต่ต้องยังรายงานว่ามีของร้อน 2 ก้อนที่ระบุไม่ได้"


def test_named_species_with_low_confidence_still_waits():
    """แต่ถ้าโมเดลบอกชนิดมาแล้วไม่มั่นใจ ('น่าจะช้าง 0.5') การรอยังมีเหตุผล

    เฟรมถัดไปช่วยยืนยันได้จริงในกรณีนี้ ต่างจาก unknown
    """
    h = hist([0] * 5)
    f, _ = decide({"elephant": 2}, {"elephant": 0.5}, h, T0 + 50)
    assert f.state == "unconfirmed"


def test_provisional_does_not_wait_forever():
    """รอได้สูงสุด 2 เฟรม (20 วิ = ช้างเดินไป ~28 ม.) เกินนั้นต้อง commit"""
    h = hist([0] * 5)
    f, _ = decide({"elephant": 2}, {"elephant": 0.4}, h, T0 + 50, provisional_streak=2)
    assert f.accepted is True, "เกินเพดานแล้วต้อง commit ไม่ค้างไม่มีกำหนด"


def test_stale_frames_fall_out_of_window():
    """กล้องเน็ตหลุด 3 นาทีแล้วกลับมา ของเก่าต้องไม่ถูกนับ

    ถ้านับ '10 เฟรมล่าสุด' แทนที่จะนับ '100 วิล่าสุด' ข้อนี้จะตก
    """
    old = hist([3] * 10, start=T0 - 400)
    _, w = decide({"elephant": 1}, {"elephant": 0.9}, old, T0)
    assert w.frames_used == 0, f"ของเกิน 100 วิต้องหลุดออก เหลือ {w.frames_used}"
    assert w.complete is False


def test_window_reports_incompleteness():
    """ข้อมูลไม่ครบต้องบอก ปลายทางจะได้ลดน้ำหนักคำตอบเอง"""
    _, w = decide({"elephant": 1}, {"elephant": 0.9}, hist([1, 1, 1]), T0 + 30)
    assert w.frames_used == 3 and w.frames_expected == 10 and w.complete is False


def test_pooled_confidence_is_not_naive_bayes():
    """🔴 0.52 + 0.55 ต้องไม่กลายเป็น 0.78

    สูตร 1-Π(1-c) ตั้งบนสมมติฐานว่าอิสระต่อกัน ซึ่งไม่จริงสำหรับเฟรมห่าง 10 วิ
    ที่เป็นสัตว์ตัวเดิม มุมเดิม โมเดลเดิม
    """
    v = pooled_confidence([0.52, 0.55])
    assert v < 0.65, f"รวมแล้วต้องไม่พองเกินจริง ได้ {v}"
    assert v >= 0.52, "แต่ก็ต้องไม่ต่ำกว่าเฟรมเดียว"
    assert pooled_confidence([0.5] * 20) <= 0.85, "ต้องมีเพดาน"


def test_history_never_mutated():
    """ชั้นนี้เป็นฟังก์ชันบริสุทธิ์ ห้ามแก้ของที่รับเข้ามา"""
    h = hist([2, 2, 2])
    before = [(x.ts, dict(x.counts)) for x in h]
    decide({"elephant": 2}, {"elephant": 0.9}, h, T0 + 30)
    assert [(x.ts, dict(x.counts)) for x in h] == before


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as e:
                fails += 1
                print(f"  FAIL  {name}: {e}")
            except Exception as e:
                fails += 1
                print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{'ตกทั้งหมด ' + str(fails) + ' ข้อ' if fails else 'ผ่านหมด'}")
    sys.exit(1 if fails else 0)
