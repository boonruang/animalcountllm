"""เทสต์ชั้น CV ด้วยภาพสังเคราะห์ที่รู้คำตอบล่วงหน้า

รันได้โดย **ไม่ต้องมี LLM ไม่ต้องมี GPU ไม่ต้องมีเน็ต ไม่ต้องมีภาพจากไซต์**
นี่คือเหตุผลที่แยกชั้น CV ออกมา มันพิสูจน์ตัวเองได้

    python -m pytest tests/test_cv.py -v
    python tests/test_cv.py          (รันตรงๆ ไม่ต้องมี pytest ก็ได้)
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.cv import thermal  # noqa: E402


def make_frame(blobs, size=(640, 512), bg=40, noise=3, depth=8, blur=1.2):
    """สร้างภาพความร้อนเทียม: พื้นเย็น + ก้อนอุ่นตามที่สั่ง

    blobs = [(x, y, rx, ry, ความสว่าง), ...]
    """
    w, h = size
    im = Image.new("L", size, bg)
    d = ImageDraw.Draw(im)
    for (x, y, rx, ry, val) in blobs:
        d.ellipse([x - rx, y - ry, x + rx, y + ry], fill=val)
    im = im.filter(ImageFilter.GaussianBlur(blur))

    a = np.asarray(im).astype(np.float32)
    if noise:
        rng = np.random.default_rng(7)  # ตรึง seed เทสต์ต้องได้ผลเดิมทุกครั้ง
        a = a + rng.normal(0, noise, a.shape)
    a = np.clip(a, 0, 255)

    if depth == 16:
        im2 = Image.fromarray((a * 257).astype(np.uint16))
    else:
        im2 = Image.fromarray(a.astype(np.uint8))
    buf = io.BytesIO()
    im2.save(buf, format="PNG")
    return buf.getvalue()


def test_empty_frame_costs_nothing():
    """เฟรมว่าง = ไม่มีก้อน = ไม่ต้องยิง LLM = ไม่เสียเงิน

    นี่คือกรณีส่วนใหญ่ของกล้องจริง และเป็นที่มาของการประหยัด >90%
    """
    r = thermal.analyze(make_frame([]))
    assert r.blobs == [], f"เฟรมว่างไม่ควรเจอก้อน แต่เจอ {len(r.blobs)}"
    assert r.has_candidates is False


def test_counts_three_animals():
    r = thermal.analyze(make_frame([(120, 150, 26, 18, 220),
                                    (330, 240, 30, 20, 230),
                                    (500, 380, 24, 16, 215)]))
    assert len(r.blobs) == 3, f"ควรได้ 3 ก้อน ได้ {len(r.blobs)}"
    assert all(b.detection_confidence > 0.4 for b in r.blobs)


def test_small_animal_on_a_clean_background():
    """🔴 regression · เคยพลาดมาแล้ว 2026-08-16

    สัตว์ที่กินพื้นที่ไม่ถึง 0.5% ของเฟรม อยู่เหนือ percentile 99.5 ทั้งตัว
    บนพื้นหลังเรียบ (ไม่มี noise) p1 กับ p99.5 เลยเท่ากัน แล้ว normalize คืนภาพดำทั้งใบ
    **สัตว์หายเกลี้ยงโดยไม่มี error ให้เห็น** ซึ่งคือความล้มเหลวที่แย่ที่สุดของระบบเตือนภัย

    เทสต์เดิมทุกข้อใส่ noise ไว้ ซึ่งบังเอิญกลบบั๊กนี้หมด
    """
    for rx in (8, 12, 16, 20):
        r = thermal.analyze(make_frame([(320, 256, rx, int(rx * 0.75), 235)], noise=0))
        assert len(r.blobs) >= 1, f"ก้อน {rx}x{int(rx*0.75)} บนพื้นเรียบต้องเจอ ได้ {len(r.blobs)}"


def test_flat_frame_finds_nothing():
    """เฟรมแบนสนิทจริงๆ ต้องไม่กุก้อนขึ้นมา (อีกด้านของ regression ข้างบน)"""
    assert thermal.analyze(make_frame([], noise=0)).blobs == []


def test_deterministic():
    """รันภาพเดิมสองครั้งต้องได้ผลเท่ากันเป๊ะ

    นี่คือคุณสมบัติที่ LLM ไม่มี และเป็นเหตุผลที่ 'จำนวน' ต้องมาจากชั้นนี้
    """
    f = make_frame([(200, 200, 28, 20, 225), (450, 300, 22, 22, 210)])
    a, b = thermal.analyze(f), thermal.analyze(f)
    assert [x.model_dump() for x in a.blobs] == [x.model_dump() for x in b.blobs]


def test_rejects_single_pixel_noise():
    """พิกเซลเสียเดี่ยวๆ ต้องไม่กลายเป็นสัตว์"""
    r = thermal.analyze(make_frame([(100, 100, 1, 1, 255), (300, 300, 1, 1, 255)], blur=0.6))
    assert len(r.blobs) == 0, f"พิกเซลเดี่ยวไม่ควรนับ แต่นับได้ {len(r.blobs)}"


def test_small_blob_is_passed_up_not_dropped():
    """🔴 ก้อนขนาดกำกวม (~100 px) ต้อง **ผ่านขึ้นไปให้ LLM ตัดสิน** ไม่ใช่ตัดทิ้งเงียบๆ

    ช้างที่ระยะ 200 ม. กินราว 150 px นกใกล้ๆ ก็ 100 px เหมือนกัน
    **ชั้น CV แยกสองอย่างนี้ไม่ออกจริงๆ** เพราะภาพความร้อนไม่มีลาย ไม่มีสี
    เหลือแค่ขนาดเชิงมุม ซึ่งแปลเป็นขนาดจริงไม่ได้ถ้าไม่รู้ระยะ

    ตัดทิ้งที่ชั้นนี้ = ช้างไกลหายเงียบ ไม่มีใครรู้ว่าเคยมี
    ส่งขึ้นไป = เสียค่า LLM นิดหน่อย แต่ยังมีโอกาสตอบถูก และมี log ให้ย้อนดู
    เลือกอย่างหลัง เพราะต้นทุนของการพลาดสูงกว่าต้นทุนของการถามเกิน
    """
    r = thermal.analyze(make_frame([(100, 100, 2, 2, 240), (300, 300, 3, 2, 250)]))
    assert len(r.blobs) == 2, f"ก้อนกำกวมต้องผ่านขึ้นไป ได้ {len(r.blobs)}"
    big = thermal.analyze(make_frame([(320, 256, 28, 20, 240)])).blobs[0]
    assert all(b.area_px < big.area_px / 5 for b in r.blobs), \
        "แต่ต้องเล็กกว่าสัตว์เต็มตัวชัดเจน เพื่อให้ prompt บอก LLM ได้ว่านี่คือของกำกวม"


def test_rejects_huge_hot_region():
    """พื้นที่ร้อนขนาดครึ่งเฟรม = แดดส่องพื้น ไม่ใช่สัตว์"""
    r = thermal.analyze(make_frame([(320, 256, 300, 200, 230)]))
    assert len(r.blobs) == 0, "ก้อนใหญ่เกิน MAX_AREA_FRAC ต้องถูกตัด"


def test_16bit_radiometric_same_as_8bit():
    """กล้องส่ง 16-bit หรือ 8-bit ต้องนับได้เท่ากัน

    ยังไม่รู้ว่ากล้องที่ไซต์ส่งแบบไหน (CLAUDE.md 'ที่ยังไม่รู้') โค้ดเลยต้องรับได้ทั้งคู่
    """
    spec = [(150, 150, 26, 18, 220), (400, 300, 28, 20, 230)]
    r8 = thermal.analyze(make_frame(spec, depth=8))
    r16 = thermal.analyze(make_frame(spec, depth=16))
    assert r8.bit_depth == 8 and r16.bit_depth == 16
    assert len(r8.blobs) == len(r16.blobs) == 2


def test_gain_shift_does_not_change_count():
    """กล้อง thermal ปรับ gain เองตลอด ภาพสว่างขึ้นทั้งใบต้องนับได้เท่าเดิม

    ถ้าเทสต์นี้ตก แปลว่า normalize ใช้ค่าสัมบูรณ์อยู่ที่ไหนสักแห่ง
    """
    spec = [(200, 200, 28, 20, 200), (450, 350, 26, 18, 195)]
    dim = thermal.analyze(make_frame(spec, bg=30))
    bright = thermal.analyze(make_frame([(x, y, rx, ry, v + 40) for x, y, rx, ry, v in spec], bg=70))
    assert len(dim.blobs) == len(bright.blobs) == 2


def test_edge_blob_is_penalised():
    """ก้อนติดขอบเฟรมอาจโดนตัดครึ่ง ขนาดที่วัดได้เชื่อไม่ได้เต็มที่"""
    mid = thermal.analyze(make_frame([(320, 256, 28, 20, 225)])).blobs
    edge = thermal.analyze(make_frame([(6, 256, 28, 20, 225)])).blobs
    assert mid and edge
    assert edge[0].touches_edge is True
    assert edge[0].detection_confidence < mid[0].detection_confidence


def test_faint_blob_gets_lower_confidence():
    """ก้อนจางต้องได้ confidence ต่ำกว่าก้อนชัด **ในเฟรมเดียวกัน**

    ต้องเทียบในเฟรมเดียวกันเท่านั้น เพราะ normalize() ยืดสเกลตามเฟรม
    ก้อนจางในเฟรมที่ไม่มีอะไรอื่นเลย จะถูกยืดจนกลายเป็นก้อนชัด **ซึ่งตั้งใจให้เป็นแบบนั้น**
    (กล้อง thermal ปรับ gain เองตลอด ความสว่างสัมบูรณ์เชื่อไม่ได้ เหลือแต่ความต่างในเฟรม)
    เทสต์เวอร์ชันแรกเทียบข้ามเฟรมแล้วตก ซึ่งเทสต์ผิด ไม่ใช่โค้ดผิด
    """
    blobs = thermal.analyze(make_frame([(180, 200, 28, 20, 250),
                                        (460, 320, 28, 20, 110)], bg=60)).blobs
    assert len(blobs) == 2, f"ควรเจอทั้งสองก้อน เจอ {len(blobs)}"
    strong, faint = sorted(blobs, key=lambda b: -b.mean_intensity)
    assert faint.detection_confidence < strong.detection_confidence


# ---------------------------------------------------------------- ภาพผิดชนิด
# 🔴 กลุ่มนี้เพิ่ม 2026-08-16 หลังยิงภาพถ่ายช้าง RGB เข้าไปแล้วได้ "0 ตัว" กลับมาเงียบๆ
# ระบบเตือนช้างที่ตอบว่าไม่มีช้าง ทั้งที่มีช้างสองตัวเต็มเฟรม คือความล้มเหลวที่แย่ที่สุด
# ที่มันมีได้ และแย่กว่านั้นคือ **ไม่มี error ให้ใครเห็น**

def make_daylight_photo(size=(320, 256)):
    """ภาพถ่ายกลางวันจำลอง: ฟ้าสว่างด้านบน ทุ่งกลาง วัตถุมืดด้านหน้า

    ประเด็นทั้งหมดอยู่ตรงที่ **วัตถุมืดกว่าพื้นหลัง** ซึ่งกลับด้านกับสมมติฐานของทั้งท่อ
    (ร้อน = สว่าง) ไม่ได้พยายามให้เหมือนรูปถ่ายจริง แค่ให้สถิติเป็นแบบเดียวกัน
    """
    w, h = size
    a = np.zeros((h, w), np.float32)
    a[: h // 3] = 200.0        # ท้องฟ้ากับก้อนเมฆ สว่างสุดในเฟรม
    a[h // 3 : 2 * h // 3] = 120.0
    a[2 * h // 3 :] = 90.0
    im = Image.fromarray(a.astype(np.uint8))
    d = ImageDraw.Draw(im)
    d.ellipse([w // 4, h // 4, w // 4 + 90, h // 4 + 110], fill=45)   # ช้าง มืด
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def test_daylight_photo_is_flagged_not_silently_empty():
    """ภาพถ่ายทั่วไปต้องถูกตีธง ไม่ใช่ตอบ "ไม่เจอสัตว์" เฉยๆ

    ก่อนแก้: blobs=0 · looks_thermal ไม่มีอยู่ · ปลายทางแยกไม่ออกจากเฟรมว่างจริง
    """
    r = thermal.analyze(make_daylight_photo())
    assert r.looks_thermal is False, \
        f"ภาพถ่ายกลางวันต้องถูกตีธง (mad={r.frame_mad} thr={r.threshold_used})"
    assert r.plausibility_reason, "ตีธงแล้วต้องบอกเหตุผลพร้อมตัวเลขด้วย"
    assert f"mad={r.frame_mad:.3f}" in r.plausibility_reason, \
        f"เหตุผลต้องมีตัวเลขที่วัดได้ ไม่ใช่ข้อความลอยๆ: {r.plausibility_reason}"


def test_real_rgb_elephant_photo_is_flagged():
    """ภาพจริงที่ Toy ส่งมาทดสอบ 2026-08-16 · ช้างสองตัวเต็มเฟรม ตอบ 0 ตัว

    ข้ามเทสต์ถ้าไม่มีไฟล์ เพื่อให้ชุดเทสต์ยังรันได้บนเครื่องที่ clone repo เปล่าๆ
    """
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "1786846183763.jpg")
    if not os.path.exists(p):
        return
    r = thermal.analyze(open(p, "rb").read())
    assert r.blobs == [], "ภาพนี้ท่อ thermal หาก้อนไม่เจอ นั่นคือข้อเท็จจริงที่ยอมรับ"
    assert r.looks_thermal is False, "แต่ต้องบอกว่าอ่านไม่เป็น ไม่ใช่บอกว่าไม่มีสัตว์"


def test_normal_thermal_frames_are_never_flagged():
    """🔴 ด้านกลับ · ธงนี้ห้ามไปโดนเฟรมปกติเข้า

    ธงเตือนที่ดังผิด แย่กว่าไม่มีธง เพราะปลายทางจะเรียนรู้ที่จะเมินมัน
    ครอบทุกเฟรมที่เทสต์อื่นในไฟล์นี้ใช้ ทั้งที่เจอก้อนและไม่เจอ
    """
    frames = {
        "เฟรมว่างมี noise": make_frame([]),
        "เฟรมแบนสนิท": make_frame([], noise=0),
        "สามตัว": make_frame([(120, 150, 26, 18, 220), (330, 240, 30, 20, 230),
                              (500, 380, 24, 16, 215)]),
        "ตัวเล็กบนพื้นเรียบ": make_frame([(320, 256, 8, 6, 235)], noise=0),
        "แดดส่องพื้นครึ่งเฟรม": make_frame([(320, 500, 900, 300, 220)]),
        "ก้อนใหญ่เกินเกณฑ์": make_frame([(320, 256, 300, 200, 230)]),
        "16-bit": make_frame([(150, 150, 26, 18, 220)], depth=16),
        "จางกับชัดในเฟรมเดียว": make_frame([(180, 200, 28, 20, 250),
                                            (460, 320, 28, 20, 110)], bg=60),
    }
    for name, f in frames.items():
        r = thermal.analyze(f)
        assert r.looks_thermal is True, \
            f"'{name}' ไม่ควรโดนธง: mad={r.frame_mad} thr={r.threshold_used} — {r.plausibility_reason}"


def test_flag_never_fires_when_blobs_were_found():
    """เจอก้อนแล้ว = ท่อทำงานได้กับภาพใบนี้จริง ห้ามเดาทับว่ามันไม่ควรทำงาน

    นี่คือสิ่งที่กันไม่ให้ธงนี้กลายเป็นช่องโหว่ใหม่ที่โยนเฟรมถูกต้องทิ้ง
    ต่อให้สถิติหน้าตาแปลกแค่ไหน ถ้านับได้ก็คือนับได้
    """
    assert thermal.plausibility(1, mad=0.9, thr=5.0) == (True, "")
    assert thermal.plausibility(0, mad=0.9, thr=0.5)[0] is False


def test_high_threshold_alone_is_not_evidence():
    """🔴 regression · เคยตีธงเมื่อ thr > 1.0 แล้วเทสต์ข้างบนตีตก

    เฟรม thermal จริงที่แดดส่องพื้นครึ่งล่างได้ thr=1.037 พร้อม mad=0.009
    ซึ่งคือเฟรมปกติที่ควรตอบ "ไม่มีสัตว์" ไม่ใช่เฟรมที่อ่านไม่เป็น
    เกณฑ์เหลือ MAD อย่างเดียว thr เป็นแค่ข้อมูลประกอบในข้อความ
    """
    assert thermal.plausibility(0, mad=0.01, thr=1.5)[0] is True


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
