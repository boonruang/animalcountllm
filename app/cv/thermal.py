"""ชั้น CV — Computer Vision แบบดั้งเดิม ไม่มีโมเดล AI ไม่มี GPU ไม่ต้องเทรน

ทำสี่อย่าง: normalize → threshold → นับก้อนที่ติดกัน → วัดแต่ละก้อน

หน้าที่ของชั้นนี้คือตอบว่า "มีกี่ก้อน อยู่ตรงไหน ใหญ่แค่ไหน"
ไม่ใช่ "ก้อนนั้นคือสัตว์อะไร" (นั่นเป็นงานของ LLM)

จำนวนมาจากที่นี่ ไม่ใช่จาก LLM เพราะที่นี่รันซ้ำได้ผลเดิมเป๊ะ ส่วน LLM ไม่
"""
from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage

from ..schemas import Blob, CVResult

# ---- พารามิเตอร์ที่ปรับได้ ค่าเริ่มต้นตั้งไว้กว้างเพราะยังไม่เคยเห็นภาพจากไซต์จริง
import os

# 🔴 ค่านี้คือ "สัตว์ที่เล็กที่สุดที่เรากล้าบอกว่าตรวจเจอ" ไม่ใช่ค่ากัน noise เฉยๆ
#
# ปัญหาคือมันแลกกันตรงๆ: ตั้งต่ำ = นกกับพิกเซลเสียหลุดไปถึง LLM (เสียเงิน เตือนผิด)
# ตั้งสูง = ช้างที่อยู่ไกลหลุดหายไปเงียบๆ (พลาดของจริง ซึ่งแพงกว่ามาก)
#
# ช้างที่ระยะ 200 ม. บนเซ็นเซอร์ 640x512 FOV ~25° กินราว 15x10 px = 150 px
# แปลว่า **จุดขนาด 100-150 px แยกจากช้างไกลๆ ไม่ได้จริง** ไม่ว่าจะเขียนโค้ดดีแค่ไหน
#
# ค่าเริ่มต้น 40 = ยอมให้ของกำกวมผ่านขึ้นไปให้ LLM ตัดสิน ดีกว่าตัดทิ้งเงียบๆ ที่ชั้นนี้
# ตัวที่กัน LLM ไม่ให้ถูกยิงรัวคือ MIN_CONTRAST กับ has_candidates ไม่ใช่ค่านี้
#
# ⚠️ ต้องจูนใหม่เมื่อได้ภาพจากไซต์จริง ต้องรู้ FOV ความละเอียด และระยะไกลสุดที่ต้องการ
MIN_AREA_PX = int(os.environ.get("CV_MIN_AREA_PX", "40"))
MAX_AREA_FRAC = 0.35      # ใหญ่กว่านี้ = ไม่ใช่สัตว์ (แดดส่องพื้น เครื่องยนต์ ไฟ)
MIN_CONTRAST = 0.08       # ต่างจากพื้นหลังน้อยกว่านี้ = ไม่มั่นใจว่าเป็นวัตถุ
SIGMA_K = 2.5             # threshold = พื้นหลัง + k × ส่วนเบี่ยงเบน
SMOOTH_SIGMA = 1.0        # กัน noise เกลือพริกไทยของเซ็นเซอร์ thermal


def load_image(raw: bytes) -> Tuple[np.ndarray, int]:
    """อ่านภาพเป็น grayscale float 0-1 พร้อมบอก bit depth เดิม

    รองรับทั้ง radiometric 16-bit (มีค่าอุณหภูมิจริง) และ 8-bit ที่ผ่าน palette สีมาแล้ว
    เพราะยังไม่รู้ว่ากล้องที่ไซต์ส่งแบบไหน (ดู CLAUDE.md หัวข้อ 'ที่ยังไม่รู้')
    ตรวจเอาเองจากไฟล์ ไม่ต้องให้ปลายทางบอก
    """
    import io

    im = Image.open(io.BytesIO(raw))
    depth = 16 if im.mode in ("I;16", "I;16B", "I", "F") else 8

    if im.mode not in ("L", "I;16", "I;16B", "I", "F"):
        # ภาพสีที่ผ่าน palette ความร้อนมาแล้ว: ความสว่างยังสื่อความร้อนอยู่
        im = im.convert("L")

    a = np.asarray(im).astype(np.float32)
    if a.ndim == 3:
        a = a.mean(axis=2)

    hi = 65535.0 if depth == 16 else 255.0
    return a / hi, depth


def normalize(a: np.ndarray) -> np.ndarray:
    """ยืดสเกลด้วย percentile ไม่ใช่ min/max

    กล้อง thermal ปรับ gain เองตลอดเวลา ภาพฉากเดียวกันคนละนาทีค่าดิบไม่เท่ากัน
    ถ้าใช้ min/max พิกเซลเสียจุดเดียวก็ลากทั้งภาพ percentile ทนกว่า
    """
    lo, hi = np.percentile(a, 1.0), np.percentile(a, 99.5)

    # 🔴 บั๊กที่เจอ 2026-08-16 ตอนสร้างภาพตัวอย่างแบบไม่มี noise
    # สัตว์ที่กินพื้นที่ไม่ถึง 0.5% ของเฟรม **อยู่เหนือ percentile 99.5 ทั้งตัว**
    # บนพื้นหลังเรียบๆ p1 กับ p99.5 เลยเท่ากัน ฟังก์ชันคืนภาพดำทั้งใบ แล้วสัตว์หายเกลี้ยง
    # ทดสอบแล้วหายตั้งแต่ 8x6 ไปจนถึง 20x15 พิกเซล ซึ่งคือช่วง "ช้างอยู่ไกล" พอดี
    #
    # ที่ไม่เคยเจอเพราะภาพทดสอบเดิมใส่ noise ไว้ ซึ่งบังเอิญถ่าง p1 กับ p99.5 ออกจากกัน
    # กล้องจริงมี noise เกือบตลอด แต่ "เกือบ" ไม่ใช่ "เสมอ" และ stream ที่ผ่าน denoise
    # มาแรงๆ ก็จะเข้าเคสนี้ · ปล่อยไว้ = ช้างหายเงียบโดยไม่มี error ให้เห็น
    if hi - lo < 1e-3:
        hi = float(a.max())   # ฉากเรียบเกินกว่าที่ percentile จะจับได้ ใช้ค่าสูงสุดแทน
    if hi - lo < 1e-6:
        return np.zeros_like(a)   # เฟรมแบนสนิทจริงๆ ไม่มีอะไรให้ดู
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def segment(a: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """แยกพิกเซลที่ร้อนกว่าพื้นหลัง

    สัตว์เลือดอุ่นสว่างกว่าพื้นดินเสมอ นี่คือข้อได้เปรียบทั้งหมดของกล้องความร้อน
    ใช้ median กับ MAD เป็นฐาน ไม่ใช่ mean กับ SD เพราะถ้าในเฟรมมีสัตว์ตัวใหญ่
    mean จะถูกตัววัตถุเองลากขึ้นไป แล้ว threshold จะสูงเกินจนตัดวัตถุนั้นทิ้ง
    """
    sm = ndimage.gaussian_filter(a, SMOOTH_SIGMA)
    bg = float(np.median(sm))
    mad = float(np.median(np.abs(sm - bg))) or 1e-6
    thr = bg + SIGMA_K * mad * 1.4826  # 1.4826 = แปลง MAD ให้เทียบเท่า SD

    # พื้นล่างของ threshold: ในเฟรมที่แทบว่าง MAD จะเล็กมากจน threshold ต่ำติดพื้น
    # แล้วรัศมีเบลอรอบจุดเล็กๆ (นก แมลง พิกเซลเสีย) จะถูกนับเป็นก้อนใหญ่
    # พิกเซลต้องสว่างกว่าพื้นหลังอย่างน้อย MIN_CONTRAST ถึงจะนับ ใช้เกณฑ์เดียวกับตอนคัดก้อน
    thr = max(thr, bg + MIN_CONTRAST)
    return (sm > thr), bg, thr


def measure(mask: np.ndarray, a: np.ndarray, bg: float) -> List[Blob]:
    """จับพิกเซลสว่างที่ติดกันเป็นก้อน แล้ววัดทีละก้อน

    connected component หนึ่งก้อน = หนึ่งตัว (ก่อนที่ LLM จะบอกให้ยุบ/แยก)
    """
    # morphological opening: กัดขอบแล้วขยายคืน
    # ลบชายเบลอบางๆ รอบจุดเล็ก (นก แมลง พิกเซลเสีย) ที่ threshold เดี่ยวๆ ตัดไม่ออก
    # ก้อนใหญ่ขนาดสัตว์แทบไม่เสียพื้นที่ ก้อนจิ๋วหายไปเลย
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)), iterations=1)

    lab, n = ndimage.label(mask)
    if n == 0:
        return []

    h, w = mask.shape
    max_area = MAX_AREA_FRAC * h * w
    out: List[Blob] = []

    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        region = lab[sl] == i
        area = int(region.sum())
        if area < MIN_AREA_PX or area > max_area:
            continue

        ys, xs = sl
        bw, bh = xs.stop - xs.start, ys.stop - ys.start
        vals = a[sl][region]
        mean_i = float(vals.mean())
        contrast = mean_i - bg
        if contrast < MIN_CONTRAST:
            continue

        touches = (xs.start == 0 or ys.start == 0 or xs.stop >= w or ys.stop >= h)

        out.append(
            Blob(
                id=len(out) + 1,
                bbox=[int(xs.start), int(ys.start), int(bw), int(bh)],
                area_px=area,
                aspect=round(bw / bh, 3) if bh else 0.0,
                fill_ratio=round(area / (bw * bh), 3) if bw * bh else 0.0,
                mean_intensity=round(mean_i, 4),
                contrast=round(contrast, 4),
                touches_edge=bool(touches),
                detection_confidence=score(area, contrast, touches, area / (bw * bh) if bw * bh else 0),
            )
        )
    return out


def score(area: int, contrast: float, touches_edge: bool, fill: float) -> float:
    """detection_confidence — วัดได้ อธิบายได้ทีละพจน์ ไม่ใช่เลขที่โผล่มาเฉยๆ

    ตั้งใจให้ตรงข้ามกับ species_confidence ของ LLM ที่อธิบายที่มาไม่ได้
    """
    # ยิ่งใหญ่ยิ่งมั่นใจ แต่อิ่มตัว ก้อน 5000 px กับ 50000 px ไม่ได้ต่างกัน 10 เท่า
    s_area = min(1.0, np.log1p(area) / np.log1p(3000))
    # ยิ่งต่างจากพื้นหลังยิ่งมั่นใจ
    s_contrast = min(1.0, contrast / 0.35)
    # ก้อนตันน่าเชื่อกว่าก้อนกระจาย (สัตว์เป็นก้อนตัน เงาความร้อนไม่ใช่)
    s_fill = min(1.0, fill / 0.6)
    v = 0.45 * s_area + 0.40 * s_contrast + 0.15 * s_fill
    if touches_edge:
        v *= 0.75  # ติดขอบเฟรม = อาจโดนตัดครึ่ง ขนาดที่วัดได้เชื่อไม่ได้เต็มที่
    return round(float(np.clip(v, 0.0, 1.0)), 3)


def analyze(raw: bytes) -> CVResult:
    """ทางเข้าเดียวของชั้นนี้ · รันภาพเดิมกี่ครั้งก็ได้ผลเดิมเป๊ะ"""
    t0 = time.perf_counter()
    a, depth = load_image(raw)
    norm = normalize(a)
    mask, bg, thr = segment(norm)
    blobs = measure(mask, norm, bg)
    return CVResult(
        blobs=blobs,
        frame_w=int(a.shape[1]),
        frame_h=int(a.shape[0]),
        bit_depth=depth,
        background_level=round(bg, 4),
        threshold_used=round(thr, 4),
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 2),
    )
