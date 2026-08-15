"""ชั้นกรองตามเวลา — Hampel (median+MAD) + hysteresis + corroboration

ทั้งหมดในไฟล์นี้เป็นฟังก์ชันบริสุทธิ์ ไม่แตะ DB ไม่แตะเน็ต
รับ 'ประวัติ' เข้ามาเป็น argument แล้วคืนคำตัดสิน เทสต์ได้โดยไม่ต้องมีอะไรเลย

🔴 อ่านก่อนแก้: filter นี้ **ไม่สมมาตรโดยตั้งใจ**
ระบบเตือนภัยที่กรองค่ากระโดดออก จะกรองเหตุการณ์ที่มันมีไว้เพื่อจับทิ้งไปด้วย
ช้างเดินเข้าเฟรม = จำนวนกระโดด 0 → 3 นั่นแหละคือของจริง
ต้นทุนเตือนผิด = คนรำคาญ · ต้นทุนไม่เตือน = ช้างถึงตัวคน
สองอย่างนี้ไม่เท่ากัน filter ก็ไม่ควรเท่ากัน
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..schemas import FilteredResult, WindowInfo

WINDOW_SECONDS = 100
EXPECTED_INTERVAL_S = 10          # กล้องส่งทุก 10 วิ
EXPECTED_FRAMES = WINDOW_SECONDS // EXPECTED_INTERVAL_S   # = 10

CONF_COMMIT = 0.60                # ต่ำกว่านี้ต้องรอเฟรมถัดไปยืนยัน (มติ Toy 2026-08-16)
MAX_PROVISIONAL_FRAMES = 2        # รอได้สูงสุด 2 เฟรม = 20 วิ = ช้างเดินไป ~28 ม.
HAMPEL_K = 3.0                    # ห่างจาก median เกิน k×MAD = ผิดปกติ
RISE_CONFIRM_FRAMES = 2           # ขาขึ้น: ยืนยัน 2 เฟรมพอ (fast attack)
FALL_CONFIRM_FRAMES = 4           # ขาลง: ต้องนิ่ง 4 เฟรม (slow release)


@dataclass
class HistoryItem:
    """หนึ่งเฟรมในหน้าต่าง มาจาก DB ชั้นนี้ไม่รู้จัก DB"""

    ts: float                     # epoch seconds
    counts: Dict[str, int]        # ค่าดิบ ไม่ใช่ค่าที่กรองแล้ว
    confidence: Dict[str, float]  # species -> confidence สูงสุดของเฟรมนั้น


def window_slice(history: List[HistoryItem], now: float) -> List[HistoryItem]:
    """เอาเฉพาะที่อยู่ใน 100 วิล่าสุด

    คิดเป็นเวลา ไม่ใช่จำนวนเฟรม ปกติ 100 วิ = 10 เฟรมพอดี
    แต่ถ้ากล้องหลุดไป 3 นาทีแล้วกลับมา '10 เฟรมล่าสุด' จะกลายเป็นข้อมูลข้าม 4 นาที
    แล้วเอาอดีตมาตัดสินปัจจุบัน ซึ่งผิด
    """
    return [h for h in history if now - h.ts <= WINDOW_SECONDS]


def median_mad(values: List[float]) -> Tuple[float, float]:
    """median กับ MAD ไม่ใช่ mean กับ SD

    ค่าเฉลี่ยถูกลากด้วยค่าผิดปกติที่เราพยายามจะตัดพอดี เฟรมเดียวที่ตอบว่าเห็นช้าง 40 ตัว
    จะทำให้ค่าเฉลี่ยของหน้าต่างเสียไปอีก 10 เฟรม ส่วน median ไม่ขยับเลย
    นี่คือเหตุผลทั้งหมดที่ Hampel ใช้ median
    """
    if not values:
        return 0.0, 0.0
    med = statistics.median(values)
    mad = statistics.median([abs(v - med) for v in values])
    return float(med), float(mad)


def pooled_confidence(confs: List[float]) -> float:
    """รวม confidence จากหลายเฟรมที่เห็นตรงกัน

    🔴 ห้ามใช้ 1 - Π(1-c) สูตรนั้นตั้งบนสมมติฐานว่าแต่ละครั้งเป็นอิสระต่อกัน ซึ่งไม่จริง:
    เฟรมห่าง 10 วิ เป็นสัตว์ตัวเดิม มุมเดิม โมเดลเดิม พลาดรอบแรกก็พลาดแบบเดิมรอบสอง
    สูตรนั้นจะยก 0.52+0.55 เป็น 0.78 ซึ่งเป็นตัวเลขปลอบใจตัวเอง

    ที่ใช้: median + 0.05 ต่อเฟรมที่ยืนยัน เพดาน 0.85
    **เป็น heuristic ไม่ใช่ความน่าจะเป็นที่ calibrate แล้ว** พอมี ground truth จาก
    /truth พอสมควรค่อยแก้ด้วยข้อมูลจริง จนกว่าจะถึงตอนนั้นห้ามอ้างว่าเป็นความน่าจะเป็น
    """
    if not confs:
        return 0.0
    base = statistics.median(confs)
    return round(min(0.85, base + 0.05 * (len(confs) - 1)), 3)


def _direction(current: int, med: float) -> str:
    if current > med:
        return "rising"
    if current < med:
        return "falling"
    return "stable"


def decide(
    raw_counts: Dict[str, int],
    raw_conf: Dict[str, float],
    history: List[HistoryItem],
    now: float,
    provisional_streak: int = 0,
) -> Tuple[FilteredResult, WindowInfo]:
    """คำตัดสินทั้งหมดอยู่ที่นี่ ฟังก์ชันเดียว ไม่มี state ซ่อน

    provisional_streak = จำนวนเฟรมติดกันก่อนหน้าที่ยังไม่ commit
    """
    win = window_slice(history, now)
    species = sorted(set(raw_counts) | {s for h in win for s in h.counts})

    med_map: Dict[str, float] = {}
    mad_map: Dict[str, float] = {}
    for s in species:
        series = [float(h.counts.get(s, 0)) for h in win]
        med_map[s], mad_map[s] = median_mad(series)

    info = WindowInfo(
        span_seconds=WINDOW_SECONDS,
        frames_used=len(win),
        frames_expected=EXPECTED_FRAMES,
        complete=len(win) >= EXPECTED_FRAMES,
        median={k: round(v, 2) for k, v in med_map.items()},
        mad={k: round(v, 2) for k, v in mad_map.items()},
    )

    # ---- หน้าต่างว่าง: หลัง deploy ทุกครั้ง เพราะ SQLite หายไปกับ container (ทาง ค.)
    # ไม่ใช่บั๊ก ต้องบอกตรงๆ ไม่ใช่แกล้งกรอง
    if not win:
        top = max(raw_conf.values(), default=0.0)
        return (
            FilteredResult(
                counts=dict(raw_counts),
                confidence=round(top, 3),
                method="none",
                accepted=True,
                reason="cold start, window empty",
                state="cold_start",
                corroborated_frames=1,
            ),
            info,
        )

    # ---- confidence ต่ำ: รอเฟรมถัดไปยืนยัน แต่ตอบกลับทันที ไม่ค้าง HTTP request
    weak = {s: c for s, c in raw_conf.items() if c < CONF_COMMIT and raw_counts.get(s, 0) > 0}
    if weak and provisional_streak < MAX_PROVISIONAL_FRAMES:
        agree = [
            h.confidence[s]
            for s in weak
            for h in win[-MAX_PROVISIONAL_FRAMES:]
            if h.counts.get(s, 0) == raw_counts.get(s, 0) and s in h.confidence
        ]
        if len(agree) + 1 >= 2:
            pooled = pooled_confidence(agree + list(weak.values()))
            return (
                FilteredResult(
                    counts=dict(raw_counts),
                    confidence=pooled,
                    method="corroboration",
                    accepted=True,
                    reason=f"low confidence confirmed by {len(agree)} earlier frame(s)",
                    state=_direction(sum(raw_counts.values()), sum(med_map.values())),
                    corroborated_frames=len(agree) + 1,
                ),
                info,
            )
        return (
            FilteredResult(
                counts=dict(raw_counts),
                confidence=round(max(weak.values()), 3),
                method="corroboration",
                accepted=False,
                reason=f"confidence {max(weak.values()):.2f} < {CONF_COMMIT}, awaiting next frame",
                state="unconfirmed",
                corroborated_frames=1,
            ),
            info,
        )

    # ---- Hampel + hysteresis แบบไม่สมมาตร
    out: Dict[str, int] = {}
    reasons: List[str] = []
    accepted = True
    state = "stable"

    for s in species:
        cur = raw_counts.get(s, 0)
        med, mad = med_map[s], mad_map[s]
        scale = mad * 1.4826 if mad > 0 else 0.5   # หน้าต่างนิ่งสนิท MAD=0 ต้องมีพื้น
        deviation = abs(cur - med)
        direction = _direction(cur, med)

        if deviation <= HAMPEL_K * scale:
            out[s] = cur
            continue

        recent = [h.counts.get(s, 0) for h in win[-FALL_CONFIRM_FRAMES:]]
        if direction == "rising":
            # fast attack: 2 เฟรมยืนยันพอ ปล่อยผ่านเลย
            support = sum(1 for v in recent[-RISE_CONFIRM_FRAMES:] if v >= cur)
            out[s] = cur
            state = "rising"
            reasons.append(
                f"{s}: jump {med:.1f}->{cur} accepted (rising, {support + 1} frame(s))"
            )
        else:
            # slow release: ต้องนิ่งหลายเฟรมก่อนยอมรับว่าหายไปแล้ว
            steady = sum(1 for v in recent if v <= cur)
            if steady >= FALL_CONFIRM_FRAMES:
                out[s] = cur
                state = "falling"
                reasons.append(f"{s}: drop {med:.1f}->{cur} accepted after {steady} steady frames")
            else:
                out[s] = int(round(med))
                accepted = False
                state = "falling"
                reasons.append(
                    f"{s}: drop {med:.1f}->{cur} held, only {steady}/{FALL_CONFIRM_FRAMES} steady"
                )

    out = {k: v for k, v in out.items() if v > 0}
    confs = [c for s, c in raw_conf.items() if out.get(s, 0) > 0]

    return (
        FilteredResult(
            counts=out,
            confidence=round(statistics.median(confs), 3) if confs else 0.0,
            method="hampel_median_mad + hysteresis",
            accepted=accepted,
            reason="; ".join(reasons) or f"within {HAMPEL_K} MAD of window median",
            state=state,
            corroborated_frames=1,
        ),
        info,
    )
