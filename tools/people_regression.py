"""ยิงภาพชุดเดิมซ้ำหลังแก้ prompt แล้วเทียบผลเป็นตาราง

    python tools/people_regression.py inbox/iplus_sample1.jpg inbox/iplus_sample2.jpg
    python tools/people_regression.py --url https://xxx.ondigitalocean.app --key $KEY *.jpg
    python tools/people_regression.py --save before.json ...      # เก็บผลรอบนี้ไว้
    python tools/people_regression.py --against before.json ...   # เทียบกับรอบก่อน

🔴 มีเครื่องมือนี้เพราะกฎของโปรเจคเองข้อหนึ่ง:
**ห้ามเปลี่ยน prompt แล้วไม่ยิงภาพชุดเดิมซ้ำ**

วัดมาแล้ว 2026-09-11: แค่เพิ่มคำถามเรื่องทิศเข้า/ออก จำนวนคนที่เจอใน iplus_sample1
ขยับจาก 2 เป็น 3 ทั้งที่ภาพใบเดิม · prompt f2 (2026-09-12) แตะกฎเรื่อง
"ใครนับเป็นคน" ตรงๆ (overlay ที่ฝังมาในภาพ · คนนอกประตูกระจก) ซึ่งจะทำให้
ตัวเลขขยับอีกแน่นอน **ถ้าไม่มีตัวเลขรอบก่อนเก็บไว้ ก็ไม่มีทางรู้ว่าดีขึ้นหรือแย่ลง**

⚠️ เครื่องมือนี้ **ไม่ตัดสินว่าใครถูก** มันบอกแค่ว่าอะไรเปลี่ยน คำตอบว่า "3 คน
หรือ 8 คนถูก" อยู่ที่ทีมคุณสุชาติว่านับใครเป็นคนในเฟรม ไม่ใช่ที่เราเดาเอง
ของจริงต้องมาจาก POST /people/v1/persons/{rid}/{oid}/truth
"""
import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request


def post(url: str, key: str, path: str, timeout: float) -> dict:
    with open(path, "rb") as f:
        body = json.dumps({"camera_id": "regression",
                           "image_base64": base64.b64encode(f.read()).decode(),
                           "note": f"regression:{os.path.basename(path)}"}).encode()
    req = urllib.request.Request(f"{url.rstrip('/')}/people/v1/frames", data=body,
                                 headers={"Content-Type": "application/json",
                                          "X-API-Key": key})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"status": "error", "reason": f"HTTP {e.code}: {e.read()[:200]!r}",
                "persons": [], "summary": {}}
    except OSError as e:
        return {"status": "error", "reason": f"{type(e).__name__}: {e}",
                "persons": [], "summary": {}}


def digest(d: dict) -> dict:
    """ย่อคำตอบเหลือเฉพาะตัวเลขที่ต้องเฝ้า · ไม่เก็บคำบรรยาย ไม่เก็บภาพ"""
    s = d.get("summary", {})
    return {"status": d.get("status"), "reason": d.get("reason", ""),
            "people": s.get("people_found", len(d.get("persons", []))),
            **{k: s.get(k, 0) for k in
               ("direction_in", "direction_out", "direction_unknown",
                "age_child", "age_teen", "age_adult", "age_senior", "age_unknown",
                "gender_male", "gender_female", "gender_unknown",
                "in_uniform", "uniform_unknown", "groups_found",
                "emotion_positive", "emotion_neutral", "emotion_negative",
                "emotion_unknown")},
            "uniform_text": sorted({p["appearance"]["uniform"]["text"]
                                    for p in d.get("persons", [])
                                    if p.get("appearance", {}).get("uniform", {}).get("text")}),
            "where": [p.get("where", "")[:60] for p in d.get("persons", [])]}


def main() -> None:
    ap = argparse.ArgumentParser(usage=__doc__)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--url", default=os.environ.get("PEOPLE_URL",
                                                    "http://127.0.0.1:8080"))
    ap.add_argument("--key", default=os.environ.get("PEOPLE_API_KEY", ""))
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--save", help="เขียนผลย่อของรอบนี้ลงไฟล์")
    ap.add_argument("--against", help="ไฟล์ผลรอบก่อน เอามาเทียบ")
    args = ap.parse_args()

    before = {}
    if args.against:
        with open(args.against, encoding="utf-8") as f:
            before = json.load(f)

    now = {}
    for path in args.images:
        name = os.path.basename(path)
        print(f"\n=== {name} ===", flush=True)
        cur = digest(post(args.url, args.key, path, args.timeout))
        now[name] = cur
        old = before.get(name)
        for k, v in cur.items():
            if k == "where":
                continue
            if old is None:
                print(f"  {k:20} {v}")
            elif old.get(k) != v:
                # 🔴 เน้นเฉพาะที่เปลี่ยน · ตารางที่พิมพ์ทุกบรรทัดเท่ากันหมด
                # คือตารางที่ไม่มีใครอ่าน แล้วความเปลี่ยนแปลงจริงจะจมหายไป
                print(f"  {k:20} {old.get(k)}  ->  {v}   <-- เปลี่ยน")
            else:
                print(f"  {k:20} {v}")
        if cur["where"]:
            print("  ตำแหน่งที่โมเดลบอก:")
            for w in cur["where"]:
                print(f"    · {w}")

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(now, f, ensure_ascii=False, indent=2)
        print(f"\nเก็บผลรอบนี้ไว้ที่ {args.save} แล้ว")

    if before:
        print("\n⚠️ ตัวเลขที่ขยับ ไม่ได้แปลว่าดีขึ้นหรือแย่ลง มันแปลว่า 'เปลี่ยน'"
              "\n   ใครถูกต้องมาจาก /truth ที่คนตรวจแล้วเท่านั้น")


if __name__ == "__main__":
    main()
