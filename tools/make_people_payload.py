"""แปลงไฟล์ภาพเป็น body ของ POST /people/v1/persons

    # ครอปมาแล้วไฟล์ละคน
    python tools/make_people_payload.py --cam N3-1C23A a.jpg b.jpg > payload.json

    # เฟรมเต็ม + กรอบ (x,y,w,h ในพิกเซลของเฟรมนั้น)
    python tools/make_people_payload.py --frame lobby.jpg --box 011:795,95,180,515 \
                                        --box 021:885,315,225,660 > payload.json

    curl -X POST $URL/people/v1/persons -H "Content-Type: application/json" \
         -H "X-API-Key: $KEY" -d @payload.json

id ของคนมาจากชื่อไฟล์ หรือจากหน้า --box · ปลายทางจริงจะส่ง track id ของเขามาเอง
"""
import argparse
import base64
import json
import os
import sys
from datetime import datetime, timezone


def b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def main() -> None:
    ap = argparse.ArgumentParser(usage=__doc__)
    ap.add_argument("images", nargs="*", help="ภาพที่ครอปมาแล้ว ไฟล์ละคน")
    ap.add_argument("--frame", help="เฟรมเต็ม ใช้คู่กับ --box")
    ap.add_argument("--box", action="append", default=[],
                    help="id:x,y,w,h · ใส่ได้หลายครั้ง")
    ap.add_argument("--cam", default="cam-01")
    args = ap.parse_args()

    if not args.images and not args.box:
        sys.exit(__doc__)

    body = {"camera_id": args.cam,
            "ts": datetime.now(timezone.utc).astimezone().isoformat(),
            "objects": []}

    for p in args.images:
        oid = os.path.splitext(os.path.basename(p))[0][:64]
        body["objects"].append({"id": oid, "image_base64": b64(p)})

    if args.box:
        if not args.frame:
            sys.exit("🔴 --box ต้องมี --frame ด้วย ไม่งั้นไม่รู้ว่ากรอบอยู่บนภาพไหน")
        body["frame_base64"] = b64(args.frame)
        for spec in args.box:
            try:
                oid, nums = spec.split(":", 1)
                x, y, w, h = (int(v) for v in nums.split(","))
            except ValueError:
                sys.exit(f"🔴 รูปแบบ --box ผิด: {spec!r} · ต้องเป็น id:x,y,w,h")
            body["objects"].append({"id": oid, "bbox": [x, y, w, h]})

    if len(body["objects"]) > 16:
        sys.exit(f"🔴 {len(body['objects'])} คน เกินเพดาน 16 คนต่อ request")

    print(json.dumps(body, ensure_ascii=False))


if __name__ == "__main__":
    main()
