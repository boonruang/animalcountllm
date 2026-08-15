"""แปลงไฟล์ภาพเป็น body พร้อมยิง

    python tools/make_payload.py frame.png cam-01 > payload.json

แล้วใน Postman เลือก Body -> raw -> JSON แล้ววางเนื้อไฟล์ หรือใช้กับ curl:
    curl -X POST $URL/v1/frames -H "Content-Type: application/json" -d @payload.json
"""
import base64, json, sys
from datetime import datetime, timezone

if len(sys.argv) < 2:
    sys.exit(__doc__)
path = sys.argv[1]
camera_id = sys.argv[2] if len(sys.argv) > 2 else "cam-01"
raw = open(path, "rb").read()
print(json.dumps({
    "camera_id": camera_id,
    "image_base64": base64.b64encode(raw).decode(),
    "ts": datetime.now(timezone.utc).astimezone().isoformat(),
    "note": f"from {path}",
}, ensure_ascii=False))
