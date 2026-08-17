# API spec สำหรับยิงทดสอบ (Postman / curl)

**Base URL (prod):** `https://seal-app-qv5ao.ondigitalocean.app`
**Base URL (local):** `http://127.0.0.1:8080`

หน้าเอกสารอัตโนมัติของ FastAPI: `{base}/docs` กดยิงจากหน้านั้นได้เลยโดยไม่ต้องใช้ Postman

## Header ทุก request

| Header | ค่า | จำเป็นไหม |
|---|---|---|
| `Content-Type` | `application/json` | ใช่ ทุก POST |
| `X-API-Key` | ค่าเดียวกับ env `API_KEY` | เฉพาะตอนที่ตั้ง `API_KEY` ไว้ · ว่าง = ไม่ต้องส่ง |

`API_KEY` ว่าง = **ใครก็ยิงได้ และทุกครั้งที่ยิงคือเงินค่า LLM ของเจ้าของ app**

---

## 0. `GET /verify` — หน้าเว็บทดสอบ

เปิด `{base}/verify` ในเบราว์เซอร์ เลือกภาพจากเครื่อง ใส่ `X-API-Key` ในช่องบนหน้า
กด "ตรวจภาพ" แล้วดูผลที่โมเดลตอบ พร้อม JSON เต็ม ไม่ต้องแปลง base64 เอง ไม่ต้องใช้ Postman

ช่อง `X-API-Key` ไม่มีค่าเริ่มต้น ว่างไว้ได้ถ้าฝั่งเซิร์ฟเวอร์ไม่ได้ตั้ง `API_KEY`
ภาพอยู่ในแท็บเบราว์เซอร์เท่านั้น หน้าเว็บไม่เก็บอะไรลงเครื่อง
ฝั่งเซิร์ฟเวอร์ยังทำตาม env `SAVE_IMAGES` ตามเดิม ตั้งเป็น `none` ถ้าไม่อยากให้เก็บภาพไว้เลย

**หน้านี้ไม่ใช่ endpoint ของ API** เป็น static site คนละคอมโพเนนต์ที่บังเอิญอยู่โดเมนเดียวกัน
โค้ดใน `app/` ไม่รู้จักมัน · ตอน dev รัน `python ui/serve.py` แล้วเปิด `http://127.0.0.1:3000`
(ต้องผ่านตัวนี้ เพราะ API ไม่มี CORS เปิดจาก `file://` ตรงๆ เบราว์เซอร์บล็อก)

---

## 1. `GET /healthz`

เช็คว่าแอปขึ้นและต่อกับอะไรอยู่ ไม่ต้องมี header ไม่มี body

```json
{
  "status": "ok",
  "provider": "openrouter",
  "model": "qwen/qwen3.7-flash",
  "prompt_version": "v1",
  "store": "sqlite",
  "store_path": "/tmp/animalcountllm/animals.db",
  "store_error": null,
  "image_dir": "/tmp/animalcountllm/frames",
  "tracing": true
}
```

`status: "degraded"` + `store_error` = แอปขึ้นแล้วแต่เปิดฐานข้อมูลไม่ได้
**ไม่มี field `store_path` = กำลังรันโค้ดเวอร์ชันเก่าอยู่**

---

## 2. `POST /v1/frames` — เส้นหลัก

### body

```json
{
  "camera_id": "cam-01",
  "image_base64": "iVBORw0KGgoAAAANSUhEUg...",
  "ts": "2026-08-16T06:30:00+07:00",
  "note": "ทดสอบจาก Postman"
}
```

| field | จำเป็น | หมายเหตุ |
|---|---|---|
| `camera_id` | ใช่ | **หน้าต่าง 100 วิ แยกตามค่านี้** กล้องคนละจุดห้ามใช้ค่าเดียวกัน |
| `image_base64` | ใช่ | base64 ของไฟล์ภาพ **ไม่ต้องมี `data:image/png;base64,` นำหน้า** |
| `ts` | ไม่ | ISO 8601 · ไม่ส่ง = ใช้เวลาที่เซิร์ฟเวอร์รับ · ส่งมาผิดรูปก็ไม่ error แค่ใช้เวลาเซิร์ฟเวอร์แทน |
| `note` | ไม่ | ข้อความอิสระ |

### ภาพตัวอย่างพร้อมยิง

อยู่ใน `docs/samples/` เปิดไฟล์ `.b64` แล้วก๊อปทั้งไฟล์ไปวางใน `image_base64` ได้เลย

| ไฟล์ | ขนาด base64 | ควรได้ผลว่า |
|---|---|---|
| `empty.b64` | 1.8 KB | `counts: {}` · `timing_ms.vlm = 0` **ไม่ยิง LLM ไม่เสียเงิน** |
| `one-far.b64` | 2.3 KB | 1 ก้อน 289 px · ทดสอบว่าสัตว์ตัวเล็กไม่ถูกตัดทิ้งเงียบๆ |
| `two-animals.b64` | 4.3 KB | 2 ก้อน · ยิง LLM |
| `three-animals.b64` | 5.0 KB | 3 ก้อน |

### แปลงภาพของตัวเองเป็น base64

PowerShell
```powershell
[Convert]::ToBase64String([IO.File]::ReadAllBytes("C:\path\frame.png")) | Set-Clipboard
```

bash
```bash
base64 -w0 frame.png > frame.b64
```

หรือใช้ตัวช่วยที่เตรียมไว้ สร้าง body ให้ครบทั้งก้อน
```bash
python tools/make_payload.py frame.png cam-01 > payload.json
```

### response

```json
{
  "request_id": "bb24bc8f-4225-4cd9-a4d8-a93a5acb6725",
  "camera_id": "cam-01",
  "received_at": "2026-08-16T06:30:00+00:00",
  "status": "ok",

  "raw": {
    "detections": [
      {"id": 1, "bbox": [135, 165, 61, 41], "area_px": 2321, "aspect": 1.488,
       "species": "elephant", "species_confidence": 0.72,
       "detection_confidence": 0.986, "overall_confidence": 0.71}
    ],
    "counts": {"elephant": 1},
    "overall_confidence": 0.71
  },

  "filtered": {
    "counts": {"elephant": 1},
    "confidence": 0.72,
    "method": "hampel_median_mad + hysteresis",
    "accepted": true,
    "reason": "within 3.0 MAD of window median",
    "state": "stable",
    "corroborated_frames": 1
  },

  "window": {
    "span_seconds": 100, "frames_used": 7, "frames_expected": 10,
    "complete": false, "median": {"elephant": 1}, "mad": {"elephant": 0.0}
  },

  "model": {"provider": "openrouter", "name": "qwen/qwen3.7-flash",
            "prompt_version": "v1", "finish_reason": "stop", "completion_tokens": 375},
  "timing_ms": {"cv": 29.9, "vlm": 4900.0, "total": 4935.2}
}
```

### `status` แปลว่าอะไร

| ค่า | ความหมาย | ปลายทางควรทำอะไร |
|---|---|---|
| `ok` | ตอบได้ครบ | ใช้ `filtered` ได้ |
| `provisional` | ความเชื่อมั่นต่ำกว่า 0.60 **ยังไม่ยืนยัน** รอเฟรมถัดไป | รออีก 10 วิ อย่าเพิ่งทำอะไร |
| `degraded` | **LLM ใช้ไม่ได้ตอนนี้** (ล่ม/ชื่อโมเดลผิด/ตอบไม่จบ) เหลือแต่ผล CV ทุกก้อนเป็น `unknown` | รู้ว่ามีของร้อนกี่ก้อน แต่ไม่รู้ว่าอะไร · ต้องไปแก้ที่ระบบ ไม่ใช่รอ |
| `error` | อ่านภาพไม่ได้ | ตรวจ payload |

`provisional` กับ `degraded` **ห้ามสับสนกัน** อันแรกคือ "รอแป๊บเดี๋ยวก็รู้" อันหลังคือ "ระบบพังอยู่ รอไปก็ไม่รู้"

### `state` แปลว่าอะไร

| ค่า | ความหมาย |
|---|---|
| `stable` | จำนวนตรงกับที่เห็นมาตลอดหน้าต่าง |
| `rising` | เพิ่มขึ้นและระบบ**ยอมรับทันที** (fast attack — สัตว์กำลังเข้ามา) |
| `falling` | ลดลง · ถ้า `accepted: false` แปลว่ายัง**ไม่ยอมประกาศว่าหมด** ต้องนิ่งอีกหลายเฟรม |
| `unstable` | เฟรมติดกันเห็นไม่ตรงกัน ระบบไม่เลือกให้ |
| `unconfirmed` | รอเฟรมถัดไปยืนยัน |
| `cold_start` | **หน้าต่างว่าง** เกิดหลัง deploy ทุกครั้ง เพราะฐานข้อมูลหายไปกับ container · ไม่ใช่บั๊ก |

### ค่าความเชื่อมั่น 2 ตัว อย่าปนกัน

- `detection_confidence` — จากชั้น CV **วัดได้ อธิบายได้ ทำซ้ำได้เป๊ะ**
- `species_confidence` — จาก LLM **โมเดลพ่นออกมาเอง ยังไม่ calibrate** 0.9 ไม่ได้แปลว่าถูก 90%
- `overall_confidence` = คูณกัน ถ่วงน้ำหนักด้วยพื้นที่ก้อน

---

## 3. `GET /v1/cameras/{camera_id}/window`

ดูว่าตอนนี้ filter กำลังตัดสินจากข้อมูลอะไร ใช้ debug เวลาผลไม่เป็นอย่างที่คิด

```json
{"camera_id": "cam-01", "span_seconds": 100, "frames_used": 4,
 "frames_expected": 10, "complete": false, "provisional_streak": 0,
 "items": [{"age_s": 32.1, "counts": {"elephant": 2}, "confidence": {"elephant": 0.71}}]}
```

## 4. `GET /v1/frames/{request_id}`

ย้อนดูเฟรมเดียว ได้ผล CV ดิบ + response ดิบของ LLM + ground truth (ถ้ามี)

## 5. `POST /v1/frames/{request_id}/truth`

บันทึกคำตอบที่คนตรวจแล้ว

```json
{"counts": {"elephant": 2}, "reviewer": "toy", "comment": "ดูจากภาพจริงแล้ว"}
```

**ไม่มีข้อมูลนี้ = calibrate `species_confidence` ไม่ได้ตลอดกาล** และย้อนไปเก็บทีหลังไม่ได้

## 6. `POST /v1/maintenance/rollup`

สรุปวันละแถวแล้วลบของดิบที่เกิน `RETENTION_DAYS` · เรียกจาก cron ข้างนอก

---

## ลำดับการทดสอบที่แนะนำ

ยิงตามลำดับนี้ด้วย `camera_id` เดียวกัน จะเห็นพฤติกรรมของ filter ครบ

1. `empty.b64` → `cold_start` · vlm 0 ms
2. `empty.b64` อีก 2 ครั้ง → `stable` · window โต 1, 2
3. `two-animals.b64` → **`rising`** พร้อม reason `jump 0.0->2 accepted`
   นี่คือจุดที่ filter ทั่วไปจะตัดทิ้ง แต่ของเราต้องปล่อยผ่าน
4. `empty.b64` → **`falling` แต่ `accepted: false`** ยังไม่ประกาศว่าหมด
5. `empty.b64` ซ้ำอีก 3-4 ครั้ง → ถึงจะยอมรับว่าหมดจริง

ถ้าข้อ 3 ไม่ขึ้น `rising` แปลว่ามีคนไปแก้ filter ให้สมมาตร ซึ่งจะทำให้ระบบเงียบตอนช้างมา

## curl

```bash
curl -s -X POST https://seal-app-qv5ao.ondigitalocean.app/v1/frames \
  -H "Content-Type: application/json" \
  -d "{\"camera_id\":\"cam-01\",\"image_base64\":\"$(cat docs/samples/two-animals.b64)\"}"
```
