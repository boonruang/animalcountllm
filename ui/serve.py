"""เสิร์ฟหน้า verify ตอน dev · stdlib ล้วน ไม่มี dependency

🔴 ทำไมต้องมีไฟล์นี้ ไม่เปิด index.html ตรงๆ

`app/main.py` ไม่ได้ใส่ CORS middleware และเราจะไม่ไปแตะมัน หน้าเว็บที่เปิดจาก
`file://` หรือคนละพอร์ต ยิง fetch ข้าม origin ไปหา API แล้วเบราว์เซอร์บล็อกทิ้ง
โดยที่ตัว API ตอบ 200 ปกติ ไล่บั๊กเสียเวลาเปล่า

ตัวนี้เลยเสิร์ฟทั้ง `index.html` และ **proxy `/v1/*` กับ `/healthz` ไปที่ API**
เบราว์เซอร์เห็นเป็น origin เดียว จบเรื่อง CORS โดยไม่แก้ฝั่ง API สักบรรทัด
บน prod ไม่ได้ใช้ไฟล์นี้ App Platform เสิร์ฟ `ui/` เป็น static site
ที่ path `/verify` ของโดเมนเดียวกับ API อยู่แล้ว (ดู .do/app.yaml)

    python ui/serve.py                       # API ที่ 127.0.0.1:8080
    python ui/serve.py --api http://1.2.3.4  # ชี้ไปที่อื่น
    python ui/serve.py --port 3000
"""
from __future__ import annotations

import argparse
import os
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
PROXY_PREFIXES = ("/v1/", "/healthz", "/docs", "/openapi.json")

# X-API-Key ต้องผ่านไปถึง API ไม่งั้น auth พังทั้งที่หน้าเว็บส่งมาถูก
FORWARD_HEADERS = ("content-type", "x-api-key", "accept")


class Handler(SimpleHTTPRequestHandler):
    api = "http://127.0.0.1:8080"

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=HERE, **kw)

    def log_message(self, fmt, *args):
        print(f"{self.command} {self.path} -> {args[1] if len(args) > 1 else ''}")

    def _proxied(self) -> bool:
        return self.path.startswith(PROXY_PREFIXES)

    def do_GET(self):
        if self._proxied():
            return self._forward()
        super().do_GET()

    def do_POST(self):
        if self._proxied():
            return self._forward()
        self.send_error(404)

    def _forward(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        req = urllib.request.Request(self.api + self.path, data=body,
                                     method=self.command)
        for h in FORWARD_HEADERS:
            if self.headers.get(h):
                req.add_header(h, self.headers[h])
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                payload, status, ctype = r.read(), r.status, r.headers.get("Content-Type")
        except urllib.error.HTTPError as e:
            # ต้องส่ง 401/400 ของจริงกลับไป ไม่ใช่กลืนเป็น 502
            # ไม่งั้นหน้าเว็บจะบอกว่า "ยิงไม่ถึงเซิร์ฟเวอร์" ทั้งที่แค่คีย์ผิด
            payload, status, ctype = e.read(), e.code, e.headers.get("Content-Type")
        except urllib.error.URLError as e:
            payload = f'{{"detail":"proxy: ต่อ {self.api} ไม่ได้ ({e.reason})"}}'.encode()
            status, ctype = 502, "application/json"
        self.send_response(status)
        self.send_header("Content-Type", ctype or "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=3000)
    ap.add_argument("--api", default="http://127.0.0.1:8080")
    args = ap.parse_args()
    Handler.api = args.api.rstrip("/")
    print(f"UI  http://127.0.0.1:{args.port}/\nAPI {Handler.api} (proxy)")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
