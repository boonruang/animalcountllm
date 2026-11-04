"""เทสต์การขยายภาพก่อนส่งเข้าโมเดล · รันได้โดยไม่มีเน็ต ไม่มีโมเดล"""
import base64
import io

from PIL import Image

from app import main


def _jpeg(w, h, mode="RGB"):
    buf = io.BytesIO()
    Image.new(mode, (w, h), 128).save(buf, format="JPEG")
    return buf.getvalue()


def test_ภาพเล็กถูกขยายจนด้านสั้นถึงเกณฑ์():
    b64, w, h = main.upscale_for_model(_jpeg(640, 480))
    assert (w, h) == (1280, 960)
    with Image.open(io.BytesIO(base64.b64decode(b64))) as im:
        assert im.size == (1280, 960)


def test_ภาพที่ถึงเกณฑ์แล้วไม่ถูกแตะ():
    assert main.upscale_for_model(_jpeg(1280, 960)) is None


def test_ด้านยาวไม่เกินเพดาน():
    _, w, h = main.upscale_for_model(_jpeg(1600, 400))
    assert w == 2000
    assert h == 500


def test_ภาพเทาแบบ_mode_L_ขยายได้():
    assert main.upscale_for_model(_jpeg(640, 480, "L"))[1:] == (1280, 960)


def test_ปิดได้ด้วยค่า_0(monkeypatch):
    monkeypatch.setattr(main, "MIN_IMAGE_PX", 0)
    assert main.upscale_for_model(_jpeg(640, 480)) is None
