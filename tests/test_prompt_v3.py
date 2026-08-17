"""เทสต์เส้น llm · รันได้โดยไม่มีเน็ต ไม่มี numpy ไม่มีโมเดล

สิ่งที่เทสต์นี้เฝ้าคือความผิดพลาดที่เพิ่งทำให้ระบบตอบผิดมาทั้งวัน:
prompt ที่เดินไปบอกโมเดลเองว่า "ตรวจไม่เจออะไร" แล้วโมเดลก็เชื่อ
"""
from app.llm import prompt_v3
from app.llm.client import image_type_of, mime_of, parse


def test_prompt_v3_ไม่บอกโมเดลว่าตรวจเจอหรือไม่เจออะไร():
    """🔴 หัวใจของ v3 · v2 พังเพราะบรรทัด 'detector found 0 warm region(s)'

    โมเดลอ่านแล้วเชื่อว่าเฟรมว่าง ทั้งที่ช้างเต็มเฟรม ถ้าคำพวกนี้กลับเข้ามา
    ในเทมเพลตอีก บั๊กเดิมจะกลับมาทั้งดุ้นโดยไม่มีใครเห็น
    """
    system, user = prompt_v3.build(1041, 628, "cam-01")
    low = (system + user).lower()
    for banned in ["detector found", "warm region", "has already found"]:
        assert banned not in low, f"prompt v3 ห้ามมีคำว่า {banned!r}"


def test_prompt_v3_ยังสั่งให้ตอบ_unknown_เมื่อดูไม่ออก():
    """ห้ามแลกความระวังทิ้งไปเพื่อให้จับช้างได้เยอะขึ้น

    ภาพความร้อนที่ระยะไกล ช้าง/ควาย/วัว เป็นก้อนขาวเหมือนกัน
    ถ้า prompt เชียร์ให้ตอบ elephant ไว้ก่อน ตัวเลขจะสวยแต่เชื่อไม่ได้
    """
    system, _ = prompt_v3.build(640, 512)
    assert "unknown" in system.lower()
    assert "do not invent" in system.lower()


def test_prompt_v3_ไม่ขอ_bbox():
    """วัดแล้ว bbox ของโมเดลเชื่อไม่ได้ (คืน y=998 บนภาพสูง 628) เลยไม่ขอ"""
    system, user = prompt_v3.build(640, 512)
    assert "bbox" not in (system + user).lower()


def test_image_type_ที่โมเดลตอบมา():
    assert image_type_of('{"image_type":"colour","animals":[]}') == "colour"
    assert image_type_of('{"image_type":"THERMAL","animals":[]}') == "thermal"
    assert image_type_of('{"image_type":"banana","animals":[]}') is None
    assert image_type_of('{"animals":[]}') is None          # v2 ไม่มีฟิลด์นี้ ไม่ใช่ error
    assert image_type_of("") is None


def test_parse_ยังอ่านคำตอบ_v3_ที่มี_image_type_ปนมา():
    v, err = parse('{"image_type":"colour","animals":[{"species":"elephant",'
                   '"count":2,"confidence":0.95}]}')
    assert err is None
    assert [(a.species, a.count) for a in v.animals] == [("elephant", 2)]


def test_mime_เดาจากหัว_base64():
    """เคยฝัง image/png ไว้ตายตัว ทั้งที่ภาพจากไซต์เป็น JPEG"""
    assert mime_of("/9j/4AAQSkZJRg") == "image/jpeg"
    assert mime_of("iVBORw0KGgo") == "image/png"
    assert mime_of("") == "image/png"
