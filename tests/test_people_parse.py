"""แกะและลดทอนคำตอบของโมเดล · ไม่ต้องมีเน็ต ไม่ต้องมีโมเดล

ที่นี่คือด่านที่ตัดสินว่าคำตอบเพี้ยนๆ ของโมเดลจะกลายเป็น "เสียหนึ่งช่อง"
หรือ "เสียทั้งคน" · โมเดลตอบนอกชุดค่าเป็นเรื่องปกติ ไม่ใช่เหตุฉุกเฉิน
แต่ถ้าปล่อยหลุดไปถึง pydantic มันจะโยน ValidationError แล้วคนคนนั้นหายไปทั้งคน
ทั้งที่ฟิลด์อื่นอ่านได้ครบ
"""
from people.llm.client import normalize, parse


def test_แกะ_json_ที่ห่อด้วย_markdown_fence():
    """สั่งว่าอย่าห่อแล้วมันก็ยังห่อ · ฝั่งช้างเจอมาแล้ว"""
    d, err = parse('```json\n{"gender":"male"}\n```')
    assert err is None and d["gender"] == "male"


def test_แกะ_json_ที่มี_object_ซ้อน_ไม่ตัดกลาง():
    """คำตอบฝั่งนี้มี top/bottom/footwear ซ้อนอยู่ regex ที่ขี้เกียจจะตัดกลาง"""
    text = ('นี่คือคำตอบครับ {"gender":"female","appearance":'
            '{"top":{"type":"blouse","color":"white"}},"description":"ok"} จบ')
    d, err = parse(text)
    assert err is None
    assert d["appearance"]["top"]["color"] == "white"


def test_คำตอบว่างหรือไม่ใช่_json():
    assert parse("")[0] is None
    assert parse("ผมขอโทษ ผมดูภาพนี้ไม่ออกครับ")[0] is None


def test_ค่านอกชุดกลายเป็น_unknown_ไม่ใช่ระเบิด():
    n = normalize({"gender": "ชาย", "age_range": "ประมาณ 35 ปี",
                   "appearance": {"skin_tone": "light-medium",
                                  "top": {"type": "shirt", "color": "crimson"}}})
    assert n["gender"] == "unknown"
    assert n["age_range"] == "unknown"
    assert n["appearance"].skin_tone == "unknown"
    assert n["appearance"].top.color == "unknown"
    # ชนิดเสื้อเป็น free text โดยตั้งใจ ค่าที่ถูกต้องยังต้องผ่านมาได้
    assert n["appearance"].top.type == "shirt"


def test_สะกดอเมริกันยังผ่าน():
    """gray/multicolor เป็นคำที่โมเดลส่วนใหญ่ใช้ ทิ้งข้อมูลจริงเพราะสะกดคนละแบบคือโง่"""
    n = normalize({"appearance": {"top": {"color": "gray"},
                                  "bottom": {"color": "multicolor"}}})
    assert n["appearance"].top.color == "grey"
    assert n["appearance"].bottom.color == "multicolour"


def test_boolean_แทน_yes_no():
    n = normalize({"appearance": {"glasses": True, "face_mask": False}})
    assert n["appearance"].glasses == "yes"
    assert n["appearance"].face_mask == "no"


def test_confidence_ที่เพี้ยนถูกหนีบ_ไม่ใช่ทำให้พัง():
    n = normalize({"gender_confidence": 1.8, "age_range_confidence": -2,
                   "appearance_confidence": "สูงมาก"})
    assert n["gender_confidence"] == 1.0
    assert n["age_range_confidence"] == 0.0
    assert n["appearance_confidence"] == 0.0


def test_ไม่มี_appearance_มาเลยก็ยังได้ของครบ():
    """ทุกช่องเป็น unknown ไม่ใช่ None และไม่ใช่ฟิลด์หาย"""
    n = normalize({"gender": "female"})
    a = n["appearance"]
    assert a.skin_tone == a.hair_color == a.footwear.type == "unknown"
    assert a.carrying == []


def test_carrying_กรองของที่ไม่รู้จักทิ้งและมีเพดาน():
    n = normalize({"appearance": {"carrying": [
        "backpack", "กระเป๋า", "phone", "phone", "umbrella", "badge",
        "lanyard", "box", "tote"]}})
    c = n["appearance"].carrying
    assert "backpack" in c and "กระเป๋า" not in c
    assert len(c) == len(set(c)), "ของซ้ำต้องถูกยุบ"
    assert len(c) <= 6


def test_description_ยาวเกินถูกตัด_ไม่ใช่ปฏิเสธ():
    n = normalize({"description": "ก" * 5000})
    assert len(n["description"]) == 600


def test_ห้ามเดาแทนโมเดล():
    """🔴 กฎข้อเดียวที่ normalize ห้ามละเมิด

    ไม่ตอบ = unknown เท่านั้น · ห้ามมีบรรทัดไหนเติมค่าที่ดูน่าจะใช่ให้
    """
    n = normalize({})
    assert n["gender"] == "unknown"
    assert n["age_range"] == "unknown"
    assert n["description"] == ""
    assert n["gender_confidence"] == 0.0
