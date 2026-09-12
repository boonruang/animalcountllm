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


def test_direction_นอกชุดเป็น_unknown_ไม่ใช่เดาความหมาย():
    """🔴 ห้ามแปลง entering/เข้า/towards ให้เอง

    ตัวเลขเข้าออกเป็นของที่ปลายทางเอาไปนับ เดาผิดหนึ่งคนคือยอดผิดหนึ่งคน
    โดยไม่มีใครเห็น · ถ้าโมเดลตอบนอกชุดบ่อย ให้ไปแก้ prompt
    """
    assert normalize({"direction": "entering"})["direction"] == "unknown"
    assert normalize({"direction": "เข้า"})["direction"] == "unknown"
    assert normalize({"direction": "IN"})["direction"] == "in"
    assert normalize({})["direction"] == "unknown"
    assert normalize({})["direction_confidence"] == 0.0


# ------------------------------------------------------- demographic + uniform
# เพิ่ม 2026-09-12 · Toy สั่งเพิ่ม Demographic Analytics, uniform, อารมณ์ 1-5

def test_age_group_คำนวณจาก_age_range_ขัดกันเองไม่ได้():
    """🔴 ไม่ได้ถามโมเดล เลยไม่มีทางได้ age_range "0-12" คู่กับ age_group "adult"

    ถ้าวันไหนมีใครเปลี่ยนไปถามโมเดลแทน เทสต์นี้จะยังเขียวอยู่จนกว่าโมเดลจะขัดกันเอง
    ซึ่งไม่เกิดในเทสต์ · เลยเช็คที่ฟังก์ชันตรงๆ ด้วย ไม่ใช่เช็คแค่ผลลัพธ์
    """
    from people.llm.client import age_group_of
    assert age_group_of("0-12") == "child"
    assert age_group_of("13-19") == "teen"
    for band in ("20-29", "30-39", "40-49", "50-59"):
        assert age_group_of(band) == "adult"
    assert age_group_of("60+") == "senior"
    assert age_group_of("unknown") == "unknown"
    assert age_group_of("ประมาณ 35 ปี") == "unknown"


def test_age_group_ตามค่าที่ลดทอนแล้ว_ไม่ใช่ค่าดิบ():
    """age_range ที่โมเดลตอบนอกชุด -> unknown -> age_group ต้อง unknown ตามกัน"""
    assert normalize({"age_range": "30-39"})["age_group"] == "adult"
    assert normalize({"age_range": "ประมาณ 35 ปี"})["age_group"] == "unknown"
    assert normalize({})["age_group"] == "unknown"


def test_uniform_ไม่ได้ใส่_กับ_ดูไม่ออก_ต้องแยกกัน():
    """🔴 none = แต่งตัวปกติ · unknown = ดูไม่ออก · ยุบกันเมื่อไหร่ยอดนับพนักงานเพี้ยน"""
    assert normalize({"appearance": {"uniform": {"kind": "none"}}}
                     )["appearance"].uniform.kind == "none"
    assert normalize({"appearance": {}})["appearance"].uniform.kind == "unknown"
    # ค่านอกชุด = ดูไม่ออก ไม่ใช่ = ไม่ได้ใส่
    assert normalize({"appearance": {"uniform": {"kind": "staff uniform"}}}
                     )["appearance"].uniform.kind == "unknown"


def test_uniform_อ่านข้อความบนชุดมาได้_และตัดที่_40_ตัว():
    """รปภ.ใน iplus_sample2 มีคำว่า SECURITY ปักอยู่ ซึ่งคือหลักฐานที่ตรวจย้อนได้"""
    u = normalize({"appearance": {"uniform": {"kind": "security", "color": "green",
                                              "id_badge": "yes",
                                              "text": "SECURITY"}}})["appearance"].uniform
    assert (u.kind, u.color, u.id_badge, u.text) == ("security", "green", "yes", "SECURITY")
    long = normalize({"appearance": {"uniform": {"text": "ก" * 90}}})
    assert len(long["appearance"].uniform.text) == 40


def test_อารมณ์_มองไม่เห็นหน้า_ต้องไม่กลายเป็น_neutral():
    """🔴 unknown/0 กับ neutral/3 คนละเรื่อง เหมือน unknown กับ degraded

    ยุบสองอันนี้เมื่อไหร่ "ลูกค้าเฉยๆ 80%" จะแปลว่า "เรามองไม่เห็นหน้า 80%"
    โดยไม่มีใครรู้
    """
    for raw in ({}, {"emotion": {}}, {"emotion": {"label": "unknown", "valence": 3}},
                {"emotion": {"label": "ยิ้ม"}}):
        e = normalize(raw)["emotion"]
        assert (e.label, e.valence) == ("unknown", 0), raw


def test_อารมณ์_label_ขัดกับ_valence_ถูกดึงกลับเข้าช่วง_ไม่ใช่ทิ้งทั้งก้อน():
    """โมเดลตอบ happy คู่กับ 1 ได้สบายๆ · เราเชื่อ label แล้วดึงเลขกลับเข้าช่วง"""
    e = normalize({"emotion": {"label": "happy", "valence": 1, "confidence": 0.8}})["emotion"]
    assert e.label == "happy" and e.valence == 4 and e.confidence == 0.8
    e = normalize({"emotion": {"label": "angry", "valence": 5}})["emotion"]
    assert e.label == "angry" and e.valence == 2
    # ตอบ label มาแต่ไม่ตอบเลข = เติมจากตาราง ไม่ใช่ 0
    assert normalize({"emotion": {"label": "neutral"}})["emotion"].valence == 3
    # 🔴 สะกดรูป adjective ที่โมเดลชอบใช้ ต้องรับไว้ ไม่ใช่ทิ้งเป็น unknown
    # เราขอ noun ตาม FER2013 แต่โมเดลตอบ "surprised" มาบ่อยกว่า "surprise"
    for said, want in (("surprised", "surprise"), ("fearful", "fear"),
                       ("disgusted", "disgust"), ("anger", "angry"),
                       ("calm", "neutral")):
        e = normalize({"emotion": {"label": said}})["emotion"]
        assert e.label == want, f"{said} -> {e.label}, ควรเป็น {want}"
        assert e.valence > 0
    # เลขที่อยู่ในช่วงอยู่แล้ว ห้ามถูกแตะ
    assert normalize({"emotion": {"label": "happy", "valence": 5}})["emotion"].valence == 5


def test_group_เส้น_persons_ต้องเป็น_unknown_ไม่ใช่_alone():
    """🔴 โมเดลเห็นคนเดียวจะรู้ได้ยังไงว่าเขามากับใคร

    "อยู่คนเดียวในภาพที่เราส่งให้ดู" ไม่ได้แปลว่า "เดินคนเดียว"
    ต่อให้โมเดลตอบ group มาเองก็ต้องทิ้ง
    """
    g = normalize({"group": "G1"})["group"]
    assert (g.ref, g.size, g.type) == ("", 0, "unknown")


def test_group_นับขนาดจาก_ref_ที่ซ้ำกัน_ไม่ให้โมเดลนับ():
    from people.llm.client import apply_groups
    ns = [normalize({"group": g}, with_group=True)
          for g in ("G1", "G1", "G2", "G1", "G3")]
    apply_groups(ns)
    sizes = [(n["group"].size, n["group"].type) for n in ns]
    assert sizes == [(3, "group_3_plus"), (3, "group_3_plus"), (1, "alone"),
                     (3, "group_3_plus"), (1, "alone")]


def test_group_คนที่โมเดลไม่ตอบ_ref_เป็น_unknown_ไม่ใช่_alone():
    from people.llm.client import apply_groups
    ns = [normalize({"group": "G1"}, with_group=True),
          normalize({}, with_group=True),
          normalize({"group": "unknown"}, with_group=True)]
    apply_groups(ns)
    assert ns[0]["group"].type == "alone"
    assert ns[1]["group"].type == "unknown" and ns[1]["group"].size == 0
    assert ns[2]["group"].type == "unknown"


def test_สัญชาติปิดอยู่_ตอบมาก็ทิ้ง():
    """🔴 ชั้นที่สองของสวิตช์ · prompt ไม่ถามอยู่แล้ว แต่โมเดลตอบเองได้

    ด่านที่มีชั้นเดียวคือด่านที่พังเงียบวันที่มีคนแก้ prompt แล้วลืมสวิตช์
    """
    import people.llm.client as c
    assert c.ALLOW_NATIONALITY is False, "ค่าเริ่มต้นต้องปิด ไม่งั้นมันหลุดขึ้น prod"
    n = normalize({"nationality": "thai", "nationality_confidence": 0.9})
    assert n["nationality"] == "unknown" and n["nationality_confidence"] == 0.0


def test_ชุดค่าในโค้ดกับใน_prompt_ต้องตรงกันทุกตัว():
    """🔴 บทเรียน `laptop` 2026-09-11 · ชุดค่าที่ต้องตรงกันแต่แก้คนละที่ จะไม่ตรงกัน

    ตอนนั้น `laptop` อยู่ในโค้ดแต่ไม่อยู่ใน prompt โมเดลเลยไม่รู้ว่าเลือกได้
    มันเขียนว่า "ถือแล็ปท็อปสีเทา" ใน distinctive ถูกต้อง แต่ carrying ว่าง
    **ไม่มี error ให้เห็น มีแต่ข้อมูลที่หายไปเงียบๆ**

    เทสต์นี้ไล่จาก schema เป็นตัวตั้ง ไม่ใช่จากรายการที่พิมพ์ไว้เอง
    เพิ่มค่าใหม่ใน schemas.py แล้วลืมเติมใน prompt = ตกที่นี่ทันที
    ไม่ต้องรอให้ใครสังเกตว่าคำตอบมันแปลกๆ
    """
    import typing
    from people.llm.prompt_p1 import ALLOWED
    from people.schemas import (Appearance, Emotion, Garment, Uniform)

    checks = [("uniform.kind", Uniform, "kind"),
              ("uniform.id_badge", Uniform, "id_badge"),
              ("emotion.label", Emotion, "label"),
              ("direction", None, None),        # ไล่ด้วยมือด้านล่าง
              ("gender", None, None)]
    for name, cls, field in checks:
        if cls is None:
            continue
        want = set(typing.get_args(cls.model_fields[field].annotation))
        line = [l for l in ALLOWED.splitlines() if l.startswith(name + ":")]
        assert line, f"ไม่มีบรรทัด {name} ใน prompt เลย"
        got = {v.strip() for v in line[0].split(":", 1)[1].split(",")}
        assert want == got, f"{name} ไม่ตรงกัน: {want ^ got}"

    # ของที่มีอยู่ก่อนแล้ว ตรวจด้วยกันไปเลยจะได้ไม่ต้องมีเทสต์สองตัว
    for name, cls, field in (("skin_tone", Appearance, "skin_tone"),
                             ("build", Appearance, "build"),
                             ("height", Appearance, "height"),
                             ("top_sleeve", Appearance, "top_sleeve"),
                             ("hair_length", Appearance, "hair_length"),
                             ("hair_color", Appearance, "hair_color")):
        want = set(typing.get_args(cls.model_fields[field].annotation))
        line = [l for l in ALLOWED.splitlines() if l.startswith(name + ":")][0]
        got = {v.strip() for v in line.split(":", 1)[1].split(",")}
        assert want == got, f"{name} ไม่ตรงกัน: {want ^ got}"

    # carrying เป็น List[Literal] ซ้อนอีกชั้น และใน prompt ไม่มี unknown โดยตั้งใจ
    want = set(typing.get_args(typing.get_args(
        Appearance.model_fields["carrying"].annotation)[0]))
    line = [l for l in ALLOWED.splitlines() if l.startswith("carrying:")][0]
    got = {v.strip() for v in line.split(":", 1)[1].split(",")}
    assert want == got, f"carrying ไม่ตรงกัน: {want ^ got}"

    # สีกับลาย ใช้คำว่า "any colour field" ใน prompt เลยต้องหาแบบอื่น
    for key, field in (("any colour field", "color"), ("any pattern field", "pattern")):
        want = set(typing.get_args(Garment.model_fields[field].annotation))
        line = [l for l in ALLOWED.splitlines() if l.startswith(key + ":")][0]
        got = {v.strip() for v in line.split(":", 1)[1].split(",")}
        assert want == got, f"{key} ไม่ตรงกัน: {want ^ got}"


def test_prompt_ไม่ถามสัญชาติเลยตอนสวิตช์ปิด():
    """ปิดแล้วต้องไม่มีคำนี้ใน prompt เลย ไม่ใช่ถามแล้วทิ้งคำตอบ

    ถามแล้วทิ้ง = จ่ายเงินให้โมเดลคิดเรื่องที่เราไม่ได้จะใช้ แถมมันจะเอาไปถ่วง
    คำตอบช่องอื่นด้วย · เปิดแล้วกฎต้องอยู่ **ก่อน** บรรทัด "Reply with JSON only"
    โมเดลบางตัวถือว่าบรรทัดนั้นจบคำสั่ง แล้วอ่านของที่ตามมาเป็นข้อมูลแทน
    """
    from people.llm import prompt_f1, prompt_p1
    for mod, args in ((prompt_p1, ("021", 224, 515, "cam-1")),
                      (prompt_f1, (1916, 1080, "cam-1", 12))):
        sys_off, user_off = mod.build(*args, nationality=False)
        assert "nationality" not in sys_off and "nationality" not in user_off
        sys_on, user_on = mod.build(*args, nationality=True)
        assert "nationality" in sys_on and "nationality" in user_on
        assert sys_on.index("`nationality`") < sys_on.index("Reply with JSON only")


# ------------------------------------------------- ค่าเป็นไทย คีย์เป็นอังกฤษ
# Toy สั่ง 2026-09-12 · แปลตอนขาออกที่เดียว ของในฐานยังเป็นอังกฤษ (ดู people/th.py)

def test_ทุกค่าใน_schema_ต้องมีคำแปลไทย():
    """🔴 ด่านเดียวกับ prompt parity แต่คนละปลาย · ไล่จาก schema เป็นตัวตั้ง

    เพิ่มค่าใหม่ใน `schemas.py` แล้วลืมเติมคำแปล = **ค่านั้นหลุดเป็นอังกฤษ
    ปนไทยไปให้ปลายทางเจอเอง** ซึ่งไม่มี error ให้เห็น มีแต่ JSON ที่หน้าตาแปลกๆ
    บางช่อง · เป็นบั๊กสายเดียวกับ `laptop` เป๊ะ ต่างแค่ปลายทาง
    """
    import typing
    from people import th
    from people.schemas import (Appearance, Emotion, Garment, GroupInfo,
                                PersonOut, Uniform)

    def vals(cls, field):
        ann = cls.model_fields[field].annotation
        args = typing.get_args(ann)
        return set(args) if args else set()

    checks = [
        ("direction", vals(PersonOut, "direction")),
        ("gender", vals(PersonOut, "gender")),
        ("age_group", vals(PersonOut, "age_group")),
        ("nationality", vals(PersonOut, "nationality")),
        ("group.type", vals(GroupInfo, "type")),
        ("emotion.label", vals(Emotion, "label")),
        ("uniform.kind", vals(Uniform, "kind")),
        ("uniform.id_badge", vals(Uniform, "id_badge")),
        ("color", vals(Garment, "color")),
        ("secondary_color", vals(Garment, "secondary_color")),
        ("pattern", vals(Garment, "pattern")),
    ]
    for f in ("skin_tone", "build", "height", "hair_length", "hair_color",
              "glasses", "face_mask", "facial_hair", "headwear", "top_sleeve"):
        checks.append((f, vals(Appearance, f)))
    # carrying เป็น List[Literal[...]] ซ้อนอีกชั้น
    checks.append(("carrying", set(typing.get_args(typing.get_args(
        Appearance.model_fields["carrying"].annotation)[0]))))

    for path, want in checks:
        table = th.FIELD.get(path)
        assert table, f"ไม่มีตารางแปลของ {path}"
        missing = want - set(table)
        assert not missing, f"{path} ไม่มีคำแปลของ {sorted(missing)}"
        # แปลแล้วต้องเป็นไทยจริง ไม่ใช่ก๊อปคำอังกฤษมาวาง
        for k in want:
            assert table[k] != k, f"{path}.{k} ยังเป็นอังกฤษอยู่"

    # ชนิดเสื้อผ้าใช้ตารางแยกตามชิ้น เพราะคีย์ชื่อ `type` เหมือนกันหมด
    from people.schemas import BottomType, FootwearType, TopType
    for path, ann in (("top.type", TopType), ("bottom.type", BottomType),
                      ("footwear.type", FootwearType)):
        want = set(typing.get_args(ann))
        table = th.FIELD[path]
        assert not (want - set(table)), f"{path} ขาด {sorted(want - set(table))}"
    # outer.type ไม่มี Literal ของตัวเองใน schema (Garment.type เป็น str อิสระ)
    # ชุดค่าจริงอยู่ใน prompt · เช็คว่าอย่างน้อยมี none กับ unknown แยกกัน
    assert th.FIELD["outer.type"]["none"] != th.FIELD["outer.type"]["unknown"]


def test_คีย์โปรโตคอลห้ามถูกแปล():
    """🔴 `status` `source` `id` `ref` `uniform.text` ต้องเป็นอังกฤษ/ค่าเดิมเสมอ

    ปลายทางเขียน `if status == "ok"` ไว้แล้ว · แปลเมื่อไหร่โค้ดเขาพังเงียบๆ
    ในวันที่ทุกอย่างดูปกติดี · ส่วน `uniform.text` คือข้อความที่ลอกมาจากชุดจริง
    มันคือหลักฐาน ไม่ใช่ค่าที่เราตีความ แปลแล้วไม่ใช่หลักฐานอีกต่อไป
    """
    from people.th import thai
    d = thai({"status": "ok", "id": "P1", "ref": "iplus-001",
              "image": {"source": "full_frame", "w": 100},
              "appearance": {"uniform": {"kind": "security", "text": "SECURITY"}},
              "group": {"ref": "G1", "type": "pair", "size": 2}})
    assert d["status"] == "ok"
    assert d["id"] == "P1" and d["ref"] == "iplus-001"
    assert d["image"]["source"] == "full_frame" and d["image"]["w"] == 100
    assert d["appearance"]["uniform"]["text"] == "SECURITY"   # หลักฐาน ห้ามแตะ
    assert d["appearance"]["uniform"]["kind"] == "รปภ."        # ค่าตีความ ต้องแปล
    assert d["group"]["ref"] == "G1" and d["group"]["size"] == 2
    assert d["group"]["type"] == "มากัน 2 คน"


def test_แปลแล้วไม่แก้ของเดิม():
    """🔴 ก้อนเดียวกันถูกเอาไปเขียนลงฐานด้วย แก้ในที่ = ฐานกลายเป็นไทยโดยไม่ตั้งใจ"""
    from people.th import thai
    src = {"gender": "male", "appearance": {"glasses": "yes", "carrying": ["phone"]}}
    out = thai(src)
    assert src == {"gender": "male",
                   "appearance": {"glasses": "yes", "carrying": ["phone"]}}
    assert out["gender"] == "ชาย"
    assert out["appearance"]["glasses"] == "ใส่"
    assert out["appearance"]["carrying"] == ["โทรศัพท์"]


def test_yes_no_ใช้คำต่างกันตามฟิลด์():
    """แว่นตา/แมสก์/หมวก = ใส่/ไม่ใส่ · หนวดเครา/ป้ายชื่อ = มี/ไม่มี (Toy ยกตัวอย่างเอง)"""
    from people.th import thai
    d = thai({"appearance": {"glasses": "yes", "face_mask": "no", "headwear": "yes",
                             "facial_hair": "yes",
                             "uniform": {"id_badge": "no"}}})["appearance"]
    assert d["glasses"] == "ใส่" and d["face_mask"] == "ไม่ใส่"
    assert d["headwear"] == "ใส่"
    assert d["facial_hair"] == "มี" and d["uniform"]["id_badge"] == "ไม่มี"


def test_ค่าที่ไม่รู้จักปล่อยผ่าน_ไม่ใช่ทำให้พัง():
    """ตารางแปลไม่ใช่ด่านตรวจ · ด่านตรวจคือ normalize ที่ทำไปก่อนหน้านี้แล้ว"""
    from people.th import thai
    d = thai({"gender": "อะไรสักอย่าง", "direction": None, "weird_key": "value"})
    assert d["gender"] == "อะไรสักอย่าง"
    assert d["direction"] is None and d["weird_key"] == "value"


def test_คำไทยในหน้า_verify_ต้องตรงกับ_people_th():
    """🔴 หน้า verify คีย์ตารางสี/ไอคอนด้วย **คำไทย** ซึ่งมาจาก people/th.py

    ถอดตารางแปลของหน้าเว็บทิ้งไปแล้ว แต่ตารางสีกับไอคอนยังต้องรู้ค่าถึงจะเลือกได้
    เลยยังมีคำไทยฝังอยู่ในไฟล์ html · **คำพวกนั้นกับ people/th.py ต้องตรงกันเป๊ะ**
    แก้คำแปลฝั่ง Python แล้วลืมแก้หน้าเว็บ = ป้ายกลายเป็นสีเทาหมด ไอคอนเป็น
    เครื่องหมายคำถามหมด **โดยไม่มี error ให้เห็นเลย** หน้ายังโหลดปกติทุกประการ
    """
    import pathlib
    import re
    from people import th

    html = (pathlib.Path(__file__).resolve().parent.parent
            / "people" / "ui" / "index.html").read_text(encoding="utf-8")

    def keys_of(name):
        m = re.search(r"const " + name + r"\s*=\s*\{(.*?)\};", html, re.S)
        assert m, f"หาตาราง {name} ในหน้าเว็บไม่เจอ"
        return set(re.findall(r"'([^']*[ก-๙][^']*)'\s*:", m.group(1)))

    assert keys_of("DIRC") == set(th.FIELD["direction"].values())
    assert keys_of("DIRI") == set(th.FIELD["direction"].values())
    assert keys_of("EMOI") == set(th.FIELD["emotion.label"].values())
    assert keys_of("HEX") == set(th.FIELD["color"].values())

    # ค่าคงที่ที่หน้าเว็บใช้เทียบตรงๆ (T.unknown, T.noUniform, ...)
    m = re.search(r"const T = \{(.*?)\};", html, re.S)
    assert m
    t = dict(re.findall(r"(\w+):\s*'([^']*)'", m.group(1)))
    assert t["unknown"] == th.FIELD["direction"]["unknown"]
    assert t["noFace"] == th.FIELD["emotion.label"]["unknown"]
    assert t["noUniform"] == th.FIELD["uniform.kind"]["none"]
    assert t["noOuter"] == th.FIELD["outer.type"]["none"]
    assert t["plain"] == th.FIELD["pattern"]["plain"]
    assert t["has"] == th.FIELD["uniform.id_badge"]["yes"]

    # เพศที่หน้าเว็บใช้นับยอดในกล่องสรุป ต้องเป็นคำเดียวกับที่ API ส่งมา
    assert f"p.gender === '{th.FIELD['gender']['male']}'" in html
    assert f"p.gender === '{th.FIELD['gender']['female']}'" in html


def test_เฉดสีที่แคบเกินไปต้องไม่ถูกทิ้ง():
    """🔴 วัดจริงบน prod 2026-09-12 · รปภ. ชุดเขียวเข้มได้ color unknown กลับมา

    โมเดลอ่านถูก เขียนว่า "เสื้อสีเขียวเข้ม" ในช่องบรรยาย แต่ `dark_green`
    ไม่อยู่ในชุดสี เราเลยทิ้งทิ้งไปเอง **บั๊ก `laptop` รอบที่สอง ต่างแค่ฟิลด์**

    ปอกคำขยายทิ้ง ไม่ใช่เพิ่มเฉดลงในชุด · ชุดสีตั้งใจให้หยาบเพื่อให้กรองได้ว่า
    "หาคนเสื้อเขียว" แล้วเจอทั้งเขียวเข้มเขียวอ่อน เพิ่มเฉดเมื่อไหร่การกรองก็แตก
    """
    for said, want in (("dark_green", "green"), ("light blue", "blue"),
                       ("deep_navy", "navy"), ("bright_red", "red"),
                       ("pale_pink", "pink")):
        n = normalize({"appearance": {"top": {"color": said}}})
        assert n["appearance"].top.color == want, f"{said} -> {n['appearance'].top.color}"
    # คำขยายที่ปอกแล้วยังไม่อยู่ในชุด ต้องเป็น unknown เหมือนเดิม ไม่ใช่ปล่อยผ่าน
    assert normalize({"appearance": {"top": {"color": "dark_teal"}}}
                     )["appearance"].top.color == "unknown"
    # และใช้กับ uniform.color ด้วย ซึ่งคือช่องที่เจอปัญหาจริง
    assert normalize({"appearance": {"uniform": {"color": "dark_green"}}}
                     )["appearance"].uniform.color == "green"


def test_prompt_บอกว่าบัตรแขวนคอไม่ใช่เครื่องแบบ():
    """🔴 วัดจริงบน prod 2026-09-12 · คู่ที่ใส่เชิ้ตขาว+สูทดำ+บัตรแขวนคอ
    ถูกตอบว่า `corporate` ทั้งที่ prompt เขียนไว้แล้วว่าสูทไปทำงานคือ `none`

    บัตรแขวนคอมีช่องของตัวเองอยู่แล้ว (`id_badge`) มันไม่ควรมีสิทธิ์ตัดสินช่องนี้
    เทสต์นี้ล็อกถ้อยคำไว้ ไม่ใช่ล็อกพฤติกรรมของโมเดล (ซึ่งล็อกไม่ได้)
    ใครถอดกฎนี้ออกจะได้เห็นว่ามันเคยมีเหตุผล
    """
    from people.llm import prompt_f1, prompt_p1
    for mod in (prompt_p1, prompt_f1):
        sysmsg = mod.SYSTEM
        assert "lanyard" in sysmsg and "id_badge" in sysmsg
        assert "office clothes are ordinary clothes" in sysmsg
        assert 'Pick "corporate" only on real evidence' in sysmsg
