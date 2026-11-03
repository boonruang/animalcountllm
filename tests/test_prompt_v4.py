"""เทสต์ prompt v4 · นับช้างอย่างเดียว · รันได้โดยไม่มีเน็ต ไม่มีโมเดล"""
from app.llm import prompt_v4
from app.llm.client import VisionLLM, LLMResult, elephant_only
from app.schemas import LLMVerdict, SpeciesTally


def _t(species, count=1, confidence=0.5):
    return SpeciesTally(species=species, count=count, confidence=confidence)


def test_ตัวเลือกชนิดมีแค่ช้างกับ_unknown():
    _, user = prompt_v4.build(640, 512)
    assert '"<elephant|unknown>"' in user
    assert "human" not in user and "cattle" not in user


def test_ตัวอย่าง_json_ไม่มีค่าจริงให้ลอก():
    _, user = prompt_v4.build(640, 512)
    assert '"species":"elephant"' not in user
    assert '"count":<whole number>' in user


def test_prompt_ไม่บอกว่าตรวจเจอหรือไม่เจออะไรมาก่อน():
    system, user = prompt_v4.build(640, 512)
    low = (system + user).lower()
    for banned in ["detector found", "warm region", "has already found", "bbox"]:
        assert banned not in low


def test_ด่านในโค้ด_ตัดคน_สัตว์อื่นเป็นช้าง_unknown_คงไว้():
    got = elephant_only([_t("human", 3), _t("cattle", 2, 0.4), _t("other_animal"),
                         _t("elephant", 1, 0.9), _t("unknown", 1, 0.2)])
    assert [(a.species, a.count, a.confidence) for a in got] == [
        ("elephant", 2, 0.4), ("elephant", 1, 0.5),
        ("elephant", 1, 0.9), ("unknown", 1, 0.2)]


def test_look_ใช้_v4_และผ่านด่าน(monkeypatch):
    seen = {}

    def fake(self, system, user, image_b64, image_hash, prompt_version, blobs, frame):
        seen["version"] = prompt_version
        return LLMResult(LLMVerdict(animals=[_t("human", 2)]), "raw", "stop", 1, 1.0)

    monkeypatch.setattr(VisionLLM, "_invoke", fake)
    res = VisionLLM().look("x", 10, 10)
    assert seen["version"] == "v4"
    assert res.verdict.animals == []
    assert res.raw == "raw"
