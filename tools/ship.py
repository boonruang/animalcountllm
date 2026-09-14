"""ship.py — ทางเดียวที่ได้รับอนุญาตให้ push main

    python tools/ship.py --check     ตรวจอย่างเดียว ไม่ push
    python tools/ship.py             ตรวจแล้ว merge dev -> main แล้ว push ทั้งคู่

🔴 main ต่อสายตรงกับ DigitalOcean (deploy_on_push: true)
push main = deploy ของจริงทันที ไม่มีขั้นให้ทบทวน

มีไฟล์นี้เพราะเคยพลาดมาแล้ว 2026-08-16: commit ลง dev แล้วสั่ง push main
ซึ่งตอนนั้น main ยังไม่มี commit นั้น · ถ้า remote ไม่ปฏิเสธไว้ Toy จะได้ deploy
ที่ไม่มีของที่คิดว่ามี · "ระวังให้มากขึ้น" แก้ปัญหาแบบนี้ไม่ได้ ต้องมีตัวตรวจ
"""
from __future__ import annotations

import re
import subprocess
import sys

SECRET_PATTERNS = [
    (r"sk-or-v1-[A-Za-z0-9]{20,}", "OpenRouter API key"),
    (r"lsv2_(pt|sk)_[A-Za-z0-9]{20,}", "LangSmith API key"),
    (r"sk-[A-Za-z0-9]{32,}", "OpenAI-style key"),
]


def run(*args: str, check: bool = True) -> str:
    r = subprocess.run(["git", *args], capture_output=True, text=True)
    if check and r.returncode:
        fail(f"git {' '.join(args)} ล้มเหลว:\n{r.stderr.strip()}")
    return r.stdout.strip()


def fail(msg: str) -> None:
    print(f"\n🔴 หยุด: {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"  ✅ {msg}")


def check_clean() -> None:
    dirty = run("status", "--porcelain")
    if dirty:
        fail("working tree ยังไม่สะอาด commit หรือ stash ก่อน:\n" + dirty)
    ok("working tree สะอาด")


def check_env_not_tracked() -> None:
    tracked = run("ls-files").splitlines()
    bad = [f for f in tracked if f == ".env" or f.startswith(".env.") and f != ".env.example"]
    if bad:
        fail(f"ไฟล์ลับถูก track อยู่: {bad}")
    ok(".env ไม่ได้ถูก track")


def check_no_secrets() -> None:
    """สแกนไฟล์ที่ track อยู่จริง ไม่ใช่แค่ diff

    repo เป็น public · คีย์ที่ commit ไปแล้วถือว่ารั่วถาวร ลบ commit ทีหลังก็ยังอยู่ใน history
    """
    hits = []
    for f in run("ls-files").splitlines():
        try:
            text = open(f, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        for pat, name in SECRET_PATTERNS:
            for m in re.finditer(pat, text):
                if "xxxx" in m.group(0).lower():
                    continue  # ตัวอย่างใน .env.example
                hits.append(f"{f}: {name} ({m.group(0)[:14]}...)")
    if hits:
        fail("เจอคีย์จริงในไฟล์ที่จะขึ้น GitHub:\n  " + "\n  ".join(hits))
    ok("ไม่มีคีย์จริงในไฟล์ที่ track")


def check_version_bumped() -> None:
    """เลขเวอร์ชันต้องขยับเมื่อโค้ดของงานนั้นเปลี่ยน

    ไม่มีตัวนี้ = ไม่มีทางรู้ว่า DO รันคอมมิตไหนอยู่ ซึ่งเสียเวลาไปแล้วสามรอบ

    🔴 ตรวจแยกกันทุกงาน เพราะ **ปลายทางคนละทีม** งานช้างมี APP_VERSION
    งานคนมี PEOPLE_VERSION งาน LPR มี LPR_VERSION · ของเดิมเห็นแค่ `app/`
    แปลว่าแก้ `people/` ทั้งก้อนแล้วลืมขยับเลข ด่านนี้ก็ไม่ร้อง ซึ่งเป็นรูเดียว
    กับที่ check_tests เคยโดน: **ด่านที่รู้จัก repo แค่บางส่วน จะล้าสมัยวันที่
    repo โตขึ้น และมันจะล้าสมัยแบบเขียว** ไม่ใช่แบบที่มีใครเห็น

    เพิ่มบริการใหม่ใต้ repo นี้เมื่อไหร่ ต้องมาเติมแถวในตารางนี้ด้วยเสมอ
    """
    changed = run("diff", "--name-only", "origin/main..dev").splitlines()
    for prefix, path, const in (("app/", "app/main.py", "APP_VERSION"),
                                ("people/", "people/main.py", "PEOPLE_VERSION"),
                                ("lpr/", "lpr/main.py", "LPR_VERSION")):
        if not any(f.startswith(prefix) for f in changed):
            continue
        cur = run("show", f"HEAD:{path}", check=False)
        old = run("show", f"origin/main:{path}", check=False)
        pattern = const + r'\s*=\s*"([^"]+)"'

        def ver(text: str) -> str:
            m = re.search(pattern, text)
            return m.group(1) if m else ""

        if old and ver(cur) and ver(cur) == ver(old):
            fail(f"โค้ดใน {prefix} เปลี่ยนแต่ {const} ยังเป็น {ver(cur)} "
                 f"· ขยับก่อน ไม่งั้นดูไม่ออกว่า deploy ทันหรือยัง")
        if ver(cur):
            ok(f"{const} = {ver(cur)}")


def check_tests() -> None:
    """รันทั้งโฟลเดอร์ `tests` ไม่ใช่ไล่จากรายชื่อไฟล์

    🔴 ของเดิมไล่จากรายชื่อ `test_cv.py` กับ `test_filters.py` ซึ่งเป็นสองไฟล์
    ที่มีอยู่ ณ วันที่เขียน ship.py · เทสต์ฝั่งคนอีก 70 กว่าข้อ ด่านนี้ไม่เคยเห็นเลย
    **และ commit ที่มีเทสต์ตกขึ้น main ไปแล้วจริงๆ โดยไม่มีใครร้อง** (2026-09-11)

    ที่เจ็บกว่านั้น: commit `8c937e5` เขียนในข้อความว่าแก้เรื่องนี้แล้ว
    **แต่ไฟล์นี้ไม่ได้อยู่ใน commit นั้น** แก้ในข้อความ ไม่ได้แก้ในไฟล์ เลยกลายเป็น
    ด่านที่ทั้ง CLAUDE.md ทั้ง vault และทั้ง commit message บอกว่าปิดรูแล้ว
    ทั้งที่รูยังเปิดอยู่ · เจอ 2026-09-12 ตอนจะ deploy เพราะอ่านโค้ดก่อนใช้
    **เอกสารที่บอกว่าแก้แล้ว ไม่ใช่หลักฐานว่าแก้แล้ว**

    **ด่านที่ตรวจจากรายชื่อไฟล์ จะล้าสมัยเงียบๆ เสมอ** ชี้ที่โฟลเดอร์แทน
    """
    r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"],
                       capture_output=True, text=True,
                       env={"PYTHONUTF8": "1", **__import__("os").environ})
    if r.returncode:
        fail(f"pytest tests ไม่ผ่าน:\n{r.stdout[-2000:]}")
    lines = [l for l in r.stdout.strip().splitlines() if l.strip()]
    ok(f"pytest tests: {lines[-1] if lines else 'ผ่าน'}")


def check_branches() -> tuple[str, str]:
    """ของที่จะ deploy ต้องเป็นของที่เพิ่งเทสต์ ไม่ใช่ของบน branch อื่น"""
    cur = run("rev-parse", "--abbrev-ref", "HEAD")
    if cur != "dev":
        fail(f"ต้องอยู่บน dev ตอนสั่ง ship · ตอนนี้อยู่ {cur}")
    ok("อยู่บน dev")

    run("fetch", "-q", "origin")
    behind = run("rev-list", "--count", "dev..origin/dev")
    if behind != "0":
        fail(f"origin/dev นำหน้าอยู่ {behind} commit · pull ก่อน")
    ok("dev ตรงกับ origin/dev")

    ahead = run("rev-list", "--count", "origin/main..dev")
    if ahead == "0":
        fail("dev ไม่มีอะไรใหม่กว่า origin/main ไม่มีอะไรต้อง deploy")
    return cur, ahead


def main() -> None:
    check_only = "--check" in sys.argv
    print("ตรวจก่อน ship\n")
    check_clean()
    check_env_not_tracked()
    check_no_secrets()
    check_tests()
    check_version_bumped()
    _, ahead = check_branches()

    print(f"\nสิ่งที่จะ deploy ({ahead} commit):")
    for line in run("log", "--oneline", "origin/main..dev").splitlines():
        print("  " + line)

    if check_only:
        print("\n--check เท่านั้น ไม่ push")
        return

    print("\npush...")
    run("checkout", "-q", "main")
    merged = subprocess.run(["git", "merge", "--ff-only", "dev"],
                            capture_output=True, text=True)
    if merged.returncode:
        # main มี commit ที่ dev ไม่มี (เช่น merge PR บน GitHub) รวมแบบปกติแทน
        run("merge", "--no-edit", "dev")
    run("push", "origin", "main")
    run("checkout", "-q", "dev")
    run("merge", "--ff-only", "main")
    run("push", "origin", "dev")
    print("  ✅ push main + dev แล้ว · DigitalOcean จะ deploy เอง")
    print("     เช็คหลัง deploy: /healthz ต้องมี store_path และ status=ok")


if __name__ == "__main__":
    main()
