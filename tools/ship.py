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
    """APP_VERSION ต้องขยับเมื่อโค้ดใน app/ เปลี่ยน

    ไม่มีตัวนี้ = ไม่มีทางรู้ว่า DO รันคอมมิตไหนอยู่ ซึ่งเสียเวลาไปแล้วสามรอบ
    """
    changed = run("diff", "--name-only", "origin/main..dev").splitlines()
    if not any(f.startswith("app/") for f in changed):
        return
    cur = run("show", "HEAD:app/main.py")
    try:
        old = run("show", "origin/main:app/main.py", check=False)
    except SystemExit:
        return

    def ver(text: str) -> str:
        m = re.search(r'APP_VERSION\s*=\s*"([^"]+)"', text)
        return m.group(1) if m else ""

    if old and ver(cur) and ver(cur) == ver(old):
        fail(f"โค้ดใน app/ เปลี่ยนแต่ APP_VERSION ยังเป็น {ver(cur)} "
             f"· ขยับก่อน ไม่งั้นดูไม่ออกว่า deploy ทันหรือยัง")
    ok(f"APP_VERSION = {ver(cur)}")


def check_tests() -> None:
    for t in ("tests/test_cv.py", "tests/test_filters.py"):
        r = subprocess.run([sys.executable, t], capture_output=True, text=True,
                           env={"PYTHONUTF8": "1", **__import__("os").environ})
        if r.returncode:
            fail(f"{t} ไม่ผ่าน:\n{r.stdout[-800:]}")
        ok(f"{t} ผ่าน")


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
