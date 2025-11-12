import os
import re
import csv
import time
from urllib.parse import urljoin
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

BASE = "https://linkareer.com"
LIST_URL = "https://linkareer.com/cover-letter/search?page={page}&sort=PASSED_AT&tab=all"

# === 저장 경로: 프로젝트 루트(/.../25-2_DArtB_Academic_Seminar)/data ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # python code 상위
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
OUT_CSV = os.path.join(DATA_DIR, "linkareer_coverletters_p111_140.csv")

print("[INFO] __file__:", __file__)                     # ⬇ 추가 (실행 파일 확인)
print("[INFO] DATA_DIR:", DATA_DIR)                     # ⬇ 추가 (폴더 확인)
print("[INFO] OUT_CSV:", OUT_CSV)                       # ⬇ 추가 (정확한 저장 경로)

def normalize_text(raw: str) -> str:
    tokens = re.findall(r'"(.*?)"', raw, flags=re.S)
    text = "".join(tokens) if tokens else raw
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def extract_links_from_list(page) -> list:
    anchors = page.eval_on_selector_all(
        "a[href]", "els => els.map(e => e.getAttribute('href'))"
    )
    ids = set()
    links = []
    for href in anchors:
        if not href:
            continue
        m = re.search(r"/cover-letter/(\d+)", href)
        if m:
            cid = m.group(1)
            if cid not in ids:
                ids.add(cid)
                links.append(urljoin(BASE, f"/cover-letter/{cid}"))
    return links

def extract_coverletter_text(page) -> str:
    selectors = [
        "article#coverLetterContent",
        "main.CoverLetterDetailContent__StyledMain-sc-177ad02f-0",
        "main",
    ]
    raw = ""
    for sel in selectors:
        try:
            raw = page.eval_on_selector(sel, "el => el.innerText")
            if raw and raw.strip():
                break
        except (PWTimeout, Exception):
            continue
    if not raw:
        try:
            raw = page.evaluate("() => document.body.innerText || ''")
        except Exception:
            raw = ""
    return normalize_text(raw)

def ensure_header(csv_path: str, fieldnames):
    need_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    if need_header:
        print("[INFO] Writing CSV header...")            # ⬇ 추가
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

def crawl():
    fieldnames = ["page", "id", "url", "length", "text"]
    ensure_header(OUT_CSV, fieldnames)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        for page_no in range(111, 141):
            list_url = LIST_URL.format(page=page_no)
            print(f"[LIST] visiting {list_url}")
            page.goto(list_url, wait_until="networkidle", timeout=60000)

            links = extract_links_from_list(page)
            print(f"  └─ found {len(links)} detail links")

            for idx, detail_url in enumerate(links, 1):
                try:
                    page.goto(detail_url, wait_until="networkidle", timeout=60000)
                    text = extract_coverletter_text(page)
                    cid = re.search(r"/cover-letter/(\d+)", detail_url).group(1)
                    row = {
                        "page": page_no,
                        "id": cid,
                        "url": detail_url,
                        "length": len(text),
                        "text": text
                    }

                    # ↙↙ 즉시 저장 + 에러 출력
                    try:
                        with open(OUT_CSV, "a", newline="", encoding="utf-8-sig") as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writerow(row)
                            f.flush()
                    except Exception as fe:
                        print(f"[ERROR] CSV write failed for id={cid}: {fe}")  # ⬇ 추가
                        raise

                    print(f"    [{idx}/{len(links)}] id={cid} len={len(text)} (saved)")
                    time.sleep(1.0)
                except Exception as e:
                    print(f"    [skip] {detail_url} -> {e}")
                    time.sleep(1.0)

            time.sleep(2.0)

        print(f"\n✅ Finished. CSV at: {OUT_CSV}")
        browser.close()

if __name__ == "__main__":
    crawl()