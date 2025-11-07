import re
import csv
import time
from urllib.parse import urljoin
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

BASE = "https://linkareer.com"
LIST_URL = "https://linkareer.com/cover-letter/search?page={page}&sort=PASSED_AT&tab=all"
OUT_CSV = "linkareer_coverletters_p103_110.csv"

def normalize_text(raw: str) -> str:
    tokens = re.findall(r'"(.*?)"', raw, flags=re.S)
    if tokens:
        text = "".join(tokens)
    else:
        text = raw

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def extract_links_from_list(page) -> list:

    # 목록 페이지의 모든 a[href] 중 /cover-letter/숫자 패턴만 추출
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
                # 정규화된 상세 링크(페이지 파라미터는 유지해도 되고 생략해도 됨)
                links.append(urljoin(BASE, f"/cover-letter/{cid}"))
    return links

def extract_coverletter_text(page) -> str:
    selectors = [
        "article#coverLetterContent",            # 스크린샷 기준 최유력
        "main.CoverLetterDetailContent__StyledMain-sc-177ad02f-0",  # 클래스 기반
        "main",                                  # 폴백
    ]
    raw = ""
    for sel in selectors:
        try:
            raw = page.eval_on_selector(sel, "el => el.innerText")
            if raw and len(raw.strip()) > 0:
                break
        except PWTimeout:
            continue
        except Exception:
            continue

    if not raw:
        # 최후 폴백: body 전체 텍스트
        try:
            raw = page.evaluate("() => document.body.innerText || ''")
        except Exception:
            raw = ""

    return normalize_text(raw)

def crawl():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        rows = []
        # 여기의 page 넘버를 바꿔야 함. 여기서는 103부터 111로 되어있음. 
        for page_no in range(103, 111):
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
                    rows.append({
                        "page": page_no,
                        "id": cid,
                        "url": detail_url,
                        "length": len(text),
                        "text": text
                    })
                    print(f"    [{idx}/{len(links)}] id={cid} len={len(text)}")
                    time.sleep(1.2)  # 과도한 요청 방지
                except Exception as e:
                    print(f"    [skip] {detail_url} -> {e}")
                    time.sleep(1.2)

            # 페이지 사이 간격
            time.sleep(2.0)

        # CSV 저장 (UTF-8 BOM: 엑셀 호환 좋음)
        with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["page", "id", "url", "length", "text"])
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nSaved {len(rows)} cover letters → {OUT_CSV}")
        browser.close()

if __name__ == "__main__":
    crawl()