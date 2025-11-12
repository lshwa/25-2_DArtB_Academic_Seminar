# python_code/benchmark.py
# -*- coding: utf-8 -*-
"""
Essay_data.csv 에 담긴 자기소개서(text 열)를 읽어
GPT-Killer식 5대 지표 + 세부 특징들을 산출하여
data/result/Essay_metrics.csv 로 저장한다.

지표 목록(최종 컬럼):
- style_uniformity_score      : 문체 일정성 (높을수록 '너무 일정함')
- narrative_monotony_score    : 전개 단조성 (높을수록 '단조로움')
- abstractness_score          : 구체성 결여 (높을수록 '모호함')
- rhythm_uniformity_score     : 리듬 일정성 (높을수록 '리듬 일정')
- global_simplicity_score     : 전체 단순성 (높을수록 '단순')

세부 특징(설명용):
- sent_len_mean, sent_len_std, sent_len_cv
- token_len_mean, token_len_std
- ttr, type_count, token_count
- vague_count, vague_ratio
- concrete_num_count, proper_noun_proxy_count
- ending_diversity, ending_top3
- pos_proxy_noun_ratio, pos_proxy_verb_ratio, pos_proxy_adj_ratio
- trans_diversity, trans_ratio
- para_count, para_sim_mean
- seg_style_sim_mean (5분할 구간 평균 유사도)
- comma_total, comma_per_sent, comma_relpos_mean, comma_relpos_std

실행:
  python python_code/benchmark.py
"""

import os
import re
import json
import math
import logging
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# sklearn은 광범위 환경에서 기본적으로 많이 설치되어 있음.
# 미설치 환경 대비 try/except로 우회
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SK_OK = True
except Exception:
    SK_OK = False


# ------------------------------
# 경로/로깅 설정
# -> 여기서 csv 파일 이름만 바꾸고 실행하면 됩니다. 
# ------------------------------
ROOT = Path(__file__).resolve().parents[1]  # 프로젝트 루트
DATA_DIR = ROOT / "data"
IN_CSV = DATA_DIR / "Essay_data.csv"
OUT_DIR = DATA_DIR / "result"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "Essay_metrics.csv"
LOG_FILE = OUT_DIR / "run.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ],
)


# ------------------------------
# 규칙/사전
# ------------------------------
VAGUE_WORDS = set("""
부분 측면 요소 과정 경우 문제 역량 노력 기반 중심 방향 적정 적절 중요 핵심
개선 강화 확대 추진 마련 진행 적용 활용 변화 영향 고려 방안 효과
이슈 사항 수준 형태 역할 경험 가치 의미 목표 계획 전략 결과 성과
""".split())

TRANSITION_WORDS = set("""
그러나 하지만 또한 따라서 그러므로 한편 반면 먼저 다음으로 게다가
다만 또한 뿐만 아니라 더불어 결국 즉 즉시 이후 동시에 동시에또한
""".split())

# 간단한 종결어미 후보(마지막 형태/어미 다양도 측정용)
ENDING_REGEX = re.compile(r'([가-힣]{1,4})(?:\s*[.!?…」”\)」\]]*)$')

# 문장 분할(한국어+기본 구두점)
SENT_SPLIT = re.compile(r'(?:[.!?…]+|[다요죠까니]\s*)(?=\s|$)')
# 단어 토큰(한글/영문/숫자)
TOKEN_RE = re.compile(r'[A-Za-z]+|[0-9]+|[가-힣]+')

# "고유명사 proxy" 판단: 대문자 시작 영단어, 전부 대문자 약어, 숫자/연도 포함 토큰 등
def is_proper_noun_proxy(tok: str) -> bool:
    if re.match(r'^[A-Z][a-z]+$', tok):  # Samsung, Hyundai
        return True
    if re.match(r'^[A-Z]{2,}$', tok):    # AI, NLP
        return True
    if re.search(r'\d', tok):            # 2024, 3D, MZ
        return True
    # 대표적인 한국어 고유명사 접미(대략적) 감지
    if re.search(r'(대학교|중학교|은행|공단|전자|생명|바이오|백화점|호텔|공사|연구소|센터|지점|팀)$', tok):
        return True
    return False


# ------------------------------
# 유틸 함수
# ------------------------------
def split_sentences(text: str):
    text = (text or "").strip()
    if not text:
        return []
    # 줄바꿈을 점으로 대략 치환 후 split, 과분할 방지 위해 후처리
    tmp = re.sub(r'\n+', '. ', text)
    parts = [s.strip() for s in SENT_SPLIT.split(tmp) if s and s.strip()]
    # 너무 길게 합쳐졌거나 빈 부분 보정
    sents = []
    for s in parts:
        s2 = s.strip()
        if s2:
            sents.append(s2)
    return sents

def tokenize(text: str):
    return TOKEN_RE.findall(text or "")

def char_len_stats(sents):
    lengths = [len(s) for s in sents] if sents else []
    if not lengths:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(lengths))
    std = float(np.std(lengths))
    cv = float(std / mean) if mean > 0 else 0.0
    return mean, std, cv

def token_len_stats(tokens):
    lengths = [len(t) for t in tokens] if tokens else []
    if not lengths:
        return 0.0, 0.0
    return float(np.mean(lengths)), float(np.std(lengths))

def type_token_ratio(tokens):
    if not tokens:
        return 0.0, 0, 0
    types = len(set(tokens))
    total = len(tokens)
    return float(types / total) if total else 0.0, types, total

def count_vague(tokens):
    toks_lower = [t.lower() for t in tokens]
    cnt = sum(1 for t in toks_lower if t in VAGUE_WORDS)
    return cnt, (cnt / len(tokens)) if tokens else 0.0

def count_concrete_signals(text, tokens):
    num_cnt = len(re.findall(r'\d', text or ""))
    proper_cnt = sum(1 for t in tokens if is_proper_noun_proxy(t))
    return num_cnt, proper_cnt

def ending_diversity(sents):
    ends = []
    for s in sents:
        m = ENDING_REGEX.search(s)
        if m:
            tail = m.group(1)[-2:]  # 마지막 1~2글자 정도
            ends.append(tail)
    if not ends:
        return 0.0, ""
    uniq = len(set(ends))
    div = uniq / len(ends)
    top3 = ",".join([f"{k}:{v}" for k, v in Counter(ends).most_common(3)])
    return float(div), top3

def pos_proxy_ratios(tokens):
    # 매우 거친 proxy:
    # - '다/했다/한다/됩니다/합니다' 등의 동사/서술어 흔적이 있는 토큰 수를 verb proxy로,
    # - 형용사 흔적: '하다/로운/스러운/같은/적인'
    # - 나머지 한글명사 후보(영문/숫자 제외)를 noun proxy로
    if not tokens:
        return 0.0, 0.0, 0.0
    n_total = len(tokens)
    hangul = [t for t in tokens if re.match(r'^[가-힣]+$', t)]
    noun_like = [t for t in hangul if not re.search(r'(하다|했다|한다|됩니다|합니다|적인|스러운|같은|로운)$', t)]
    verb_like = [t for t in hangul if re.search(r'(하다|했다|한다|됩니다|합니다)$', t)]
    adj_like  = [t for t in hangul if re.search(r'(적인|스러운|같은|로운)$', t)]
    return (
        len(noun_like)/n_total,
        len(verb_like)/n_total,
        len(adj_like)/n_total
    )

def transition_stats(tokens):
    if not tokens:
        return 0.0, 0.0
    toks = [t for t in tokens]
    tr = [t for t in toks if t in TRANSITION_WORDS]
    div = len(set(tr))/len(tr) if tr else 0.0
    ratio = len(tr)/len(toks)
    return float(div), float(ratio)

def paragraph_similarity(text):
    paras = [p.strip() for p in re.split(r'\n\s*\n+', text or "") if p.strip()]
    if not paras:
        # 문장 길이 기준으로 임의 2~3문장씩 묶어 파라 대체
        sents = split_sentences(text or "")
        if len(sents) <= 1:
            return 1, 1.0  # 파라 1개, 유사도 1.0으로 간주
        paras = []
        step = max(2, len(sents)//3)
        for i in range(0, len(sents), step):
            paras.append(" ".join(sents[i:i+step]))
    if len(paras) == 1:
        return 1, 1.0
    if SK_OK:
        vec = TfidfVectorizer(min_df=1).fit_transform(paras)
        sim = cosine_similarity(vec)
        # 인접 유사도 평균
        sims = []
        for i in range(len(paras)-1):
            sims.append(sim[i, i+1])
        return len(paras), float(np.mean(sims)) if sims else 1.0
    else:
        # 스키킷런이 없으면 자모 수 기반 매우 단순 유사도
        def sim_char(a,b):
            sa, sb = set(a), set(b)
            return len(sa & sb) / max(1, len(sa | sb))
        sims = []
        for i in range(len(paras)-1):
            sims.append(sim_char(paras[i], paras[i+1]))
        return len(paras), float(np.mean(sims)) if sims else 1.0

def segment_style_similarity(text, nseg=5):
    sents = split_sentences(text or "")
    if not sents:
        return 1.0
    segs = []
    step = max(1, len(sents)//nseg)
    for i in range(0, len(sents), step):
        segs.append(sents[i:i+step])
        if len(segs) == nseg:
            break
    # 각 구간의 스타일 벡터 구성
    feat = []
    for ss in segs:
        toks = tokenize(" ".join(ss))
        mean_len, std_len, cv = char_len_stats(ss)
        ttr, _, _ = type_token_ratio(toks)
        end_div, _ = ending_diversity(ss)
        vcnt, vrat = count_vague(toks)
        trans_div, trans_ratio = transition_stats(toks)
        feat.append([
            mean_len, std_len, cv, ttr, end_div, vrat, trans_div, trans_ratio
        ])
    if len(feat) == 1:
        return 1.0
    X = np.array(feat, dtype=float)
    # 코사인 유사도 평균(인접)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    sims = []
    for i in range(len(Xn)-1):
        sims.append(float(np.dot(Xn[i], Xn[i+1])))
    return float(np.mean(sims)) if sims else 1.0

def comma_features(text):
    sents = split_sentences(text or "")
    if not sents:
        return 0, 0.0, 0.0, 0.0
    total = text.count(",")
    cps = [s.count(",") for s in sents]
    per_sent = float(np.mean(cps))
    # 콤마 상대 위치: 각 문장 내 콤마 인덱스 / 문장길이
    relpos = []
    for s in sents:
        idxs = [m.start() for m in re.finditer(",", s)]
        L = max(1, len(s))
        relpos.extend([i/L for i in idxs])
    mean = float(np.mean(relpos)) if relpos else 0.0
    std = float(np.std(relpos)) if relpos else 0.0
    return total, per_sent, mean, std


# ------------------------------
# 지표(스코어) 집계
# * 점수는 "높을수록 GPT-like" 방향으로 정규화
# ------------------------------
def compute_metrics(text: str):
    sents = split_sentences(text)
    toks = tokenize(text)

    # 기본 통계
    sent_mean, sent_std, sent_cv = char_len_stats(sents)
    tlen_mean, tlen_std = token_len_stats(toks)
    ttr, type_cnt, token_cnt = type_token_ratio(toks)

    vague_cnt, vague_ratio = count_vague(toks)
    num_cnt, proper_cnt = count_concrete_signals(text, toks)

    end_div, end_top3 = ending_diversity(sents)
    pnoun_ratio, pverb_ratio, padj_ratio = pos_proxy_ratios(toks)
    trans_div, trans_ratio = transition_stats(toks)
    para_cnt, para_sim_mean = paragraph_similarity(text)
    seg_sim_mean = segment_style_similarity(text, nseg=5)
    comma_total, comma_per, comma_pos_mean, comma_pos_std = comma_features(text)

    # 1) 문체 일정성: 구간 스타일 유사도(평균) -> 높을수록 일정
    style_uniformity = seg_sim_mean  # [0~1] 쪽으로 수렴

    # 2) 전개 단조성: 문단 인접 유사도 -> 높을수록 단조
    narrative_monotony = para_sim_mean

    # 3) 구체성 결여: 모호어 비율↑, 숫자/고유명사 신호↓
    #    간단 결합: vague_ratio - 0.5*concrete_ratio
    concrete_ratio = (num_cnt + proper_cnt) / max(1, token_cnt)
    abstractness = max(0.0, min(1.0, vague_ratio - 0.5 * concrete_ratio))

    # 4) 리듬 일정성: 문장길이 변동계수(CV) 낮을수록 일정
    rhythm_uniformity = 1.0 - max(0.0, min(1.0, min(1.0, sent_cv)))  # CV를 [0,1]로 클리핑 후 반전

    # 5) 전체 단순성: (1 - TTR) + (낮은 복문성)
    #    복문 proxy: 콤마/접속부사 비율
    compound_proxy = min(1.0, (comma_per + trans_ratio))  # 많을수록 '복문'
    global_simplicity = max(0.0, min(1.0, (1.0 - ttr) + (1.0 - compound_proxy) * 0.5))

    row = dict(
        sent_len_mean=sent_mean,
        sent_len_std=sent_std,
        sent_len_cv=sent_cv,
        token_len_mean=tlen_mean,
        token_len_std=tlen_std,
        ttr=ttr, type_count=type_cnt, token_count=token_cnt,
        vague_count=vague_cnt, vague_ratio=vague_ratio,
        concrete_num_count=num_cnt, proper_noun_proxy_count=proper_cnt,
        ending_diversity=end_div, ending_top3=end_top3,
        pos_proxy_noun_ratio=pnoun_ratio,
        pos_proxy_verb_ratio=pverb_ratio,
        pos_proxy_adj_ratio=padj_ratio,
        trans_diversity=trans_div, trans_ratio=trans_ratio,
        para_count=para_cnt, para_sim_mean=para_sim_mean,
        seg_style_sim_mean=seg_sim_mean,
        comma_total=comma_total, comma_per_sent=comma_per,
        comma_relpos_mean=comma_pos_mean, comma_relpos_std=comma_pos_std,

        style_uniformity_score=style_uniformity,
        narrative_monotony_score=narrative_monotony,
        abstractness_score=abstractness,
        rhythm_uniformity_score=rhythm_uniformity,
        global_simplicity_score=global_simplicity,
    )
    return row


# ------------------------------
# 메인 실행
# ------------------------------
def main():
    if not IN_CSV.exists():
        logging.error(f"입력 파일이 없습니다: {IN_CSV}")
        return

    logging.info(f"입력 로드: {IN_CSV}")
    df_iter = pd.read_csv(IN_CSV, encoding="utf-8", chunksize=200, dtype=str)

    wrote_header = False
    total_rows = 0
    for chunk_id, cdf in enumerate(df_iter, start=1):
        cdf = cdf.fillna("")
        results = []
        for i, row in cdf.iterrows():
            text = row.get("text", "")
            try:
                feats = compute_metrics(text)
            except Exception as e:
                logging.exception(f"행 처리 중 오류 (index={i}): {e}")
                feats = {k: np.nan for k in [
                    "sent_len_mean","sent_len_std","sent_len_cv",
                    "token_len_mean","token_len_std",
                    "ttr","type_count","token_count",
                    "vague_count","vague_ratio",
                    "concrete_num_count","proper_noun_proxy_count",
                    "ending_diversity","ending_top3",
                    "pos_proxy_noun_ratio","pos_proxy_verb_ratio","pos_proxy_adj_ratio",
                    "trans_diversity","trans_ratio",
                    "para_count","para_sim_mean",
                    "seg_style_sim_mean",
                    "comma_total","comma_per_sent","comma_relpos_mean","comma_relpos_std",
                    "style_uniformity_score","narrative_monotony_score","abstractness_score",
                    "rhythm_uniformity_score","global_simplicity_score"
                ]}
            results.append(feats)

        feat_df = pd.DataFrame(results)
        out_chunk = pd.concat([cdf.reset_index(drop=True), feat_df], axis=1)

        # 첫 저장은 헤더 포함, 이후 append
        mode = "w" if not wrote_header else "a"
        header = not wrote_header
        out_chunk.to_csv(OUT_CSV, index=False, mode=mode, header=header, encoding="utf-8")
        wrote_header = True

        total_rows += len(out_chunk)
        logging.info(f"[{chunk_id}] 누적 저장: {total_rows}행 -> {OUT_CSV}")

    logging.info("완료")


if __name__ == "__main__":
    main()