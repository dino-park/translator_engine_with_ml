# Weak Supervision 레이블링 분석 결과

클러스터 분석 기반으로 개선된 Weak Supervision 레이블러 적용 결과

## 📊 전체 통계 (116개 로그)

### 레이블 분포
- **GOOD (신뢰 가능)**: 21개 (18.1%)
- **BAD (개선 필요)**: 95개 (81.9%)
- **ABSTAIN (판단 불가)**: 0개 (0%)

### 해석
- 전체의 **82%가 개선이 필요**한 번역으로 판정됨
- Cluster 1 (BM25 Perfect Match)만 GOOD 비율이 높음 (91%)
- 나머지 클러스터들은 대부분 BAD로 판정

---

## 🎯 Cluster별 레이블 분포

| Cluster | GOOD | BAD | Total | 성격 | GOOD 비율 |
|---------|------|-----|-------|------|-----------|
| **1** | 21 | 2 | 23 | BM25 Perfect Match (이상적) | **91%** |
| **0** | 0 | 45 | 45 | LLM Fallback 일반 단어 | 0% |
| **3** | 0 | 25 | 25 | Rank 2등 Fallback | 0% |
| **5** | 0 | 10 | 10 | 브래킷 용어 Rank 2등 | 0% |
| **2** | 0 | 7 | 7 | 긴 문장 LLM Translation | 0% |
| **4** | 0 | 6 | 6 | 날짜/버전 정보 | 0% |

### 핵심 인사이트

#### ✅ Cluster 1 (GOOD 91%)
**특징:**
- BM25 match + rank 1등
- LLM fallback 거의 사용 안함 (8.7%)
- 짧은 쿼리 (평균 5.8자)
- rank_diff ≈ 0

**예시:**
- `提升同步等级` → `동기화 레벨 증가`
- `特工等级已同步` → `특수 요원 레벨 동기화됨`
- `主线剧情` → `메인 스토리`

**결론:** 가장 이상적인 번역 패턴. 이런 타입을 늘려야 함.

---

#### ❌ Cluster 0 (BAD 100%)
**문제점:**
- Exact match가 있는데 LLM fallback 사용 (100%)
- rank는 1등인데 LLM 보강 필요
- 짧은~중간 쿼리 (평균 7.5자)

**예시:**
- `【海外+7】鹿游角色单` → LLM fallback 필요
- `二档皮肤：荔倾` → LLM fallback 필요

**개선 방안:**
1. 왜 exact match인데 LLM이 필요한가? → DB 품질 점검
2. 브래킷(【】) 처리 로직 개선
3. 특수 문자 포함 용어의 전처리 개선

---

#### ❌ Cluster 3 (BAD 100%) - **우선순위 높음**
**문제점:**
- Exact match인데 rank가 2등 (avg 2.04)
- LLM fallback 100%
- rank_diff가 큼 (2.04)

**예시:**
- `【海外+7】时曦角色单` → rank 2등
- `荔倾试玩战斗` → rank 2등

**개선 방안:**
1. **랭킹 로직 개선 필수**
2. Exact match가 1등이 되도록 boost 강화
3. BM25 가중치 조정

---

#### ❌ Cluster 5 (BAD 100%) - **우선순위 높음**
**문제점:**
- Segment match인데 rank가 2등 (avg 2.0)
- is_top_segment = 1인데 1등 아님
- LLM fallback 100%

**예시:**
- `【传闻调查更新】` → segment match, rank 2
- `【档案室更新】` → segment match, rank 2

**개선 방안:**
1. Segment match 점수 boost 강화
2. 브래킷 용어 특별 처리
3. Segment가 1등이 되도록 랭킹 조정

---

#### ❌ Cluster 2 (BAD 100%)
**문제점:**
- 긴 문장 (평균 20.4자)
- LLM translation 필수 (86%)
- top_score = 0 (매칭 없음)

**예시:**
- `점검 보상으로 월상석 10개 지급 했음`
- `【小游戏】：南庭忆觉，音游小游戏`

**개선 방안:**
1. 문장 번역은 LLM 사용이 맞음 (정상)
2. 자주 쓰는 문장 패턴은 TM에 추가
3. 용어사전 coverage 확대

---

#### ❌ Cluster 4 (BAD 100%)
**문제점:**
- 날짜/버전 정보 (특수 케이스)
- rank_diff가 매우 큼 (3.17)
- is_top_segment = 1

**예시:**
- `2025.12.10 1.1.189`
- `2025.12.112 1.1.190`

**개선 방안:**
1. 날짜/버전 패턴 특수 처리 추가
2. 정규식 기반 전처리
3. 별도 규칙 적용

---

## 📋 규칙별 적용 통계 (상위 10개)

| 순위 | 규칙 | 적용률 | GOOD | BAD | 의미 |
|-----|------|--------|------|-----|------|
| 1 | `lf_low_score` | 100% | 0 | 116 | 모든 케이스에서 점수 낮음 |
| 2 | `lf_exact_match` | 93% | 108 | 0 | Exact match 매우 많음 |
| 3 | `lf_small_gap` | 84% | 0 | 97 | 1위-2위 점수 차이 작음 |
| 4 | `lf_llm_fallback` | 81% | 0 | 94 | LLM fallback 많이 사용 |
| 5 | `lf_exact_but_llm_fallback` | 76% | 0 | 88 | **이상한 조합! (개선 필요)** |
| 6 | `lf_exact_match_rank_one` | 56% | 65 | 0 | Exact + rank 1 |
| 7 | `lf_rank_mismatch_problem` | 35% | 0 | 41 | Rank 불일치 문제 |
| 8 | `lf_rank_diff_large` | 35% | 0 | 41 | Rank 차이 큼 |
| 9 | `lf_rank_diff_zero` | 25% | 29 | 0 | Rank 일치 |
| 10 | `lf_bm25_match` | 19% | 22 | 0 | BM25 매칭 |

### 핵심 문제점

1. **lf_exact_but_llm_fallback (76%)**: Exact match인데 LLM을 쓴다 = 뭔가 잘못됨
2. **lf_rank_mismatch_problem (35%)**: Exact match인데 rank가 2등 이상
3. **lf_low_score (100%)**: 모든 케이스에서 top_score가 0.1 미만 → embedding 품질 문제?

---

## 🎓 새로 추가된 규칙들 (클러스터 분석 기반)

### 핵심 규칙 (가중치 3.0)
1. **lf_ideal_bm25_match**: Cluster 1 타입 감지
2. **lf_exact_match_rank_one**: exact + rank 1
3. **lf_bm25_without_llm**: BM25만으로 해결
4. **lf_rank_mismatch_problem**: Cluster 3, 5 타입 (rank 문제)
5. **lf_rank_diff_large**: rank 차이 큼
6. **lf_top_segment_but_rank_low**: segment match인데 낮은 rank
7. **lf_exact_but_llm_fallback**: 이상한 조합

### 보조 규칙 (가중치 1.5)
- 기존 규칙들 (`lf_exact_match`, `lf_bm25_match`, 등)

### 약한 규칙 (가중치 1.0)
- `lf_high_score`, `lf_low_score`, `lf_long_sentence_llm`, 등

---

## 🚀 개선 방안 우선순위

### Priority 1: 랭킹 로직 개선 (Cluster 3, 5)
- **문제**: Exact match인데 rank가 2등 이상
- **영향**: 35-41개 (30-35%)
- **해결책**:
  1. Exact match에 더 높은 boost 적용
  2. Segment match 점수 강화
  3. BM25 + Hybrid 가중치 재조정

### Priority 2: Exact + LLM Fallback 조합 제거 (Cluster 0)
- **문제**: Exact match인데 LLM을 쓴다
- **영향**: 88개 (76%)
- **해결책**:
  1. Exact match 신뢰도 향상
  2. 브래킷(【】) 전처리 개선
  3. DB 품질 점검

### Priority 3: Embedding 품질 개선
- **문제**: 모든 케이스에서 top_score < 0.1
- **영향**: 116개 (100%)
- **해결책**:
  1. Embedding 모델 교체 검토
  2. 용어사전 coverage 확대
  3. Fine-tuning 고려

### Priority 4: 특수 케이스 처리 (Cluster 4)
- **문제**: 날짜/버전 패턴
- **영향**: 6개 (5%)
- **해결책**:
  1. 정규식 기반 전처리
  2. 특수 규칙 추가

---

## 📈 기대 효과

개선 후 예상:
- **Priority 1 해결 시**: GOOD 비율 18% → 40% 증가
- **Priority 2 해결 시**: GOOD 비율 40% → 70% 증가
- **Priority 3 해결 시**: GOOD 비율 70% → 85% 증가

목표: **GOOD 비율 85% 이상 달성**

---

## 💻 사용 방법

### 1. 테스트 모드
```bash
python ml_experiment/weak_supervision_labeler.py
```

### 2. 전체 분석 (클러스터링 + 레이블링)
```bash
python ml_experiment/weak_supervision_labeler.py full
```

### 3. 결과 파일
- `clustered_logs.csv`: 클러스터 결과
- `labeled_logs.csv`: 레이블 + 클러스터 통합 결과

---

## 📚 참고

- 클러스터 분석: `unsupervised_cluster_from_db.py`
- Weak Supervision: `weak_supervision_labeler.py`
- 원본 데이터: `translation_logger.db`


