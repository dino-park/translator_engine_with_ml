"""
번역 로그 DB - 학습 데이터 수집용

목적:
- 번역 요청/결과를 SQLite에 저장
- 나중에 ML 모델 학습에 활용
- Weak Supervision을 위한 피처 저장
"""

import sqlite3
import json
from pathlib import Path
from core.normalizer import normalize_for_exact_match

# DB 파일 경로 설정
db_dir = Path(__file__).parent / "database"
db_dir.mkdir(exist_ok=True)
DB_PATH = db_dir / "translation_logger.db"


class TranslationLoggerDB:
    """
    번역 로그를 SQLite DB에 저장하는 클래스
    
    사용법:
        from translation_logger_db import translation_logger_db
        translation_logger_db.log_translation(result, source="api")
    """
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_table()
    
    def _init_table(self):
        """
        테이블 생성 (없으면 새로 만듦)
        
        컬럼 설명:
        - query: 사용자 입력 쿼리
        - query_len: 쿼리 길이 (짧으면 용어, 길면 문장)
        - src_lang: 원본 언어 (cn/ko)
        - mode: term(용어) 또는 sentence(문장)
        - translation: 번역 결과
        - reason: 번역 방식 (exact_match, llm_fallback 등)
        - top_score: 검색 1위 점수 (높을수록 매칭 신뢰도 높음)
        - candidate_gap: 1위-2위 점수 차이 (클수록 확실한 매칭)
        - is_exact_match: 정확 매칭 여부 (가장 신뢰도 높음)
        - is_bm25_match: BM25(키워드)에서 매칭됨 (키워드 일치)
        - bm25_exact_rank: BM25에서 정확매칭 순위 (1이면 BM25도 1위)
        - bm25_hybrid_rank: Hybrid에서의 순위 (Vector 영향 확인)
        - is_llm_fallback: LLM으로 번역했는지 (용어사전에 없었음)
        - source: 요청 출처 (api/debug/batch)
        (추가함)
        - top_doc_type: top candidate가 어떤 타입인지(cn/cn_segment/cn_normalized/ko 등)
        - is_segment_exact_match: top candidate가 cn_segment인 경우, 정확 매칭 여부
        - segment_parent_cn: top candidate가 cn_segment인 경우 원문(parent_cn)
        """
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS translation_logger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                -- 입력 정보
                query TEXT,
                query_len INTEGER,
                src_lang TEXT,
                mode TEXT,
                
                -- 출력 정보
                translation TEXT,
                reason TEXT,
                
                -- 검색 점수 (ML 피처로 사용)
                top_score REAL,
                candidate_gap REAL,
                candidates_json TEXT,
                
                -- 매칭 유형 (ML 피처 + Weak Label 생성용)
                is_exact_match BOOLEAN,
                is_bm25_match BOOLEAN,
                is_llm_fallback BOOLEAN,
                
                -- BM25 분석 정보 (Vector vs BM25 비교)
                bm25_exact_rank INTEGER,
                bm25_hybrid_rank INTEGER,
                
                -- segment/metadata 기반 feature
                top_doc_type TEXT,
                is_segment_exact_match INTEGER,
                segment_parent_cn TEXT,
                
                -- glossary hints 관련 (1단계 대응)
                has_glossary_hints BOOLEAN,
                glossary_match_count INTEGER,
                
                -- 신뢰도 체크 관련 (1단계 대응)
                passed_bm25_check BOOLEAN,
                passed_gap_check BOOLEAN,
                
                -- 성능 관련
                response_time_ms INTEGER,
                
                -- 메타 정보
                source TEXT DEFAULT 'api'
            )
        """)
        self.conn.commit()
    
    def log_translation(self, result: dict, source: str = "api"):
        """
        번역 결과를 DB에 저장
        
        Args:
            result: translate_term() 또는 translate_sentence_with_glossary()의 반환값
            source: 요청 출처 ("api", "debug", "batch")
        """
        # 후보 정보 추출
        candidates = result.get("candidates", []) or []
        top_score = candidates[0]["score"] if candidates else None
        candidate_gap = (candidates[0]["score"] - candidates[1]["score"]) if len(candidates) > 1 else None
        
        # 매칭 유형 판단
        reason = result.get("reason", "")
        is_exact_match = "exact_match" in reason
        is_bm25_match = False  # 현재 reason에는 bm25_match가 없음, 아래에서 bm25_analysis로 판단
        is_llm_fallback = "llm" in reason.lower()
        
        # BM25 분석 정보 추출
        bm25 = result.get("bm25_analysis") or {}
        bm25_exact_rank = bm25.get("bm25_exact_rank")
        bm25_hybrid_rank = bm25.get("bm25_hybrid_rank")
        
        # BM25에서 정확 매칭이 있으면 is_bm25_match = True
        if bm25_exact_rank is not None:
            is_bm25_match = True
            
        # ========================
        # segment/metadata 기반 feature 계산
        # ========================
        top_doc_type = ""
        segment_parent_cn = ""
        is_segment_exact_match = 0
        
        if candidates:
            top = candidates[0]
            
            # candidate에서 doc_type을 얻는 방법이 두가지 있음:
            # 1. candidate의 metadata에서 doc_type을 얻음
            # 2. candidate의 matched_by 필드에서 doc_type을 얻음
            metadata = top.get("metadata", {})
            top_doc_type = metadata.get("doc_type", "") or top.get("matched_by", "") or ""
            
            segment_parent_cn = (
                metadata.get("parent_cn")
                or top.get("parent_cn")
                or top.get("segment_parent_cn")
                or ""
            )
            if top_doc_type == "cn_segment":
                normalized_query = normalize_for_exact_match(result.get("query", ""))
                top_cn_normalized = normalize_for_exact_match(top.get("cn", ""))
                if normalized_query and top_cn_normalized and normalized_query == top_cn_normalized:
                    is_segment_exact_match = 1
        
        # ========================
        # glossary hints 관련 (1단계 대응)
        # ========================
        glossary_hints = result.get("glossary_hints")
        has_glossary_hints = glossary_hints is not None
        glossary_match_count = 0
        if glossary_hints and isinstance(glossary_hints, dict):
            matches = glossary_hints.get("matches", [])
            glossary_match_count = len(matches)
        
        # ========================
        # 신뢰도 체크 관련 (1단계 대응)
        # ========================
        passed_bm25_check = result.get("passed_bm25_check")
        passed_gap_check = result.get("passed_gap_check")
        
        # ========================
        # 성능 관련
        # ========================
        response_time_ms = result.get("response_time_ms")
        
        # DB에 저장
        self.conn.execute("""
            INSERT INTO translation_logger 
            (query, query_len, src_lang, mode, translation, reason,
             top_score, candidate_gap, candidates_json,
             is_exact_match, is_bm25_match, is_llm_fallback,
             bm25_exact_rank, bm25_hybrid_rank, 
             top_doc_type, is_segment_exact_match, segment_parent_cn,
             has_glossary_hints, glossary_match_count,
             passed_bm25_check, passed_gap_check, response_time_ms,
             source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.get("query"),
            len(result.get("query", "")),
            result.get("src_lang"),
            result.get("mode"),
            result.get("translation"),
            reason,
            top_score,
            candidate_gap,
            json.dumps(candidates, ensure_ascii=False),
            is_exact_match,
            is_bm25_match,
            is_llm_fallback,
            bm25_exact_rank,
            bm25_hybrid_rank,
            top_doc_type,
            is_segment_exact_match,
            segment_parent_cn,
            has_glossary_hints,
            glossary_match_count,
            passed_bm25_check,
            passed_gap_check,
            response_time_ms,
            source
        ))
        self.conn.commit()
    
    def get_all_logs(self) -> list[dict]:
        """
        모든 로그를 딕셔너리 리스트로 반환 (ML 학습용)
        """
        cursor = self.conn.execute("SELECT * FROM translation_logger")
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        return [dict(zip(columns, row)) for row in rows]
    
    def get_logs_as_dataframe(self):
        """
        pandas DataFrame으로 반환 (ML 학습용)
        """
        import pandas as pd
        return pd.read_sql_query("SELECT * FROM translation_logger", self.conn)


# 싱글톤 인스턴스 생성 (앱 전체에서 하나만 사용)
translation_logger_db = TranslationLoggerDB()
