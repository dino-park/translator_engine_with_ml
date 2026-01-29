from openai import OpenAI
from llama_index.core.embeddings import BaseEmbedding
from pydantic import PrivateAttr
import time
from utils import get_translator_logger


logger = get_translator_logger("core.embedding_model")


# -----------------------------
# Compass API Embedding wrapper (Compass API 전용)
# -----------------------------

class CompassEmbeddingModel(BaseEmbedding):
    """
    Compass API전용 임베딩 모델 래퍼 클래스
    - compass-embedding-v4 사용
    - text-type 파라미터 지원 (document/query)
    - Rate Limit 대응: Compass API는 180 RPM
    - API 응답에서 정확한 토큰 수를 추적
    """
    
    model_name: str = "compass-embedding-v4"
    dimensions: int = 2560
    embed_batch_size: int = 100
    
    _client: OpenAI = PrivateAttr()
    _total_tokens: int = PrivateAttr(default=0)
    
    def __init__(
        self,
        api_key: str,
        api_base: str,
        model: str = "compass-embedding-v4",
        dimensions: int = 2560,
        embed_batch_size: int = 100,
        **kwargs
    ):
        super().__init__(embed_batch_size=embed_batch_size, **kwargs)
        self._client = OpenAI(api_key=api_key, base_url=api_base)
        self.model_name = model
        self.dimensions = dimensions
    
    @property
    def total_tokens(self) -> int:
        return self._total_tokens
    
    def reset_token_count(self) -> None:
        self._total_tokens = 0
    
    @classmethod
    def class_name(cls) -> str:
        return "CompassEmbeddingModel"
    
    def _embed_with_text_type(self, texts: list[str], text_type: str, rate_limit_delay: bool = True) -> list[list[float]]:
        """
        text_type을 지정하여 임베딩 수행
        - text_type="document": 인덱싱 시 (문서 저장)
        - text_type="query": 검색 시
        - rate_limit_delay: True면 요청 후 2.5초 대기 (인덱싱용), False면 대기 없음 (쿼리용)
        - Rate Limit 대응: Compass API는 30 RPM
        """        
        max_retries = 5
        base_delay = 3.0  # 429 에러 시 초기 대기 시간 (초)
        
        for attempt in range(max_retries):
            try:
                response = self._client.embeddings.create(
                    input=texts,
                    model=self.model_name,
                    dimensions=self.dimensions,
                    extra_body={"text_type": text_type}  # Compass API 전용 파라미터
                )
                
                # 토큰 수 누적
                if response.usage and response.usage.total_tokens:
                    self._total_tokens += response.usage.total_tokens
                
                # 인덱싱 시에만 Rate limit 방지 딜레이 적용 (180 RPM = 0.33초 간격)
                if rate_limit_delay:
                    time.sleep(0.5)
                
                return [d.embedding for d in response.data]
                
            except Exception as e:
                if "429" in str(e) or "Rate limit" in str(e):
                    delay = base_delay * (2 ** attempt)  # Exponential backoff: 3, 6, 12, 24, 48초
                    delay = min(delay, 60)  # 최대 60초
                    logger.warning(f"Rate limit hit, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    raise  # 다른 에러는 그대로 발생
        
        raise Exception(f"Max retries ({max_retries}) exceeded for embedding request")
    
    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """배치 임베딩 (인덱싱용, text_type=document)"""
        return self._embed_with_text_type(texts, text_type="document")
    
    def _get_text_embedding(self, text: str) -> list[float]:
        """단일 임베딩 (인덱싱용, text_type=document)"""
        return self._get_text_embeddings([text])[0]
    
    def _get_query_embedding(self, query: str) -> list[float]:
        """쿼리 임베딩 (검색용, text_type=query, 딜레이 없음)"""
        return self._embed_with_text_type([query], text_type="query", rate_limit_delay=False)[0]
    
    async def _aget_query_embedding(self, query: str) -> list[float]:
        """비동기 쿼리 임베딩"""
        return self._get_query_embedding(query)
    