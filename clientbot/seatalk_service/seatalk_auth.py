from typing import Optional
from dotenv import load_dotenv
import os, time, asyncio, httpx, logging


load_dotenv()
SEATALK_APP_ID = os.getenv("APP_ID")
SEATALK_APP_SECRET = os.getenv("APP_SECRET")
SEATALK_TOKEN_URL = os.getenv("SEATALK_TOKEN_URL")

class SeatalkAuth:
    def __init__(self):
        self._access_token: Optional[str] = None
        self._expires_at: float = 0.0
        self._lock = asyncio.Lock()
    
    def _is_valid(self) -> bool:
        # ----- 만료 60초 전부터 재발급 시도 -----
        return self._access_token and (time.time() < self._expires_at - 60)  # 60초 여유
    
    
    async def _fetch_new_token(self) -> None:
        headers = {"Content-Type": "application/json"}
        payload = {
                "app_id": SEATALK_APP_ID,
                "app_secret": SEATALK_APP_SECRET
            }
        async with httpx.AsyncClient(timeout=10) as client:
            res = await client.post(SEATALK_TOKEN_URL, json=payload, headers=headers)
            res.raise_for_status()
            res_data = res.json()
        
        self._access_token = res_data["app_access_token"]
        self._expires_at = int(res_data["expire"])  # 절대 만료 시각(Unix time)
        return self._access_token
    
    async def get_token(self) -> str:
        if self._is_valid():
            # ----- Token이 유효(만료 X)하다면, API요청하지 않고, 캐싱된 토큰 반환 -----
            return self._access_token  # type: ignore
        
        async with self._lock:
            if self._is_valid():
                return self._access_token  # 다른 코루틴이 이미 갱신했을 수도 있으니 재확인
            await self._fetch_new_token()
            return self._access_token
        
seatalk_auth = SeatalkAuth()
    