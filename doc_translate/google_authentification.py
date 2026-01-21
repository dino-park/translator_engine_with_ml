import logging

from pathlib import Path
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request


secret_dir = Path(__file__).parent / "config_file"
secret_dir.mkdir(exist_ok=True)

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def _setup_spreadsheet_logger():
    """spreadsheet 전용 로거 설정 - log/spreadsheet.log에 별도로 기록"""
    log_dir = Path(__file__).parent.parent / "log"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "spreadsheet.log"
    
    logger = logging.getLogger("spreadsheet")
    
    # 이미 핸들러가 있으면 중복 추가 방지
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # 루트 로거로 전파 안 함 (별도 파일에만 기록)
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter("%(asctime)s - [%(levelname)s] - %(name)s - %(message)s"))
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    
    return logger


logger = _setup_spreadsheet_logger()


def get_credentials_first_time():
    flow = InstalledAppFlow.from_client_secrets_file(
        secret_dir / "client_secret.json",
        SCOPES,
    )
    creds = flow.run_local_server(port=0)
    
    with open(secret_dir / "token.json", "w") as token:
        token.write(creds.to_json())
    
    logger.info("Credentials saved to token.json")
    
    return creds


def get_access_token():
    """
    Access token 발급
    Returns:
        str: Access token
    """
    creds = Credentials.from_authorized_user_file(
        secret_dir / "token.json",
        SCOPES,
    )
    if not creds or not creds.valid:
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open(secret_dir / "token.json", "w") as token:
                token.write(creds.to_json())
            logger.info("Credentials refreshed and saved to token.json")
        else:
            logger.error("Invalid credentials or expired token - need to re-authenticate")
            return None
    return f"Bearer {creds.token}"


if __name__ == "__main__":
    get_credentials_first_time()    # 최초 인증 시 실행
    access_token = get_access_token()
    logger.info(f"Access token: {access_token}")
