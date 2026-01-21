import httpx, logging
from .seatalk_auth import seatalk_auth


class SeaTalkClient:
    def __init__(self, base_url: str = "https://openapi.seatalk.io"):
        self.baese_url = base_url
        
    async def _headers(self) -> dict:
        token = await seatalk_auth.get_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
    async def send_text_message(self, e_code: str, text: str):
        """
        1:1 Single chat에 텍스트 메시지 전송 (번역 결과용)

        Args:
            e_code (str): callback으로 전송된 employee_code
            text (str): 전송할 텍스트 메시지

        Returns:
            dict : client.post로 message를 보낸 후의 json을 python dict로 반환
        """
        logging.info(f"Sending text message to employee_code: {e_code}")
        url = f"{self.baese_url}/messaging/v2/single_chat"
        payload = {
            "employee_code": e_code,
            "message": {
                "tag": "text",
                "text": {
                    "content": text
                }
            }
        }
        headers = await self._headers()
        async with httpx.AsyncClient(timeout=30) as client:
            res = await client.post(url, json=payload, headers=headers)
            res.raise_for_status()
            body = res.json()
            logging.info(f"[SeaTalk] Text Message Response: {body}")
            return body
    
    async def send_card_message(self, e_code: str, reply_text: str = None):
        """
        1:1 Single chat에 카드 형태 메시지 전송 (Google Sheet URL 감지 시)

        Args:
            e_code (str): callback으로 전송된 employee_code
            reply_text (str): 추가 안내 메시지

        Returns:
            dict : client.post로 message를 보낸 후의 json을 python dict로 반환
        """
        logging.info(f"Sending card message to employee_code: {e_code}")
        url = f"{self.baese_url}/messaging/v2/single_chat"
        employee_code = f"{e_code}"
        payload = {
            "employee_code": employee_code,
            "message": {
                "tag": "interactive_message",
                "interactive_message": {
                    "elements": [
                        {
                            "element_type": "title",
                            "title": {
                                "text": "문서 번역 지원 Bot 입니다."
                            }
                        },
                        {
                            "element_type": "description",
                            "description": {
                                "format": 1,
                                "text": 
                                    f"## Guide\n- 아래 버튼 클릭 시, 번역Web App으로 이동합니다.\n{reply_text or ''}\n\n## 참고사항\n1. word, excel, pdf, pdf만 가능\n2. 가급적 Text위주로 된 문서를 번역\n3. 50Mb이하의 문서만 가능\n4. 이미지 및 표 안의 text는 현재 번역이 불완전할 수 있음"
                            }
                        },
                        {
                            "element_type": "button",
                            "button": {
                                "button_type": "redirect",
                                "text": "문서번역 페이지로 이동",
                                "mobile_link": {
                                    "type": "web",
                                    "path": "https://ai.insea.io/app/chatflows/9482"
                                },
                                "desktop_link": {
                                    "type": "web",
                                    "path": "https://ai.insea.io/app/chatflows/9482"
                                }
                            }
                        }
                    ]
                }
            }
        }
        headers = await self._headers()
        async with httpx.AsyncClient(timeout=10) as client:
            res = await client.post(url, json=payload, headers=headers)
            res.raise_for_status()
            body = res.json()
            logging.info(f"[SeaTalk] Card Message Response: {body}")
            return body
    
    async def send_message(self, e_code: str, reply_text: str = None):
        """
        1:1 Single chat 전송 (기존 호환용 - send_card_message로 위임)

        Args:
            e_code (str): callback으로 전송된 emplaoyee_code

        Returns:
            dict : client.post로 message를 보낸 후의 json을 python dict로 반환
        """
        return await self.send_card_message(e_code, reply_text)
    
    
    async def send_group_text_message(self, group_id: str, text: str, message_id: str = None, thread_id: str = None):
        """
        그룹 채팅방에 텍스트 메시지 전송 (번역 결과용)

        Args:
            group_id (str): 그룹 채팅방 ID
            text (str): 전송할 텍스트 메시지
            message_id (str): 원본 메시지 ID
            thread_id (str): 스레드 ID

        Returns:
            dict: client.post로 message를 보낸 후의 json을 python dict로 반환
        """
        logging.info(f"Sending group text message to group_id: {group_id}")
        url = f"{self.baese_url}/messaging/v2/group_chat"
        headers = await self._headers()
        
        reply_thread_id = thread_id if (thread_id and message_id != thread_id) else message_id
        
        payload = {
            "group_id": group_id,
            "message": {
                "tag": "text",
                "text": {
                    "content": text
                },
                "thread_id": reply_thread_id
            }
        }
        async with httpx.AsyncClient(timeout=30) as client:
            res = await client.post(url, json=payload, headers=headers)
            res.raise_for_status()
            body = res.json()
            logging.info(f"[SeaTalk] Group Text Message Response: {body}")
            return body
    
    async def send_group_message(self, group_id: str, message_id: str = None, thread_id: str = None):
        """
        그룹 채팅방에 카드 형태 메시지 전송

        Args:
            group_id (str): 그룹 채팅방 ID

        Returns:
            dict: client.post로 message를 보낸 후의 json을 python dict로 반환
        """
        url = f"{self.baese_url}/messaging/v2/group_chat"
        
        headers = await self._headers()
        if not thread_id or message_id==thread_id: # thread_id가 없거나, thread_id가 message_id와 동일한 경우 (즉, 최상위 메시지에 대한 멘션인 경우)
            payload_with_message = {
                "group_id": group_id,
                "message": {
                    "tag": "interactive_message",
                    "interactive_message": {
                        "elements": [
                            {
                                "element_type": "title",
                                "title": {
                                    "text": "GATE Engine Bot 입니다."
                                }
                            },
                            {
                                "element_type": "description",
                                "description": {
                                    "format": 1,
                                    "text": 
                                        "## Guide\n- 아래 버튼 클릭 시, 요청하신 Google Sheets로 이동합니다."
                                }
                            },
                            {
                                "element_type": "button",
                                "button": {
                                    "button_type": "redirect",
                                    "text": "페이지로 이동",
                                    "mobile_link": {
                                        "type": "web",
                                        "path": "https://www.google.com/"
                                    },
                                    "desktop_link": {
                                        "type": "web",
                                        "path": "https://www.google.com/"
                                    }
                                }
                            }
                        ]
                    },
                    "thread_id": message_id
                }
            }
            async with httpx.AsyncClient(timeout=10) as client:
                res = await client.post(url, json=payload_with_message, headers=headers)
                res.raise_for_status()
                body = res.json()
                logging.info(f"[SeaTalk] Group Message Response: {body}")
                return body
        
        else:
            payload_with_thread = {
                "group_id": group_id,
                "message": {
                    "tag": "interactive_message",
                    "interactive_message": {
                        "elements": [
                            {
                                "element_type": "title",
                                "title": {
                                    "text": "문서 번역 지원 Bot 입니다."
                                }
                            },
                            {
                                "element_type": "description",
                                "description": {
                                    "format": 1,
                                    "text": 
                                        "## Guide\n- 아래 버튼 클릭 시, 번역Web App으로 이동합니다.\n\n## 참고사항\n1. word, excel, pdf, pdf만 가능\n2. 가급적 Text위주로 된 문서를 번역\n3. 50Mb이하의 문서만 가능\n4. 이미지 및 표 안의 text는 현재 번역이 불완전할 수 있음"
                                }
                            },
                            {
                                "element_type": "button",
                                "button": {
                                    "button_type": "redirect",
                                    "text": "문서번역 페이지로 이동",
                                    "mobile_link": {
                                        "type": "web",
                                        "path": "https://ai.insea.io/app/chatflows/9482"
                                    },
                                    "desktop_link": {
                                        "type": "web",
                                        "path": "https://ai.insea.io/app/chatflows/9482"
                                    }
                                }
                            }
                        ]
                    },
                    "thread_id": thread_id
                }
            }
            async with httpx.AsyncClient(timeout=10) as client:
                res = await client.post(url, json=payload_with_thread, headers=headers)
                res.raise_for_status()
                body = res.json()
                logging.info(f"[SeaTalk] Group Message (with thread) Response: {body}")
                return body
    

seatalk_client = SeaTalkClient()
