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
        1:1 Single chat에 카드 형태 메시지 전송

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
                                "text": "GATE Engine Bot 입니다."
                            }
                        },
                        {
                            "element_type": "description",
                            "description": {
                                "format": 1,
                                "text": 
                                    f"{reply_text or ''}\n\n⭐번역결과가 마음에 드신다면 'GOOD'을, 개선이 필요하다면 'BAD'를 눌러주세요⭐\n사용자의 피드백이 많아야 엔진학습에 도움이 됩니다.😊"
                            }
                        },
                        {
                            "element_type": "button",
                            "button": {
                                "button_type": "callback",
                                "text": "👍GOOD",
                                "value": "GOOD",
                            }
                        },
                        {
                            "element_type": "button",
                            "button": {
                                "button_type": "callback",
                                "text": "👎BAD",
                                "value": "BAD",
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
    
    async def send_group_message(self, group_id: str, reply_text: str, message_id: str = None, thread_id: str = None):
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
                                        f"{reply_text or ''}\n\n⭐번역결과가 마음에 드신다면 'GOOD'을, 개선이 필요하다면 'BAD'를 눌러주세요⭐\n사용자의 피드백이 많아야 엔진학습에 도움이 됩니다.😊"
                                }
                            },
                            {
                                "element_type": "button",
                                "button": {
                                    "button_type": "callback",
                                    "text": "👍GOOD",
                                    "value": "GOOD",
                                }
                            },
                            {
                                "element_type": "button",
                                "button": {
                                    "button_type": "callback",
                                    "text": "👎BAD",
                                    "value": "BAD",
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
                                    "text": "GATE Engine Bot 입니다."
                                }
                            },
                            {
                                "element_type": "description",
                                "description": {
                                    "format": 1,
                                    "text": 
                                        f"{reply_text or ''}\n\n⭐번역결과가 마음에 드신다면 'GOOD'을, 개선이 필요하다면 'BAD'를 눌러주세요⭐\n사용자의 피드백이 많아야 엔진학습에 도움이 됩니다.😊"
                                }
                            },
                            {
                                "element_type": "button",
                                "button": {
                                    "button_type": "callback",
                                    "text": "👍GOOD",
                                    "value": "GOOD",
                                }
                            },
                            {
                                "element_type": "button",
                                "button": {
                                    "button_type": "callback",
                                    "text": "👎BAD",
                                    "value": "BAD",
                                }
                            },
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
