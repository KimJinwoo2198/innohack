from __future__ import annotations

import asyncio
import logging
from typing import Dict, List
from urllib.parse import parse_qs
from uuid import uuid4

from asgiref.sync import sync_to_async
from channels.generic.websocket import AsyncJsonWebsocketConsumer

from vision.services.rag_guidance import (
    generate_food_chat_reply,
    get_food_guidance,
    CHAT_HISTORY_LIMIT,
)

logger = logging.getLogger(__name__)


class FoodChatConsumer(AsyncJsonWebsocketConsumer):
    """
    인식된 음식 정보를 바탕으로 임산부가 추가 질문을 던질 수 있는 WebSocket 채팅 컨슈머.
    """

    async def connect(self) -> None:
        params = self._parse_query_params()
        self.food_name: str = (params.get("food") or params.get("food_name") or "").strip()
        self.dialect_style: str = (params.get("dialect_style") or "standard").strip() or "standard"
        self.session_id: str = params.get("session_id") or str(uuid4())
        self.chat_history: List[Dict[str, str]] = []
        self.base_guidance: Dict[str, str] | None = None
        self.is_authenticated_user = bool(
            self.scope.get("user") and getattr(self.scope["user"], "is_authenticated", False)
        )
        # 쿼리 파라미터에서 임산부 주차를 받음 (week 또는 pregnancy_week)
        # 기본값: 20주 (2분기 중기)
        week_str = (params.get("week") or params.get("pregnancy_week") or "20").strip()
        try:
            self.pregnancy_week: int = int(week_str)
            if not (1 <= self.pregnancy_week <= 42):
                self.pregnancy_week = 20
        except ValueError:
            self.pregnancy_week = 20

        if not self.food_name:
            await self.close(code=4400)
            return

        await self.accept()
        await self.send_json(
            {
                "type": "chat.connected",
                "session_id": self.session_id,
                "food_name": self.food_name,
                "is_authenticated": self.is_authenticated_user,
                "pregnancy_week": self.pregnancy_week,
            }
        )
        asyncio.create_task(self._hydrate_baseline())

    async def disconnect(self, code: int) -> None:  # pragma: no cover - channels lifecycle
        logger.debug("FoodChatConsumer disconnected: code=%s session=%s", code, getattr(self, "session_id", None))

    async def receive_json(self, content: Dict, **kwargs) -> None:
        event_type = content.get("type") or "message"
        if event_type == "ping":
            await self.send_json({"type": "pong"})
            return
        if event_type == "user.message":
            await self._handle_user_message(content)
            return
        await self.send_json({"type": "error", "message": f"지원하지 않는 이벤트입니다: {event_type}"})

    def _parse_query_params(self) -> Dict[str, str]:
        raw = self.scope.get("query_string", b"")
        if not raw:
            return {}
        parsed = parse_qs(raw.decode("utf-8"))
        return {key: values[-1] for key, values in parsed.items() if values}

    async def _hydrate_baseline(self) -> None:
        try:
            await self.send_json({"type": "chat.status", "status": "initializing"})
        except RuntimeError as e:
            # WebSocket이 이미 종료된 경우
            logger.debug("WebSocket이 종료됨, baseline 전송 스킵: %s", e)
            return

        try:
            logger.info("Baseline 로딩 시작: food=%s, user=%s", self.food_name, self.scope.get("user"))
            self.base_guidance = await sync_to_async(
                get_food_guidance, thread_sensitive=True
            )(self.scope["user"], self.food_name, self.dialect_style)
            logger.info("Baseline 로딩 완료: %s", self.base_guidance)
            try:
                await self.send_json(
                    {
                        "type": "chat.baseline",
                        "food_name": self.food_name,
                        "guidance": self.base_guidance,
                    }
                )
            except RuntimeError as e:
                # baseline 전송 중 연결이 끝난 경우, 무시
                logger.debug("Baseline 전송 실패: WebSocket 연결 종료: %s", e)
        except Exception as exc:  # pragma: no cover - 방어적 로깅
            logger.error("기본 가이드 불러오기 실패: %s", exc, exc_info=True)
            try:
                await self.send_json(
                    {
                        "type": "assistant.error",
                        "code": "baseline_failed",
                        "message": "기본 안전 정보를 불러오지 못했습니다. 다시 시도해주세요.",
                    }
                )
            except RuntimeError as e:
                # 오류 메시지 전송도 실패한 경우, 조용히 넘김
                logger.debug("오류 메시지 전송 실패: WebSocket 이미 종료됨: %s", e)

    async def _handle_user_message(self, content: Dict) -> None:
        if not self.base_guidance:
            await self.send_json(
                {
                    "type": "assistant.error",
                    "code": "guidance_unavailable",
                    "message": "기본 안전 정보를 아직 로딩 중입니다. 잠시 후 다시 시도해주세요.",
                }
            )
            return

        message = (content.get("message") or "").strip()
        if not message:
            await self.send_json(
                {
                    "type": "assistant.error",
                    "code": "empty_message",
                    "message": "질문 내용을 입력해 주세요.",
                }
            )
            return

        trace_id = content.get("trace_id") or str(uuid4())
        await self.send_json({"type": "assistant.status", "status": "processing", "trace_id": trace_id})
        history_snapshot = list(self.chat_history)

        try:
            reply = await sync_to_async(
                generate_food_chat_reply, thread_sensitive=True
            )(
                self.scope["user"],
                self.food_name,
                self.dialect_style,
                message,
                history_snapshot,
                self.base_guidance,
                self.pregnancy_week,
            )
        except ValueError as exc:
            await self.send_json(
                {
                    "type": "assistant.error",
                    "code": "invalid_message",
                    "message": str(exc),
                    "trace_id": trace_id,
                }
            )
            return
        except Exception as exc:  # pragma: no cover - 방어적 로깅
            logger.exception("Food chat 응답 생성 실패: %s", exc)
            await self.send_json(
                {
                    "type": "assistant.error",
                    "code": "generation_failed",
                    "message": "답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
                    "trace_id": trace_id,
                }
            )
            return

        self._append_history("user", message)
        self._append_history("assistant", reply.get("answer", ""))

        await self.send_json(
            {
                "type": "assistant.reply",
                "trace_id": trace_id,
                "answer": reply.get("answer"),
                "references": reply.get("references", []),
                "retrieved_snippets": reply.get("retrieved_snippets", []),
            }
        )

    def _append_history(self, role: str, message: str) -> None:
        if not message:
            return
        self.chat_history.append({"role": role, "message": message})
        if len(self.chat_history) > CHAT_HISTORY_LIMIT * 2:
            self.chat_history = self.chat_history[-CHAT_HISTORY_LIMIT * 2 :]


