# src/main.py
"""Main — FastAPI + WebSocket + Стриминг"""

import json
import re
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from langchain_core.messages import HumanMessage

from agent.config import config


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    print("🚀 Запуск приложения...")
    from agent.agent import Agent
    app.state.agent = Agent()
    print("✅ Агент готов")
    yield
    print("🛑 Остановка приложения...")


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "model": config.llm.model}


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({"type": "status", "status": "✅ Подключено"})

    session_id = "default"

    while True:
        try:
            data = await websocket.receive_text()
            payload = json.loads(data)
            msg = payload.get("message", "")

            if not msg:
                continue

            # Отправляем что начали обработку
            await websocket.send_json({
                "type": "reasoning_start",
                "status": "🤔 Думаю..."
            })

            # Вызываем агента (со стримингом если поддерживается)
            result = app.state.agent.invoke(
                messages=[HumanMessage(content=msg)],
                session_id=session_id,
            )

            # Извлекаем рассуждения и ответ
            response_text = result.result

            reasoning = ""
            reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response_text, re.DOTALL | re.IGNORECASE)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()

            answer = ""
            answer_match = re.search(r'<answer>(.*?)</answer>|<reply>(.*?)</reply>', response_text,
                                     re.DOTALL | re.IGNORECASE)
            if answer_match:
                answer = answer_match.group(1) or answer_match.group(2) or response_text
            else:
                answer = response_text

            # Отправляем рассуждения
            if reasoning:
                await websocket.send_json({
                    "type": "reasoning",
                    "reasoning": reasoning,
                    "has_reasoning": True
                })

            # Отправляем ответ
            await websocket.send_json({
                "type": "reply",
                "reply": answer,
                "reasoning": reasoning,
                "has_reasoning": bool(reasoning)
            })

        except WebSocketDisconnect:
            print(f"🔌 Клиент отключился")
            break

        except Exception as e:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
