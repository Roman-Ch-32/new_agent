# src/main.py
import json
import re
import traceback
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from langchain_core.messages import HumanMessage

from agent.config import config


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan контекст"""
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

            print(f"📩 Получено сообщение: {msg[:50]}...")

            # === Запуск LangGraph агента ===
            result = app.state.agent.invoke(
                messages=[HumanMessage(content=msg)],
                session_id=session_id,
            )

            print(f"📤 Получен ответ: {result.result[:50] if result.result else 'EMPTY'}...")

            response_text = result.result if result.result else "Нет ответа"

            parsed = parse_reasoning(response_text)

            await websocket.send_json({
                "type": "reply",
                "reply": parsed["answer"],
                "reasoning": parsed["reasoning"],
                "has_reasoning": parsed["has_reasoning"]
            })

        except WebSocketDisconnect:
            print(f"🔌 Клиент отключился (session: {session_id})")
            break

        except Exception as e:
            # ✅ Полное логирование ошибки
            error_msg = str(e)
            error_trace = traceback.format_exc()

            print(f"\n❌ ОШИБКА: {error_msg}")
            print(f"📋 TRACEBACK:\n{error_trace}\n")

            try:
                await websocket.send_json({
                    "type": "error",
                    "message": error_msg if error_msg else "Неизвестная ошибка (см. логи сервера)",
                    "traceback": error_trace  # Для отладки
                })
            except Exception as send_error:
                print(f"⚠️ Не удалось отправить ошибку клиенту: {send_error}")
                break


def parse_reasoning(text: str) -> dict:
    """Разделяет рассуждения и финальный ответ"""
    reasoning = re.search(r'<reasoning>(.*?)</reasoning>', text, re.DOTALL)
    answer = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)

    return {
        "reasoning": reasoning.group(1).strip() if reasoning else "",
        "answer": answer.group(1).strip() if answer else text,
        "has_reasoning": bool(reasoning)
    }
