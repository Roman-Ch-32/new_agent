"""FastAPI + WebSocket entrypoint with trace streaming."""

import json
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from langchain_core.messages import HumanMessage

from agent.config import config


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    from agent.agent import Agent

    app.state.agent = Agent()
    yield


app = FastAPI(lifespan=lifespan)


@app.get('/health')
async def health():
    return {'status': 'ok', 'model': config.llm.model}


@app.websocket('/ws')
async def ws(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({'type': 'status', 'status': '✅ Подключено'})

    session_id = 'default'

    while True:
        try:
            payload = json.loads(await websocket.receive_text())
            msg = payload.get('message', '')
            if not msg:
                continue

            await websocket.send_json({'type': 'status', 'status': '🤔 Планирую и выполняю...'})

            result = app.state.agent.invoke(messages=[HumanMessage(content=msg)], session_id=session_id)

            for event in result.trace_events:
                await websocket.send_json(event)

            await websocket.send_json({
                'type': 'reply',
                'reply': result.result,
                'verification': result.verification_result,
            })
        except WebSocketDisconnect:
            break
        except Exception as e:
            await websocket.send_json({'type': 'error', 'message': str(e)})
