import asyncio
import os
import msgpack
import redis.asyncio as redis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="S.C.I.T.T.S.E. Telemetry Gateway")

raw_cors_origins = os.getenv("CORS_ALLOW_ORIGINS", "").strip()
if raw_cors_origins:
    allowed_origins = [origin.strip() for origin in raw_cors_origins.split(",") if origin.strip()]
else:
    # Local dev defaults for Vite/preview and common React dev ports.
    allowed_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

redis_client = redis.Redis(host='localhost', port=6379, db=0)

@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("📡 WebSocket client connected for telemetry stream.")
    pubsub = redis_client.pubsub()
    await pubsub.subscribe('race:telemetry')
    try:
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.1)
            if message:
                unpacked_data = msgpack.unpackb(message['data'])
                await websocket.send_json(unpacked_data)   
            else:
                await asyncio.sleep(0.01) 
    except WebSocketDisconnect:
        print("📴 WebSocket client disconnected from telemetry stream.")
    finally:
        await pubsub.unsubscribe('race:telemetry')