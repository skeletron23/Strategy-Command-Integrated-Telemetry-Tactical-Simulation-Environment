import asyncio
import msgpack
import redis.asyncio as redis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI(title="S.C.I.T.T.S.E. Telemetry Gateway")

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