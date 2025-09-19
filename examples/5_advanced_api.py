"""Example from Advanced Usage docs.

This example shows how kani can be served over an API. You'll need to install FastAPI, websockets, and an ASGI server
to run this example::

    pip install fastapi websockets uvicorn

In this example, we implement a simple websocket endpoint that allows a user to chat using WebSockets.
"""

import os

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from kani import Kani
from kani.engines.openai import OpenAIEngine

# initialize kani
api_key = os.getenv("OPENAI_API_KEY")
engine = OpenAIEngine(api_key, model="gpt-4o-mini")

# initialize FastAPI
app = FastAPI()


@app.websocket("/chat")
async def kani_chat(websocket: WebSocket):
    # accept the websocket and initialize a kani for the connection
    await websocket.accept()
    ai = Kani(engine)
    # take string messages and send string responses
    while True:
        try:
            data = await websocket.receive_text()
            resp = await ai.chat_round_str(data)
            await websocket.send_text(resp)
        # until the client disconnects
        except WebSocketDisconnect:
            return


@app.on_event("shutdown")
async def cleanup_kani():
    """When the application shuts down, cleanly close the kani engine."""
    await engine.close()


if __name__ == "__main__":
    uvicorn.run(app)
